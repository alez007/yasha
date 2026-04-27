import asyncio
import inspect
import json
import os
from collections.abc import AsyncGenerator

from llama_cpp import Llama

from modelship.infer.base_infer import BaseInfer
from modelship.infer.infer_config import LlamaCppConfig, ModelshipModelConfig, ModelUsecase, RawRequestProxy
from modelship.logging import get_logger
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    create_error_response,
)

logger = get_logger("infer.llama_cpp")


class LlamaCppInfer(BaseInfer):
    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)
        self.config = model_config.llama_cpp_config or LlamaCppConfig()

        # Automatically enable verbose mode if MSHIP_LOG_LEVEL is TRACE.
        # Other log levels are handled via the 'llama_cpp' Python logger (configured in logging.py).
        mship_log_level = os.environ.get("MSHIP_LOG_LEVEL", "INFO").upper()
        self._verbose = mship_log_level == "TRACE"

        # Force CPU-only as llama_cpp is currently compiled without GPU support in this environment.
        if self.config.n_gpu_layers != 0:
            logger.warning(
                "n_gpu_layers=%s is ignored for model '%s': llama_cpp currently only supports CPU.",
                self.config.n_gpu_layers,
                self.model_config.name,
            )
        self._n_gpu_layers = 0

        self.llamacpp: Llama | None = None
        self._chat_params: set[str] = set()
        self._embed_params: set[str] = set()
        self._lock = asyncio.Lock()  # llama-cpp-python Llama object is not thread-safe for concurrent calls
        logger.info(
            "initialising llama.cpp engine (verbose=%s) with config: %s",
            self._verbose,
            self.config.model_dump(),
        )

    def shutdown(self) -> None:
        if self.llamacpp:
            logger.info("Shutting down llama.cpp engine for %s", self.model_config.name)
            # llama-cpp-python relies on __del__ for resource cleanup.
            del self.llamacpp
            self.llamacpp = None

    def __del__(self):
        self.shutdown()

    async def start(self) -> None:
        logger.info("Start llama.cpp infer for model: %s", self.model_config)
        loop = asyncio.get_event_loop()

        # Initialize Llama in a thread pool as it can be slow/blocking
        if self.config.hf_filename:
            logger.info(
                "Loading llama.cpp model from Hugging Face: repo=%s, file=%s",
                self.model_config.model,
                self.config.hf_filename,
            )
            self.llamacpp = await loop.run_in_executor(
                None,
                lambda: Llama.from_pretrained(
                    repo_id=self.model_config.model,
                    filename=self.config.hf_filename,
                    n_gpu_layers=self._n_gpu_layers,
                    n_ctx=self.config.n_ctx,
                    n_batch=self.config.n_batch,
                    chat_format=self.config.chat_format,
                    verbose=self._verbose,
                    embedding=self.model_config.usecase == ModelUsecase.embed,
                    **self.config.model_kwargs,
                ),
            )
        else:
            self.llamacpp = await loop.run_in_executor(
                None,
                lambda: Llama(
                    model_path=self.model_config.model,
                    n_gpu_layers=self._n_gpu_layers,
                    n_ctx=self.config.n_ctx,
                    n_batch=self.config.n_batch,
                    chat_format=self.config.chat_format,
                    verbose=self._verbose,
                    embedding=self.model_config.usecase == ModelUsecase.embed,
                    **self.config.model_kwargs,
                ),
            )
        self._set_max_context_length(self.config.n_ctx)

        # Inspect and cache parameter lists once
        if self.llamacpp:
            self._chat_params = set(inspect.signature(self.llamacpp.create_chat_completion).parameters.keys())
            self._embed_params = set(inspect.signature(self.llamacpp.create_embedding).parameters.keys())

    async def warmup(self) -> None:
        if not self.llamacpp:
            return

        logger.info("Warming up llama.cpp model: %s", self.model_config.name)
        dummy_proxy = RawRequestProxy(None, {})

        if self.model_config.usecase == ModelUsecase.generate:
            request = ChatCompletionRequest(
                model=self.model_config.name,
                messages=[{"role": "user", "content": "warmup"}],
                max_tokens=1,
            )
            await self.create_chat_completion(request, dummy_proxy)
            logger.info("Warmup chat completion done for %s", self.model_config.name)
        elif self.model_config.usecase == ModelUsecase.embed:
            request = EmbeddingRequest(
                model=self.model_config.name,
                input="warmup",
            )
            await self.create_embedding(request, dummy_proxy)
            logger.info("Warmup embedding done for %s", self.model_config.name)

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ChatCompletionResponse | AsyncGenerator[str, None]:
        if not self.llamacpp or self.model_config.usecase != ModelUsecase.generate:
            return await super().create_chat_completion(request, raw_request)

        request_id = raw_request.request_id
        logger.info("chat completion request %s: stream=%s", request_id, request.stream)

        loop = asyncio.get_event_loop()
        llamacpp = self.llamacpp
        assert llamacpp is not None

        # Pre-process messages: llama-cpp-python's Jinja templates often expect 'content'
        # to be a string. If it's a list (OpenAI multi-modal format), flatten it to text.
        messages = []
        for msg in request.messages:
            processed_msg = msg.copy()
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = [part["text"] for part in content if isinstance(part, dict) and part.get("type") == "text"]
                processed_msg["content"] = " ".join(text_parts)
            messages.append(processed_msg)

        all_params = request.model_dump(exclude_none=True)
        all_params["messages"] = messages

        # Handle 'max_completion_tokens' mapping to 'max_tokens' if needed.
        if "max_tokens" not in all_params and "max_completion_tokens" in all_params:
            all_params["max_tokens"] = all_params["max_completion_tokens"]

        kwargs = {k: v for k, v in all_params.items() if k in self._chat_params and k != "stream"}

        if request.stream:

            async def stream_generator() -> AsyncGenerator[str, None]:
                async with self._lock:
                    # Run the synchronous generator in a thread
                    iterator = await loop.run_in_executor(
                        None,
                        lambda: llamacpp.create_chat_completion(**kwargs, stream=True),  # type: ignore
                    )
                    for chunk in iterator:
                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(0)
                    yield "data: [DONE]\n\n"

            return stream_generator()
        else:
            async with self._lock:
                try:
                    result = await loop.run_in_executor(
                        None,
                        lambda: llamacpp.create_chat_completion(**kwargs, stream=False),  # type: ignore
                    )
                    return ChatCompletionResponse.model_validate(result)
                except Exception as e:
                    logger.warning("llama_cpp chat inference failed: %s", e)
                    return create_error_response(e)

    async def create_embedding(
        self, request: EmbeddingRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | EmbeddingResponse:
        if not self.llamacpp or self.model_config.usecase != ModelUsecase.embed:
            return await super().create_embedding(request, raw_request)

        request_id = raw_request.request_id
        logger.info("embedding request %s", request_id)

        loop = asyncio.get_event_loop()
        llamacpp = self.llamacpp
        assert llamacpp is not None

        all_params = request.model_dump(exclude_none=True)
        kwargs = {k: v for k, v in all_params.items() if k in self._embed_params}

        async with self._lock:
            try:
                result = await loop.run_in_executor(None, lambda: llamacpp.create_embedding(**kwargs))
                return EmbeddingResponse.model_validate(result)
            except Exception as e:
                logger.warning("llama_cpp embedding failed: %s", e)
                return create_error_response(e)
