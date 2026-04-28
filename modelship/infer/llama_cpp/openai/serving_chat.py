import asyncio
import inspect
import json
from collections.abc import AsyncGenerator

from llama_cpp import Llama

from modelship.infer.base_serving import OpenAIServing
from modelship.infer.infer_config import RawRequestProxy
from modelship.infer.llama_cpp.capabilities import LlamaCppCapabilities
from modelship.logging import get_logger
from modelship.openai.chat_utils import UnsupportedContentError, normalize_chat_messages
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    create_error_response,
)
from modelship.utils import base_request_id

logger = get_logger("infer.llama_cpp.chat")


class OpenAIServingChat(OpenAIServing):
    request_id_prefix = "chat"

    def __init__(self, llama: Llama, model_name: str, capabilities: LlamaCppCapabilities):
        self._llama = llama
        self.model_name = model_name
        self._caps = capabilities
        self._lock = asyncio.Lock()
        self._accepted_params = set(inspect.signature(llama.create_chat_completion).parameters)

    async def warmup(self) -> None:
        logger.info("Warming up llama.cpp chat model: %s", self.model_name)
        request = ChatCompletionRequest(
            model=self.model_name,
            messages=[{"role": "user", "content": "warmup"}],
            max_tokens=1,
        )
        await self.create_chat_completion(request, RawRequestProxy(None, {}))
        logger.info("Warmup chat done for %s", self.model_name)

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ChatCompletionResponse | AsyncGenerator[str, None]:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info("chat completion request %s: stream=%s", request_id, request.stream)

        try:
            messages = normalize_chat_messages(
                request.messages,
                supports_image=self._caps.supports_image,
                supports_audio=self._caps.supports_audio,
            )
        except UnsupportedContentError as e:
            logger.warning("chat request %s rejected: %s", request_id, e)
            return create_error_response(e)

        kwargs = self._build_kwargs(request, messages)
        loop = asyncio.get_event_loop()
        llama = self._llama

        if request.stream:

            async def stream_generator() -> AsyncGenerator[str, None]:
                async with self._lock:
                    iterator = await loop.run_in_executor(
                        None,
                        lambda: llama.create_chat_completion(**kwargs, stream=True),  # type: ignore[arg-type]
                    )
                    while True:
                        chunk = await loop.run_in_executor(None, next, iterator, None)
                        if chunk is None:
                            break
                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(0)
                    yield "data: [DONE]\n\n"

            return stream_generator()

        async with self._lock:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: llama.create_chat_completion(**kwargs, stream=False),  # type: ignore[arg-type]
                )
                return ChatCompletionResponse.model_validate(result)
            except Exception as e:
                logger.warning("llama_cpp chat inference failed: %s", e)
                return create_error_response(e)

    def _build_kwargs(self, request: ChatCompletionRequest, messages: list[dict]) -> dict:
        params = request.model_dump(exclude_none=True)
        params["messages"] = messages
        if "max_tokens" not in params and "max_completion_tokens" in params:
            params["max_tokens"] = params["max_completion_tokens"]

        kwargs: dict = {}
        dropped: list[str] = []
        for k, v in params.items():
            if k == "stream":
                continue
            if k in self._accepted_params:
                kwargs[k] = v
            else:
                dropped.append(k)
        if dropped:
            logger.warning(
                "llama_cpp: dropping request params not supported by create_chat_completion: %s",
                dropped,
            )
        return kwargs
