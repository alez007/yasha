import asyncio
import json
import time
from collections.abc import AsyncGenerator
from threading import Thread

from transformers import Pipeline, PreTrainedTokenizerBase, TextIteratorStreamer

from modelship.infer.base_serving import OpenAIServing
from modelship.infer.infer_config import RawRequestProxy, TransformersConfig
from modelship.infer.transformers.capabilities import TransformersCapabilities
from modelship.logging import TRACE, get_logger
from modelship.openai.chat_utils import UnsupportedContentError, normalize_chat_messages
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ErrorResponse,
    UsageInfo,
    create_error_response,
)
from modelship.utils import base_request_id

logger = get_logger("infer.transformers.chat")


class OpenAIServingChat(OpenAIServing):
    request_id_prefix = "chatcmpl"

    def __init__(
        self,
        pipeline: Pipeline,
        model_name: str,
        config: TransformersConfig,
        capabilities: TransformersCapabilities,
    ):
        self.pipeline = pipeline
        self.model_name = model_name
        self.config = config
        self.capabilities = capabilities
        assert pipeline.tokenizer is not None, "text-generation pipeline must have a tokenizer"
        self.tokenizer: PreTrainedTokenizerBase = pipeline.tokenizer
        self._lock = asyncio.Lock()

    async def warmup(self) -> None:
        logger.info("Warming up chat model: %s", self.model_name)
        await self.run_in_executor(
            self._run,
            [{"role": "user", "content": "warmup"}],
            1,
        )
        logger.info("Warmup chat done for %s", self.model_name)

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: RawRequestProxy
    ) -> ChatCompletionResponse | AsyncGenerator[str, None] | ErrorResponse:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info("chat completion request %s: stream=%s", request_id, request.stream)
        logger.log(
            TRACE, "chat request %s: messages=%s, max_tokens=%s", request_id, request.messages, request.max_tokens
        )

        try:
            messages = normalize_chat_messages(
                request.messages,
                supports_image=self.capabilities.supports_image,
                supports_audio=self.capabilities.supports_audio,
            )
        except UnsupportedContentError as e:
            logger.warning("chat request %s rejected: %s", request_id, e)
            return create_error_response(e)

        max_tokens = request.max_tokens
        if max_tokens is None and request.max_completion_tokens is not None:
            max_tokens = request.max_completion_tokens

        if request.stream:
            include_usage = bool(request.stream_options and request.stream_options.include_usage)
            return self._locked_stream(request_id, messages, max_tokens, include_usage=include_usage)

        async with self._lock:
            try:
                result = await self.run_in_executor(self._run, messages, max_tokens)
            except Exception:
                logger.exception("chat completion inference failed for %s", request_id)
                return create_error_response("chat completion inference failed")

        prompt_tokens = self._count_prompt_tokens(messages)
        generated = result[0]["generated_text"]
        completion_text = generated[-1]["content"] if isinstance(generated, list) else generated
        completion_tokens = len(self.tokenizer.encode(completion_text, add_special_tokens=False))
        finish_reason = "length" if (max_tokens is not None and completion_tokens >= max_tokens) else "stop"

        response = ChatCompletionResponse(
            id=request_id,
            model=self.model_name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=completion_text),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            created=int(time.time()),
        )
        logger.log(
            TRACE,
            "chat response %s: text=%r, prompt_tokens=%d, completion_tokens=%d",
            request_id,
            completion_text,
            prompt_tokens,
            completion_tokens,
        )
        return response

    def _count_prompt_tokens(self, messages: list[dict]) -> int:
        # apply_chat_template returns a string by default (character count!) — force tokenize=True.
        token_ids = self.tokenizer.apply_chat_template(messages, tokenize=True)
        return len(token_ids)

    def _run(self, messages: list[dict], max_tokens: int | None) -> list:
        kwargs = {**self.config.pipeline_kwargs}
        if max_tokens is not None:
            kwargs["max_new_tokens"] = max_tokens
        return self.pipeline(messages, return_full_text=False, **kwargs)  # type: ignore[return-value]

    async def _locked_stream(
        self,
        request_id: str,
        messages: list[dict],
        max_tokens: int | None,
        *,
        include_usage: bool,
    ) -> AsyncGenerator[str, None]:
        async with self._lock:
            async for chunk in self._stream(request_id, messages, max_tokens, include_usage=include_usage):
                yield chunk

    async def _stream(
        self,
        request_id: str,
        messages: list[dict],
        max_tokens: int | None,
        *,
        include_usage: bool,
    ) -> AsyncGenerator[str, None]:
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore[arg-type]

        kwargs = {**self.config.pipeline_kwargs}
        if max_tokens is not None:
            kwargs["max_new_tokens"] = max_tokens

        thread = Thread(
            target=self.pipeline,
            args=(messages,),
            kwargs={
                "streamer": streamer,
                "return_full_text": False,
                **kwargs,
            },
        )
        thread.start()

        accumulated: list[str] = []
        try:
            # Per OpenAI spec, the first delta carries `role` only.
            yield self._encode_chunk(
                ChatCompletionStreamResponse(
                    id=request_id,
                    model=self.model_name,
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=0,
                            delta=DeltaMessage(role="assistant"),
                        )
                    ],
                    created=int(time.time()),
                )
            )

            for text_chunk in streamer:
                if not text_chunk:
                    continue
                accumulated.append(text_chunk)
                yield self._encode_chunk(
                    ChatCompletionStreamResponse(
                        id=request_id,
                        model=self.model_name,
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=0,
                                delta=DeltaMessage(content=text_chunk),
                            )
                        ],
                        created=int(time.time()),
                    )
                )
                await asyncio.sleep(0)

            completion_text = "".join(accumulated)
            completion_tokens = len(self.tokenizer.encode(completion_text, add_special_tokens=False))
            finish_reason = "length" if (max_tokens is not None and completion_tokens >= max_tokens) else "stop"

            yield self._encode_chunk(
                ChatCompletionStreamResponse(
                    id=request_id,
                    model=self.model_name,
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=0,
                            delta=DeltaMessage(),
                            finish_reason=finish_reason,
                        )
                    ],
                    created=int(time.time()),
                )
            )

            if include_usage:
                prompt_tokens = self._count_prompt_tokens(messages)
                yield self._encode_chunk(
                    ChatCompletionStreamResponse(
                        id=request_id,
                        model=self.model_name,
                        choices=[],
                        usage=UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        ),
                        created=int(time.time()),
                    )
                )

            yield "data: [DONE]\n\n"
        finally:
            thread.join()

    @staticmethod
    def _encode_chunk(chunk: ChatCompletionStreamResponse) -> str:
        return f"data: {json.dumps(chunk.model_dump(mode='json'))}\n\n"
