import asyncio
import json
import time
from collections.abc import AsyncGenerator
from threading import Thread

from transformers import Pipeline, PreTrainedTokenizerBase, TextIteratorStreamer

from modelship.infer.base_serving import OpenAIServing
from modelship.infer.infer_config import RawRequestProxy, TransformersConfig
from modelship.logging import TRACE, get_logger
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
    request_id_prefix = "chat"

    def __init__(self, pipeline: Pipeline, model_name: str, config: TransformersConfig):
        self.pipeline = pipeline
        self.model_name = model_name
        self.config = config
        assert pipeline.tokenizer is not None, "text-generation pipeline must have a tokenizer"
        self.tokenizer: PreTrainedTokenizerBase = pipeline.tokenizer

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

        messages = [{"role": m["role"], "content": m["content"]} for m in request.messages]  # type: ignore[index]

        if request.stream:
            return self._stream(request_id, messages, request.max_tokens)

        try:
            result = await self.run_in_executor(self._run, messages, request.max_tokens)
        except Exception:
            logger.exception("chat completion inference failed for %s", request_id)
            return create_error_response("chat completion inference failed")

        prompt_tokens = len(self.tokenizer.apply_chat_template(messages))
        generated = result[0]["generated_text"]
        completion_text = generated[-1]["content"] if isinstance(generated, list) else generated
        completion_tokens = len(self.tokenizer.encode(completion_text))

        response = ChatCompletionResponse(
            id=request_id,
            model=self.model_name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=completion_text),
                    finish_reason="stop",
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

    def _run(self, messages: list[dict], max_tokens: int | None) -> list:
        kwargs = {**self.config.pipeline_kwargs}
        if max_tokens is not None:
            kwargs["max_new_tokens"] = max_tokens
        return self.pipeline(messages, return_full_text=False, **kwargs)  # type: ignore[return-value]

    async def _stream(self, request_id: str, messages: list[dict], max_tokens: int | None) -> AsyncGenerator[str, None]:
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

        try:
            for text_chunk in streamer:
                if not text_chunk:
                    continue
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    model=self.model_name,
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=0,
                            delta=DeltaMessage(role="assistant", content=text_chunk),
                        )
                    ],
                    created=int(time.time()),
                )
                yield f"data: {json.dumps(chunk.model_dump(mode='json'))}\n\n"
                await asyncio.sleep(0)

            final_chunk = ChatCompletionStreamResponse(
                id=request_id,
                model=self.model_name,
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(),
                        finish_reason="stop",
                    )
                ],
                created=int(time.time()),
            )
            yield f"data: {json.dumps(final_chunk.model_dump(mode='json'))}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            thread.join()
