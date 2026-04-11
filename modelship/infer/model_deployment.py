import time
from collections.abc import AsyncGenerator
from typing import Any

from ray import serve

from modelship.infer.base_infer import BaseInfer
from modelship.infer.custom.custom_infer import CustomInfer
from modelship.infer.diffusers.diffusers_infer import DiffusersInfer
from modelship.infer.infer_config import DisconnectProxy, ModelLoader, ModelshipModelConfig
from modelship.infer.transformers.transformers_infer import TransformersInfer
from modelship.infer.vllm.vllm_infer import VllmInfer
from modelship.logging import configure_logging, get_logger
from modelship.metrics import (
    EMBEDDING_DURATION_SECONDS,
    GENERATION_DURATION_SECONDS,
    IMAGE_GENERATION_DURATION_SECONDS,
    MODEL_LOAD_DURATION_SECONDS,
    MODEL_LOAD_FAILURES_TOTAL,
    TRANSCRIPTION_DURATION_SECONDS,
    TTS_GENERATION_DURATION_SECONDS,
)
from modelship.openai.protocol import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
    SpeechRequest,
    TranscriptionRequest,
    TranslationRequest,
)

logger = get_logger("infer.deployment")


@serve.deployment
class ModelDeployment:
    async def __init__(self, config: ModelshipModelConfig):
        configure_logging()
        self.config = config
        start = time.monotonic()
        self.infer: BaseInfer
        try:
            if config.loader == ModelLoader.vllm:
                self.infer = VllmInfer(config)
            elif config.loader == ModelLoader.transformers:
                self.infer = TransformersInfer(config)
            elif config.loader == ModelLoader.diffusers:
                self.infer = DiffusersInfer(config)
            else:
                self.infer = CustomInfer(config)

            await self.infer.start()
            await self.infer.warmup()
        except Exception:
            MODEL_LOAD_FAILURES_TOTAL.inc(tags={"model": config.name, "loader": config.loader.value})
            raise
        finally:
            MODEL_LOAD_DURATION_SECONDS.observe(
                time.monotonic() - start, tags={"model": config.name, "loader": config.loader.value}
            )

    @staticmethod
    def _set_request_id(request_id: str | None) -> None:
        from modelship.logging import request_id_var

        request_id_var.set(request_id)

    async def generate(
        self,
        request: ChatCompletionRequest,
        request_headers: dict[str, str],
        disconnect_event: Any,
        request_id: str | None = None,
    ):
        self._set_request_id(request_id)
        proxy = DisconnectProxy(disconnect_event, request_headers)
        start = time.monotonic()
        result = await self.infer.create_chat_completion(request, proxy)
        GENERATION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def embed(
        self,
        request: EmbeddingRequest,
        request_headers: dict[str, str],
        disconnect_event: Any,
        request_id: str | None = None,
    ):
        self._set_request_id(request_id)
        proxy = DisconnectProxy(disconnect_event, request_headers)
        start = time.monotonic()
        result = await self.infer.create_embedding(request, proxy)
        EMBEDDING_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def transcribe(
        self,
        audio_data: bytes,
        request: TranscriptionRequest,
        request_headers: dict[str, str],
        disconnect_event: Any,
        request_id: str | None = None,
    ):
        self._set_request_id(request_id)
        proxy = DisconnectProxy(disconnect_event, request_headers)
        start = time.monotonic()
        result = await self.infer.create_transcription(audio_data, request, proxy)
        TRANSCRIPTION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def translate(
        self,
        audio_data: bytes,
        request: TranslationRequest,
        request_headers: dict[str, str],
        disconnect_event: Any,
        request_id: str | None = None,
    ):
        self._set_request_id(request_id)
        proxy = DisconnectProxy(disconnect_event, request_headers)
        start = time.monotonic()
        result = await self.infer.create_translation(audio_data, request, proxy)
        TRANSCRIPTION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def speak(
        self,
        request: SpeechRequest,
        request_headers: dict[str, str],
        disconnect_event: Any,
        request_id: str | None = None,
    ):
        self._set_request_id(request_id)
        proxy = DisconnectProxy(disconnect_event, request_headers)
        start = time.monotonic()
        result = await self.infer.create_speech(request, proxy)
        TTS_GENERATION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def imagine(
        self,
        request: ImageGenerationRequest,
        request_headers: dict[str, str],
        disconnect_event: Any,
        request_id: str | None = None,
    ):
        self._set_request_id(request_id)
        proxy = DisconnectProxy(disconnect_event, request_headers)
        start = time.monotonic()
        result = await self.infer.create_image_generation(request, proxy)
        IMAGE_GENERATION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result
