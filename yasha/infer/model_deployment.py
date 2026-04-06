import logging
from collections.abc import AsyncGenerator
from typing import Any

from ray import serve

from yasha.infer.custom.custom_infer import CustomInfer
from yasha.infer.diffusers.diffusers_infer import DiffusersInfer
from yasha.infer.infer_config import DisconnectProxy, ModelLoader, YashaModelConfig
from yasha.infer.transformers.transformers_infer import TransformersInfer
from yasha.infer.vllm.vllm_infer import VllmInfer
from yasha.openai.protocol import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
    SpeechRequest,
    TranscriptionRequest,
    TranslationRequest,
)

logger = logging.getLogger("ray.serve")


@serve.deployment
class ModelDeployment:
    async def __init__(self, config: YashaModelConfig):
        self.config = config
        if config.loader == ModelLoader.vllm:
            self.infer = VllmInfer(config)
        elif config.loader == ModelLoader.transformers:
            self.infer = TransformersInfer(config)
        elif config.loader == ModelLoader.diffusers:
            self.infer = DiffusersInfer(config)
        else:
            self.infer = CustomInfer(config)

        await self.infer.start()

    async def generate(self, request: ChatCompletionRequest, request_headers: dict[str, str], disconnect_event: Any):
        proxy = DisconnectProxy(disconnect_event, request_headers)
        result = await self.infer.create_chat_completion(request, proxy)
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def embed(self, request: EmbeddingRequest, request_headers: dict[str, str], disconnect_event: Any):
        proxy = DisconnectProxy(disconnect_event, request_headers)
        result = await self.infer.create_embedding(request, proxy)
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def transcribe(
        self, audio_data: bytes, request: TranscriptionRequest, request_headers: dict[str, str], disconnect_event: Any
    ):
        proxy = DisconnectProxy(disconnect_event, request_headers)
        result = await self.infer.create_transcription(audio_data, request, proxy)
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def translate(
        self, audio_data: bytes, request: TranslationRequest, request_headers: dict[str, str], disconnect_event: Any
    ):
        proxy = DisconnectProxy(disconnect_event, request_headers)
        result = await self.infer.create_translation(audio_data, request, proxy)
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def speak(self, request: SpeechRequest, request_headers: dict[str, str], disconnect_event: Any):
        proxy = DisconnectProxy(disconnect_event, request_headers)
        result = await self.infer.create_speech(request, proxy)
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def imagine(self, request: ImageGenerationRequest, request_headers: dict[str, str], disconnect_event: Any):
        proxy = DisconnectProxy(disconnect_event, request_headers)
        result = await self.infer.create_image_generation(request, proxy)
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result
