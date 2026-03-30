import logging
from collections.abc import AsyncGenerator
from ray import serve

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.speech_to_text.protocol import TranscriptionRequest, TranslationRequest
from vllm.entrypoints.pooling.embed.protocol import EmbeddingRequest

from typing import Any
from yasha.infer.infer_config import ModelLoader, SpeechRequest, YashaModelConfig, DisconnectProxy
from yasha.infer.vllm.vllm_infer import VllmInfer
from yasha.infer.transformers.transformers_infer import TransformersInfer
from yasha.infer.custom.custom_infer import CustomInfer

logger = logging.getLogger("ray.serve")


@serve.deployment
class ModelDeployment:
    async def __init__(self, config: YashaModelConfig):
        self.config = config
        if config.loader == ModelLoader.vllm:
            self.infer = VllmInfer(config)
        elif config.loader == ModelLoader.transformers:
            self.infer = TransformersInfer(config)
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

    async def transcribe(self, audio_data: bytes, request: TranscriptionRequest, request_headers: dict[str, str], disconnect_event: Any):
        proxy = DisconnectProxy(disconnect_event, request_headers)
        result = await self.infer.create_transcription(audio_data, request, proxy)
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def translate(self, audio_data: bytes, request: TranslationRequest, request_headers: dict[str, str], disconnect_event: Any):
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
