import logging
from collections.abc import AsyncGenerator
from typing import cast
import pkgutil
import importlib

from yasha.infer.infer_config import DisconnectProxy, ModelUsecase, YashaModelConfig, SpeechRequest, RawSpeechResponse
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.speech_to_text.protocol import TranscriptionRequest, TranslationRequest
from vllm.entrypoints.pooling.embed.protocol import EmbeddingRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, ErrorInfo
from starlette.requests import Request
from yasha.infer.custom.openai.serving_speech import OpenAIServingSpeech

from yasha.plugins.base_plugin import BasePlugin, PluginProto
from yasha.plugins import tts

logger = logging.getLogger("ray")


class CustomInfer():
    def __init__(self, model_config: YashaModelConfig):
        self.model_config = model_config
        self.custom_engine: BasePlugin|None = None
        self.serving_speech: OpenAIServingSpeech|None = None

    async def start(self):
        plugin = self.model_config.plugin
        if plugin is not None:
            for _, modname, ispkg in pkgutil.iter_modules(tts.__path__):
                if modname == plugin:
                    logger.info("Found submodule %s (is a package: %s)", modname, ispkg)
                    module = cast(PluginProto, importlib.import_module(
                        ".".join([tts.__name__, modname, modname]) if ispkg is True
                        else ".".join([tts.__name__, modname]),
                        package=None
                    ))
                    self.custom_engine = module.ModelPlugin(model_config=self.model_config)
                    await self.custom_engine.start()

        self.serving_speech = await self.init_serving_speech()

    async def init_serving_speech(self) -> OpenAIServingSpeech|None:
        logger.info("init serving speech with model: %s", self.model_config.name)
        return OpenAIServingSpeech(
            serving_engine=self.custom_engine
        ) if self.model_config.usecase is ModelUsecase.tts else None

    async def create_chat_completion(
        self, _request: ChatCompletionRequest, _raw_request: DisconnectProxy
    ) -> ErrorResponse:
        return ErrorResponse(error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404))

    async def create_embedding(
        self, _request: EmbeddingRequest, _raw_request: DisconnectProxy
    ) -> ErrorResponse:
        return ErrorResponse(error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404))

    async def create_transcription(
        self, _audio_data: bytes, _request: TranscriptionRequest, _raw_request: DisconnectProxy
    ) -> ErrorResponse:
        return ErrorResponse(error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404))

    async def create_translation(
        self, _audio_data: bytes, _request: TranslationRequest, _raw_request: DisconnectProxy
    ) -> ErrorResponse:
        return ErrorResponse(error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404))

    async def create_speech(
        self, request: SpeechRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | RawSpeechResponse | AsyncGenerator[str, None]:
        if self.serving_speech is None:
            return ErrorResponse(error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404))
        return await self.serving_speech.create_speech(request, cast(Request, raw_request))
