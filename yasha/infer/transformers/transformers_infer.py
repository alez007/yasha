import logging
from collections.abc import AsyncGenerator
from typing import cast
import torch

from yasha.infer.infer_config import DisconnectProxy, ModelUsecase, YashaModelConfig, SpeechRequest, RawSpeechResponse
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.speech_to_text.protocol import TranscriptionRequest, TranslationRequest
from vllm.entrypoints.pooling.embed.protocol import EmbeddingRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, ErrorInfo
from starlette.requests import Request
from yasha.infer.transformers.openai.serving_speech import OpenAIServingSpeech
import pkgutil
import importlib

from yasha.plugins.base_plugin import PluginProtoTransformers, BasePluginTransformers
from yasha.plugins import tts

logger = logging.getLogger("ray")


class TransformersInfer():
    _transformers_usecases = [ModelUsecase.tts]

    def __init__(self, model_config: YashaModelConfig):
        self.model_config = model_config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available() and model_config.num_gpus < 1.0:
            torch.cuda.set_per_process_memory_fraction(model_config.num_gpus)

    def __del__(self):
        try:
            if serving_speech := getattr(self, "serving_speech", None):
                del serving_speech
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    async def start(self):
        self.serving_chat = None
        self.serving_embedding = None
        self.serving_transcription = None
        self.serving_translation = None
        self.serving_speech = await self.init_serving_speech()

    async def init_serving_speech(self) -> OpenAIServingSpeech|None:
        logger.info("init serving speech with model: %s", self.model_config.name)

        speech_model: BasePluginTransformers|None = None
        plugin = self.model_config.plugin
        if plugin is not None:
            for _, modname, ispkg in pkgutil.iter_modules(tts.__path__):
                if ispkg is False and modname == plugin:
                    logger.info("Found submodule %s (is a package: %s)", modname, ispkg)
                    module = cast(PluginProtoTransformers, importlib.import_module(".".join([tts.__name__, modname]), package=None))
                    assert self.model_config.model is not None
                    speech_model = module.ModelPlugin(model_name=self.model_config.model, device=self.device)

        return OpenAIServingSpeech(
            speech_model=speech_model
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
