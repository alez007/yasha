import importlib
import logging
from collections.abc import AsyncGenerator
from typing import ClassVar, cast

import torch
from starlette.requests import Request

from yasha.infer.base_infer import BaseInfer
from yasha.infer.infer_config import DisconnectProxy, ModelUsecase, YashaModelConfig
from yasha.infer.transformers.openai.serving_speech import OpenAIServingSpeech
from yasha.openai.protocol import (
    ErrorResponse,
    RawSpeechResponse,
    SpeechRequest,
)
from yasha.plugins.base_plugin import BasePluginTransformers, PluginProtoTransformers

logger = logging.getLogger("ray")


class TransformersInfer(BaseInfer):
    _transformers_usecases: ClassVar[list[ModelUsecase]] = [ModelUsecase.tts]

    def __init__(self, model_config: YashaModelConfig):
        super().__init__(model_config)
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
            from yasha.metrics import RESOURCE_CLEANUP_ERRORS_TOTAL

            RESOURCE_CLEANUP_ERRORS_TOTAL.inc(tags={"model": self.model_config.name, "component": "transformers_model"})

    async def start(self):
        self.serving_speech = await self.init_serving_speech()
        if self.serving_speech and self.serving_speech.speech_model:
            self._set_max_context_length(self.serving_speech.speech_model.max_context_length())

    async def init_serving_speech(self) -> OpenAIServingSpeech | None:
        logger.info("init serving speech with model: %s", self.model_config.name)

        speech_model: BasePluginTransformers | None = None
        plugin = self.model_config.plugin
        if plugin is not None:
            logger.info("Loading plugin: %s", plugin)
            module = cast("PluginProtoTransformers", importlib.import_module(plugin))
            assert self.model_config.model is not None
            speech_model = module.ModelPlugin(model_name=self.model_config.model, device=self.device)

        return OpenAIServingSpeech(speech_model=speech_model) if self.model_config.usecase is ModelUsecase.tts else None

    async def create_speech(
        self, request: SpeechRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | RawSpeechResponse | AsyncGenerator[str, None]:
        if self.serving_speech is None:
            return await super().create_speech(request, raw_request)
        return await self.serving_speech.create_speech(request, cast("Request", raw_request))
