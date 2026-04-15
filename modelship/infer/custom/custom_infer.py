import importlib
from collections.abc import AsyncGenerator
from typing import cast

from modelship.infer.base_infer import BaseInfer
from modelship.infer.custom.openai.serving_speech import OpenAIServingSpeech
from modelship.infer.infer_config import ModelshipModelConfig, ModelUsecase, RawRequestProxy
from modelship.logging import get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    RawSpeechResponse,
    SpeechRequest,
)
from modelship.plugins.base_plugin import BasePlugin, PluginProto

logger = get_logger("infer.custom")


class CustomInfer(BaseInfer):
    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)
        self.custom_engine: BasePlugin | None = None
        self.serving_speech: OpenAIServingSpeech | None = None

    def shutdown(self) -> None:
        pass

    async def start(self):
        plugin = self.model_config.plugin
        if plugin is not None:
            module = cast("PluginProto", importlib.import_module(plugin))
            self.custom_engine = module.ModelPlugin(model_config=self.model_config)
            await self.custom_engine.start()
            self._set_max_context_length(self.custom_engine.max_context_length())

        self.serving_speech = await self.init_serving_speech()

    async def warmup(self) -> None:
        pass

    async def init_serving_speech(self) -> OpenAIServingSpeech | None:
        logger.info("init serving speech with model: %s", self.model_config.name)
        return (
            OpenAIServingSpeech(serving_engine=self.custom_engine)
            if self.model_config.usecase is ModelUsecase.tts
            else None
        )

    async def create_speech(
        self, request: SpeechRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | RawSpeechResponse | AsyncGenerator[str, None]:
        if self.serving_speech is None:
            return await super().create_speech(request, raw_request)
        return await self.serving_speech.create_speech(request, raw_request)
