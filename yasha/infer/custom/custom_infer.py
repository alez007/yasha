import importlib
import logging
from collections.abc import AsyncGenerator
from typing import cast

from starlette.requests import Request

from yasha.infer.base_infer import BaseInfer
from yasha.infer.custom.openai.serving_speech import OpenAIServingSpeech
from yasha.infer.infer_config import DisconnectProxy, ModelUsecase, YashaModelConfig
from yasha.openai.protocol import (
    ErrorResponse,
    RawSpeechResponse,
    SpeechRequest,
)
from yasha.plugins.base_plugin import BasePlugin, PluginProto

logger = logging.getLogger("ray")


class CustomInfer(BaseInfer):
    def __init__(self, model_config: YashaModelConfig):
        super().__init__(model_config)
        self.custom_engine: BasePlugin | None = None
        self.serving_speech: OpenAIServingSpeech | None = None

    async def start(self):
        plugin = self.model_config.plugin
        if plugin is not None:
            module = cast("PluginProto", importlib.import_module(plugin))
            self.custom_engine = module.ModelPlugin(model_config=self.model_config)
            await self.custom_engine.start()
            self._set_max_context_length(self.custom_engine.max_context_length())

        self.serving_speech = await self.init_serving_speech()

    async def warmup(self) -> None:
        if self.serving_speech is None:
            return
        logger.info("Warming up custom TTS model: %s", self.model_config.name)
        request = SpeechRequest(model=self.model_config.name, input="warmup", voice="default")
        result = await self.create_speech(request, DisconnectProxy(None, {}))
        if isinstance(result, AsyncGenerator):
            async for _ in result:
                pass
        logger.info("Warmup TTS done for %s", self.model_config.name)

    async def init_serving_speech(self) -> OpenAIServingSpeech | None:
        logger.info("init serving speech with model: %s", self.model_config.name)
        return (
            OpenAIServingSpeech(serving_engine=self.custom_engine)
            if self.model_config.usecase is ModelUsecase.tts
            else None
        )

    async def create_speech(
        self, request: SpeechRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | RawSpeechResponse | AsyncGenerator[str, None]:
        if self.serving_speech is None:
            return await super().create_speech(request, raw_request)
        return await self.serving_speech.create_speech(request, cast("Request", raw_request))
