import logging
import os
from typing import AsyncGenerator, Union, cast
from fastapi import Request
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, create_error_response
from vllm.entrypoints.logger import RequestLogger

from yasha.infer.infer_config import SpeechRequest, SpeechResponse, RawSpeechResponse
from yasha.plugins import tts
import pkgutil
import importlib
from yasha.plugins.base_plugin import BasePlugin, PluginProto


class OpenAIServingSpeech(OpenAIServing):
    request_id_prefix = "tts"

    """Handles speech requests"""
    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger|None = None,
        return_tokens_as_token_ids: bool = False,
        log_error_stack: bool = False,
        plugin: str|None = None,
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids,
                         log_error_stack=log_error_stack)

        logger = logging.getLogger("ray")

        if plugin is not None:
            for _, modname, ispkg in pkgutil.iter_modules(tts.__path__):
                logger.info("Found submodule %s (is a package: %s)", modname, ispkg)
                if ispkg is False:
                    module = cast(PluginProto, importlib.import_module(".".join([tts.__name__, modname]), package=None))
                    self.speech_model = module.ModelPlugin(engine_client=engine_client, model_config=model_config)
        
        # self.speech_model = OrpheusTTSPlugin(engine_client=engine_client, model_config=model_config)
    
    

    async def create_speech(self, request: SpeechRequest, raw_request: Request) -> Union[RawSpeechResponse, AsyncGenerator[str, None],
               ErrorResponse]:

        if self.speech_model is None:
            return create_error_response("tts model is not yet accessible")
        
        request_id = f"{self.request_id_prefix}-{self._base_request_id(raw_request)}"

        return await self.speech_model.generate(request.input, request.voice, request_id, request.stream_format)
        