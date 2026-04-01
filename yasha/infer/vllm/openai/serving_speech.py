import logging
import os
from typing import AsyncGenerator, Union, cast
from fastapi import Request
from vllm.config.model import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.utils import create_error_response
from vllm.entrypoints.logger import RequestLogger

import importlib
from typing import cast
from yasha.infer.infer_config import SpeechRequest, SpeechResponse, RawSpeechResponse
from yasha.plugins.base_plugin import PluginProtoVllm


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
        plugin: str|None = None,
    ):
        super().__init__(engine_client=engine_client,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids)

        logger = logging.getLogger("ray")

        if plugin is not None:
            logger.info("Loading plugin: %s", plugin)
            module = cast(PluginProtoVllm, importlib.import_module(plugin))
            self.speech_model = module.ModelPlugin(engine_client=engine_client, model_config=model_config)
    
    

    async def create_speech(self, request: SpeechRequest, raw_request: Request) -> Union[RawSpeechResponse, AsyncGenerator[str, None],
               ErrorResponse]:

        if self.speech_model is None:
            return create_error_response("tts model is not yet accessible")
        
        request_id = f"{self.request_id_prefix}-{self._base_request_id(raw_request)}"

        return await self.speech_model.generate(request.input, request.voice, request_id, request.stream_format)
        