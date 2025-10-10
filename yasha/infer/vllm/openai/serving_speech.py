import os
from typing import AsyncGenerator, Union
from fastapi import Request
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.logger import RequestLogger

from yasha.infer.infer_config import SpeechRequest, SpeechResponse, RawSpeechResponse
from yasha.plugins.tts.orpheus import OrpheusTTSPlugin


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
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids,
                         log_error_stack=log_error_stack)
        
        self.speech_model = OrpheusTTSPlugin(engine_client=engine_client, model_config=model_config)
    
    

    async def create_speech(self, request: SpeechRequest, raw_request: Request) -> Union[RawSpeechResponse, AsyncGenerator[str, None],
               ErrorResponse]:
        
        request_id = f"{self.request_id_prefix}-{self._base_request_id(raw_request)}"

        return await self.speech_model.generate(request.input, request.voice, request_id, request.stream_format)
        