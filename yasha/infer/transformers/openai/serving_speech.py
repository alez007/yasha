import logging
import os
from typing import AsyncGenerator, Union, cast
from fastapi import Request
from vllm.config.model import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, create_error_response
from vllm.entrypoints.logger import RequestLogger

from yasha.infer.infer_config import SpeechRequest, SpeechResponse, RawSpeechResponse
from yasha.plugins import tts
import pkgutil
import importlib
from yasha.plugins.base_plugin import BasePluginTransformers, PluginProtoTransformers
import torch
from transformers import pipeline, AutomaticSpeechRecognitionPipeline, TextToAudioPipeline, PretrainedConfig
from yasha.infer.infer_config import ModelUsecase, SpeechRequest, SpeechResponse, VllmEngineConfig, YashaModelConfig, RawSpeechResponse
from yasha.utils import base_request_id

logger = logging.getLogger("ray")

class OpenAIServingSpeech():
    request_id_prefix = "tts"

    """Handles speech requests"""
    def __init__(
        self,
        speech_model: BasePluginTransformers|None
    ):
        self.speech_model = speech_model
    
    
    async def create_speech(self, request: SpeechRequest, raw_request: Request) -> Union[RawSpeechResponse, AsyncGenerator[str, None],
               ErrorResponse]:

        if self.speech_model is None:
            return create_error_response("tts model is not yet accessible")
        
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"

        return await self.speech_model.generate(request.input, request.voice, request_id, request.stream_format)
        