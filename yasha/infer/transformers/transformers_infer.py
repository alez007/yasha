import logging
from typing import Annotated, cast
import sys
from transformers import pipeline, AutomaticSpeechRecognitionPipeline, TextToAudioPipeline, PretrainedConfig
import torch

from yasha.infer.infer_config import ModelUsecase, YashaModelConfig, SpeechRequest, RawSpeechResponse
from fastapi import FastAPI, Form, HTTPException, Request
from http import HTTPStatus
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.translations.protocol import TranscriptionRequest, TranslationRequest
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
)
from vllm.entrypoints.pooling.embed.protocol import EmbeddingRequest
from yasha.infer.transformers.openai.serving_speech import OpenAIServingSpeech
import pkgutil
import importlib

from yasha.plugins.base_plugin import PluginProtoTransformers, BasePluginTransformers
from yasha.plugins import tts
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("ray")

class TransformersInfer():
    _transformers_usecases = [ModelUsecase.tts]

    @staticmethod
    def check_transformers_support(model_config: YashaModelConfig) -> Exception|None:
        if model_config.use_vllm is False and model_config.usecase not in TransformersInfer._transformers_usecases:
            raise Exception("transformers is only supported for (%s) models", ", ".join(TransformersInfer._transformers_usecases))
    
    def __init__(self, model_config: YashaModelConfig):
        self.check_transformers_support(model_config)

        self.device = f"cuda:{model_config.use_gpu}" if torch.cuda.is_available() else "cpu"

        self.model_config = model_config
    
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
                if ispkg is False and modname==plugin:
                    logger.info("Found submodule %s (is a package: %s)", modname, ispkg)
                    module = cast(PluginProtoTransformers, importlib.import_module(".".join([tts.__name__, modname]), package=None))
                    speech_model = module.ModelPlugin(model_name=self.model_config.model, device=self.device)

        return OpenAIServingSpeech(
            speech_model=speech_model
        ) if self.model_config.usecase is ModelUsecase.tts else None

    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
            detail="model does not support this action")

    async def create_embedding(self, request: EmbeddingRequest, raw_request: Request):
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
            detail="model does not support this action")

    async def create_transcription(self, request: Annotated[TranscriptionRequest, Form()], raw_request: Request):
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
            detail="model does not support this action")
    
    async def create_translation(self, request: Annotated[TranslationRequest, Form()], raw_request: Request):
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
            detail="model does not support this action")
    
    async def create_speech(self, request: SpeechRequest, raw_request: Request):
        try:
            if self.serving_speech is None:
                raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
                        detail="model does not support this action")
            
            generator = await self.serving_speech.create_speech(request, raw_request)
        except Exception as e:
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                                detail=str(e)) from e
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.error.code)

        elif isinstance(generator, RawSpeechResponse):
            logger.info("returning full audio buffer response")
            return Response(content=generator.audio, media_type=generator.media_type)

        logger.info("returning streaming response")
        return StreamingResponse(content=generator, media_type="text/event-stream")


        
