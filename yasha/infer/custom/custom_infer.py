import logging
from typing import Annotated, cast
import sys
from transformers import pipeline, AutomaticSpeechRecognitionPipeline, TextToAudioPipeline, PretrainedConfig
import torch

from yasha.infer.infer_config import ModelUsecase, YashaModelConfig, SpeechRequest, RawSpeechResponse
from fastapi import FastAPI, Form, HTTPException, Request
from http import HTTPStatus
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, EmbeddingRequest, ErrorResponse, TranscriptionRequest, TranscriptionResponse, TranslationRequest
from yasha.infer.custom.openai.serving_speech import OpenAIServingSpeech
import pkgutil
import importlib

from yasha.plugins.base_plugin import BasePlugin, PluginProto, PluginProtoTransformers, BasePluginTransformers
from yasha.plugins import tts
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("ray")

class CustomInfer():
    _usecases = [ModelUsecase.tts]

    @staticmethod
    def check_support(model_config: YashaModelConfig) -> Exception|None:
        if model_config.use_vllm is False and model_config.usecase not in CustomInfer._usecases:
            raise Exception("transformers is only supported for (%s) models", ", ".join(CustomInfer._usecases))
    
    def __init__(self, model_config: YashaModelConfig):
        self.check_support(model_config)

        self.model_config = model_config

        self.custom_engine: BasePlugin|None = None

        self.serving_chat = None
        self.serving_embedding = None
        self.serving_transcription = None
        self.serving_translation = None
        self.serving_speech = None
    
    async def start(self):
        plugin = self.model_config.plugin
        if plugin is not None:
            # for _, modname, ispkg in pkgutil.walk_packages(path=tts.__path__, prefix=tts.__name__ + '.'):
            for _, modname, ispkg in pkgutil.iter_modules(tts.__path__):
                if modname==plugin:
                    logger.info("Found submodule %s (is a package: %s)", modname, ispkg)
                    module = cast(PluginProto, importlib.import_module(".".join([tts.__name__, modname, modname]) if ispkg is True else ".".join([tts.__name__, modname]), package=None))
                    self.custom_engine = module.ModelPlugin(model_config=self.model_config)
                    await self.custom_engine.start()


        self.serving_speech = await self.init_serving_speech()
    
    async def init_serving_speech(self) -> OpenAIServingSpeech|None:
        logger.info("init serving speech with model: %s", self.model_config.name)

        return OpenAIServingSpeech(
            serving_engine=self.custom_engine
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


        
