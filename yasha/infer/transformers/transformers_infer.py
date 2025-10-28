import logging
from typing import Annotated
import sys
from transformers import pipeline, AutomaticSpeechRecognitionPipeline, TextToAudioPipeline, PretrainedConfig
import torch

from yasha.infer.infer_config import ModelUsecase, YashaModelConfig, SpeechRequest
from fastapi import FastAPI, Form, HTTPException, Request
from http import HTTPStatus
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, EmbeddingRequest, TranscriptionRequest, TranscriptionResponse, TranslationRequest

logger = logging.getLogger("ray")

class TransformersInfer():
    _transformers_usecases = [ModelUsecase.tts]

    @staticmethod
    def check_transformers_support(model_config: YashaModelConfig) -> Exception|None:
        if model_config.use_vllm is False and model_config.usecase in TransformersInfer._transformers_usecases:
            raise Exception("transformers is only supported for (%s) models", ", ".join(TransformersInfer._transformers_usecases))
    
    def __init__(self, model_config: YashaModelConfig):
        self.check_transformers_support(model_config)

        self.model_config = model_config
    
    async def start(self):
        self.serving_speech = await self.init_serving_speech()
    
    async def init_serving_speech(self) -> TextToAudioPipeline|None:
        logger.info("init serving speech with model: %s", self.model_config.name)
        return pipeline(
            task="text-to-audio",
            model=self.model_config.model,
            config=PretrainedConfig(
                device=self.device
            ),
            dtype=torch.float16
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
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
            detail="model does not support this action")


        
