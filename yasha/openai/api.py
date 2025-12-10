import logging
import asyncio
from typing import Annotated
from pydantic import BaseModel, Field
import time
from yasha.infer.infer_config import YashaModelConfig, SpeechRequest
from yasha.infer.vllm.vllm_infer import VllmInfer
from yasha.infer.custom.custom_infer import CustomInfer
from yasha.infer.transformers.transformers_infer import TransformersInfer
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    TranslationRequest
)
from vllm.entrypoints.pooling.embed.protocol import EmbeddingRequest, EmbeddingResponse
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    LoRAModulePath,
    OpenAIServingModels
)
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from http import HTTPStatus
from vllm.entrypoints.logger import RequestLogger
from fastapi.middleware.cors import CORSMiddleware

from transformers import pipeline
import torch


logger = logging.getLogger()

def build_app():
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app

app = build_app()

class OpenAiModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "yasha"

class OpenaiModelList(BaseModel):
    object: str = "list"
    data: list[OpenAiModelCard] = []

class YashaAPI:
    def __init__(self, yml_api_config: list[YashaModelConfig]):
        self.yml_api_config = yml_api_config

        self.infers: dict[str, VllmInfer|TransformersInfer|CustomInfer] = {}
        for yml_model_config in yml_api_config:
            if yml_model_config.plugin is not None:
                self.infers[yml_model_config.name] = CustomInfer(model_config=yml_model_config)
            elif yml_model_config.use_vllm is not False:
                self.infers[yml_model_config.name] = VllmInfer(yml_model_config)
            else:
                self.infers[yml_model_config.name] = TransformersInfer(yml_model_config)

        self.models: list[OpenAiModelCard] = []

        asyncio.ensure_future(self.start())

    async def start(self):
        for model_name, infer in self.infers.items():
            await infer.start()
            
            if infer.serving_chat is not None:
                self.models.append(OpenAiModelCard(
                    id = model_name
                ))

    def find_model(self, model_name: str) -> OpenAiModelCard|None:
        """
            Checks if a specific model has been found
        """
        found_model: OpenAiModelCard|None = None
        for model_info in self.models:
            if model_info.id == model_name:
                found_model = model_info
        return found_model

    @app.get("/v1/models", response_model=OpenaiModelList)
    async def list_models(self):
        return OpenaiModelList(data=self.models)
    
    @app.get("/v1/models/{model}", response_model=OpenAiModelCard)
    async def model_info(self, model: str) -> OpenAiModelCard:
        logger.info("found models: %s", self.models)
        found_model = self.find_model(model)
        
        if found_model is None:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
                        detail="model not found")

        return found_model


    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        model_name = request.model
        if model_name is not None:
            infer = self.infers[model_name]
        else:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
                detail="model not found")

        return await infer.create_chat_completion(request, raw_request)

            

    @app.post("/v1/embeddings")
    async def create_embeddings(
        self, request: EmbeddingRequest, raw_request: Request
    ):
        model_name = request.model
        if model_name is not None:
            infer = self.infers[model_name]
        else:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
                detail="model not found")
        
        return await infer.create_embedding(request, raw_request)


    @app.post("/v1/audio/transcriptions")
    async def create_transcriptions(self, request: Annotated[TranscriptionRequest,
                                                   Form()], raw_request: Request):
        model_name = request.model
        if model_name is not None:
            infer = self.infers[model_name]
        else:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
                detail="model not found")
        
        return await infer.create_transcription(request, raw_request)

    @app.post("/v1/audio/translations")
    async def create_translations(self, request: Annotated[TranslationRequest,
                                                   Form()], raw_request: Request):
        model_name = request.model
        if model_name is not None:
            infer = self.infers[model_name]
        else:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
                detail="model not found")
        
        return await infer.create_translation(request, raw_request)

    @app.post("/v1/audio/speech")
    async def create_speech(self, request: SpeechRequest, raw_request: Request):
        model_name = request.model

        try:
            infer = self.infers[model_name]

            return await infer.create_speech(request, raw_request)
        except Exception as e:
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail=str(e)) from e
        