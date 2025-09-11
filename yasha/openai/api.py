import logging
import asyncio
from pydantic import BaseModel, Field
from typing import List
import time
from yasha.config.infer_config import YashaModelConfig
from yasha.infer.vllm.vllm_infer import VllmInfer
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    EmbeddingRequest,
    EmbeddingResponse
)
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    LoRAModulePath,
    OpenAIServingModels
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from http import HTTPStatus
from vllm.entrypoints.logger import RequestLogger
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("ray.serve")

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
    data: List[OpenAiModelCard] = []

class YashaAPI:
    def __init__(self, yml_api_config: List[YashaModelConfig]):
        self.yml_api_config = yml_api_config

        self.vllm_infers: Dict[str, VllmInfer] = {}
        for yml_model_config in yml_api_config:
            self.vllm_infers[yml_model_config.name] = VllmInfer(yml_model_config)

        asyncio.ensure_future(self.start())

    async def start(self):
        self.models: List[OpenAiModelCard] = []

        self.serving_chat: Dict[str, OpenAIServingChat] = {}
        self.serving_embedding: Dict[str, OpenAIServingEmbedding] = {}
        
        for model_name, vllm in self.vllm_infers.items():
            vllm_config = await vllm.engine.get_vllm_config()

            model_config = vllm_config.model_config

            supported_tasks = model_config.supported_tasks
            logger.info("Supported_tasks: %s", supported_tasks)

            self.serving_chat[model_name] = OpenAIServingChat(
                engine_client=vllm.engine,
                model_config=model_config,
                models=OpenAIServingModels(
                    engine_client=vllm.engine,
                    model_config=model_config,
                    base_model_paths=[
                        BaseModelPath(name=model_name, model_path=model_config.model)
                    ]
                ),
                response_role="assistant",
                request_logger=RequestLogger(max_log_len=None),
                chat_template=None,
                chat_template_content_format='auto',
            ) if "generate" in supported_tasks else None

            self.serving_embedding[model_name] = OpenAIServingEmbedding(
                engine_client=vllm.engine,
                model_config=model_config,
                models=OpenAIServingModels(
                    engine_client=vllm.engine,
                    model_config=model_config,
                    base_model_paths=[
                        BaseModelPath(name=model_name, model_path=model_config.model)
                    ]
                ),
                request_logger=None,
                chat_template=None,
                chat_template_content_format='auto',
            ) if "embed" in supported_tasks else None

        for model_name in self.serving_chat.keys():
            self.models.append(OpenAiModelCard(
                id = model_name
            ))

    def find_model(self, model_name: str) -> OpenAiModelCard:
        """
            Checks if a specific model has been found
        """
        found_model: OpenAiModelCard = None
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
        try:
            model_name = request.model
            serving_chat = self.serving_chat[model_name]

            if serving_chat is None:
                raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
                        detail="model not found")

            generator = await serving_chat.create_chat_completion(request, raw_request)
        except Exception as e:
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                                detail=str(e)) from e
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.code)

        elif isinstance(generator, ChatCompletionResponse):
            return JSONResponse(content=generator.model_dump())

        return StreamingResponse(content=generator, media_type="text/event-stream")

    @app.post("/v1/embeddings")
    async def create_embeddings(
        self, request: EmbeddingRequest, raw_request: Request
    ):
        try:
            model_name = request.model
            serving_embedding = self.serving_embedding[model_name]

            if serving_embedding is None:
                raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
                        detail="model not found")

            generator = await serving_embedding.create_embedding(request, raw_request)
        except Exception as e:
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                                detail=str(e)) from e
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.code)

        elif isinstance(generator, EmbeddingResponse):
            return JSONResponse(content=generator.model_dump())

        return StreamingResponse(content=generator, media_type="text/event-stream")