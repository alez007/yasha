import asyncio
import logging
from typing import cast, Annotated

from vllm.config import ModelDType, TaskOption
from vllm.config.parallel import DistributedExecutorBackend
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

from yasha.infer.infer_config import ModelUsecase, VllmEngineConfig, YashaModelConfig

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.usage.usage_lib import UsageContext
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_transcription import OpenAIServingTranscription, OpenAIServingTranslation
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    LoRAModulePath,
    OpenAIServingModels
)
from vllm.entrypoints.logger import RequestLogger
from fastapi import FastAPI, Form, HTTPException, Request
from http import HTTPStatus
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    TranscriptionRequest,
    TranscriptionResponse,
)
from fastapi.responses import JSONResponse, Response, StreamingResponse


logger = logging.getLogger("ray.serve")

class VllmInfer():
    @staticmethod
    def check_vllm_support(model_config: YashaModelConfig) -> Exception|None:
        if model_config.use_vllm is True and model_config.usecase not in [ModelUsecase.generate, ModelUsecase.embed, ModelUsecase.transcription]:
            raise Exception("vllm is only supported for embed, generate and transcription models")

    def __init__(self, model_config: YashaModelConfig):
        self.check_vllm_support(model_config)
        self.model_config = model_config

        config_engine_kwargs = model_config.vllm_engine_kwargs.model_dump() if model_config.vllm_engine_kwargs is not None else {}
        config_engine_kwargs['model'] = model_config.model
        config_engine_kwargs['task'] = model_config.usecase
        vllm_engine_kwargs: VllmEngineConfig = VllmEngineConfig(**config_engine_kwargs)
        logger.info("initialising vllm engine with args: %s", vllm_engine_kwargs.model_dump())

        engine_args = AsyncEngineArgs(
            model=vllm_engine_kwargs.model,
            tensor_parallel_size=vllm_engine_kwargs.tensor_parallel_size,
            max_model_len=vllm_engine_kwargs.max_model_len,
            dtype=cast(ModelDType, vllm_engine_kwargs.dtype),
            tokenizer=vllm_engine_kwargs.tokenizer,
            trust_remote_code=vllm_engine_kwargs.trust_remote_code,
            gpu_memory_utilization=vllm_engine_kwargs.gpu_memory_utilization,
            distributed_executor_backend=cast(DistributedExecutorBackend, vllm_engine_kwargs.distributed_executor_backend),
            enable_log_requests=vllm_engine_kwargs.enable_log_requests if vllm_engine_kwargs.enable_log_requests is not None else False,
        )
        # engine_args.engine_use_ray = True

        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)

        self.engine = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            enable_log_requests=engine_args.enable_log_requests,
            disable_log_stats=engine_args.disable_log_stats,
        )

    async def start(self):
        self.vllm_config = await self.engine.get_vllm_config()
        self.supported_tasks = self.vllm_config.model_config.supported_tasks
        logger.info("Supported_tasks: %s", self.supported_tasks)

        self.serving_chat = await self.init_serving_chat()
        self.serving_embedding = await self.init_serving_embeding()
        self.serving_transcription = await self.init_serving_transcription()

    async def init_serving_chat(self) -> OpenAIServingChat|None:
        return OpenAIServingChat(
            engine_client=self.engine,
            model_config=self.vllm_config.model_config,
            models=OpenAIServingModels(
                engine_client=self.engine,
                model_config=self.vllm_config.model_config,
                base_model_paths=[
                    BaseModelPath(name=self.model_config.name, model_path=self.model_config.model)
                ]
            ),
            response_role="assistant",
            request_logger=RequestLogger(max_log_len=None),
            chat_template=None,
            chat_template_content_format='auto',
        ) if "generate" in self.supported_tasks else None

    async def init_serving_embeding(self) -> OpenAIServingEmbedding|None:
        return OpenAIServingEmbedding(
            engine_client=self.engine,
            model_config=self.vllm_config.model_config,
            models=OpenAIServingModels(
                engine_client=self.engine,
                model_config=self.vllm_config.model_config,
                base_model_paths=[
                    BaseModelPath(name=self.model_config.name, model_path=self.model_config.model)
                ]
            ),
            request_logger=None,
            chat_template=None,
            chat_template_content_format='auto',
        ) if "embed" in self.supported_tasks else None


    async def init_serving_transcription(self) -> OpenAIServingTranscription|None:
        return OpenAIServingTranscription(
            engine_client=self.engine,
            model_config=self.vllm_config.model_config,
            models=OpenAIServingModels(
                engine_client=self.engine,
                model_config=self.vllm_config.model_config,
                base_model_paths=[
                    BaseModelPath(name=self.model_config.name, model_path=self.model_config.model)
                ]
            ),
            request_logger=None,
        ) if "transcription" in self.supported_tasks else None

    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        try:
            if self.serving_chat is None:
                raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
                    detail="model does not support this action")

            generator = await self.serving_chat.create_chat_completion(request, raw_request)
        except Exception as e:
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                                detail=str(e)) from e
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.error.code)

        elif isinstance(generator, ChatCompletionResponse):
            return JSONResponse(content=generator.model_dump())

        return StreamingResponse(content=generator, media_type="text/event-stream")


    async def create_embedding(
        self, request: EmbeddingRequest, raw_request: Request
    ):
        try:
            if self.serving_embedding is None:
                raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
                        detail="model does not support this action")

            generator = await self.serving_embedding.create_embedding(request, raw_request)
        except Exception as e:
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                                detail=str(e)) from e
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.error.code)

        elif isinstance(generator, EmbeddingResponse):
            return JSONResponse(content=generator.model_dump())

        return StreamingResponse(content=generator, media_type="text/event-stream")

    async def create_transcription(self, request: Annotated[TranscriptionRequest, Form()], raw_request: Request):
        try:
            if self.serving_transcription is None:
                raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
                        detail="model does not support this action")
            logger.info("start processing audio")
            audio_data = await request.file.read()
            logger.info("audio data %s", audio_data)
            generator = await self.serving_transcription.create_transcription(audio_data, request, raw_request)
        except Exception as e:
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                                detail=str(e)) from e
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.error.code)

        elif isinstance(generator, TranscriptionResponse):
            return JSONResponse(content=generator.model_dump())

        return StreamingResponse(content=generator, media_type="text/event-stream")