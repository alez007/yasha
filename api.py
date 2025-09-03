import os
from typing import Dict, Optional, List, Union
import logging
import ray

from fastapi import FastAPI, Depends
from ray import serve
from time import sleep
from ray.serve.handle import DeploymentHandle
import asyncio
from pydantic_yaml import parse_yaml_raw_as
from pydantic import BaseModel

from yasha.config.infer_config import YashaModelConfig
from yasha.infer.vllm.vllm_infer import VllmInfer

from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse
)
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    LoRAModulePath,
                                                    OpenAIServingModels)

from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from http import HTTPStatus

logger = logging.getLogger(__name__)

if ray.is_initialized():
    ray.shutdown()

ray.init(
    address="ray://0.0.0.0:10001"
)

app = FastAPI()

def yasha_app():
    _config_file = os.path.dirname(os.path.abspath(__file__)) + "/yasha/config/models/instruct.yaml"
    _instruct_model_config = {}
    _yml_conf: YashaModelConfig = None
    with open(_config_file, "r") as f:
        _yml_conf = parse_yaml_raw_as(YashaModelConfig, f)

    assert _yml_conf != None

    logger.info("Init yasha app with config: %s", _yml_conf)

    deployment = serve.deployment(serve.ingress(app)(YashaAPI), name="yasha api")
    
    return deployment.options(
        num_replicas=1,
        ray_actor_options=dict(
            num_cpus=3,
            num_gpus=1,
        )
    ).bind(_yml_conf)

class YashaAPI:
    def __init__(self, yml_model_config: YashaModelConfig):
        logger = logging.getLogger("ray.serve")

        self.yml_model_config = yml_model_config
        self.vllm = VllmInfer(yml_model_config)
        
        asyncio.ensure_future(self.start())

    async def start(self):
        vllm_config = await self.vllm.engine.get_vllm_config()

        model_config = vllm_config.model_config

        supported_tasks = model_config.supported_tasks
        logger.info("Supported_tasks: %s", supported_tasks)

        self.serving_chat = OpenAIServingChat(
            engine_client=self.vllm.engine,
            model_config=model_config,
            models=OpenAIServingModels(
                engine_client=self.vllm.engine,
                model_config=model_config,
                base_model_paths=[
                    BaseModelPath(name=self.yml_model_config.name, model_path=self.yml_model_config.vllm_engine_kwargs.model)
                ]
            ),
            response_role="assistant",
            request_logger=None,
            chat_template=None,
            chat_template_content_format='auto',
        ) if "generate" in supported_tasks else None

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        try:
            generator = await self.serving_chat.create_chat_completion(request, raw_request)
        except Exception as e:
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                                detail=str(e)) from e
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.code)

        elif isinstance(generator, ChatCompletionResponse):
            return JSONResponse(content=generator.model_dump())

        return StreamingResponse(content=generator, media_type="text/event-stream")

    
serve.run(yasha_app(), route_prefix='/app1', name='yasha app')
# serve.run(yasha_app(), route_prefix='/app2', name='app2')
