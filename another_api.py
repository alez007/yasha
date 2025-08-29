import os
import asyncio
import time
from typing import AsyncGenerator
import ray
import requests
from fastapi import FastAPI
from ray import serve
from pydantic import BaseModel

from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from argparse import Namespace
from vllm.engine.arg_utils import AsyncEngineArgs
from typing import Dict, Optional, List
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import (make_arg_parser, validate_parsed_serve_args)
import logging
from contextlib import asynccontextmanager
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.protocol import EngineClient
from vllm.config import VllmConfig
from starlette.datastructures import State
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    LoRAModulePath,
                                                    OpenAIServingModels)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.usage.usage_lib import UsageContext


logger = logging.getLogger("ray.serve")


if ray.is_initialized():
    ray.shutdown()

ray.init(
    address="ray://0.0.0.0:10001"
)

app = FastAPI()

@serve.deployment(
    num_replicas=1,
    # placement_group_bundles=[{"CPU": 1}, {"CPU": 1, "GPU": 1}],
    # placement_group_strategy="STRICT_PACK"
)
@serve.ingress(app)
class YashaAPI:
    def __init__(self, args: Namespace) -> serve.Application:
        self.args = args

    @staticmethod
    async def init_app_state(
        engine_client: EngineClient,
        vllm_config: VllmConfig,
        state: State,
        args: Namespace,
    ) -> None:
        logger.info("===============init app state==============")
        if args.served_model_name is not None:
            served_model_names = args.served_model_name
        else:
            served_model_names = [args.model]

        if not args.disable_log_requests:
            request_logger = RequestLogger(max_log_len=args.max_log_len)
        else:
            request_logger = None
        
        base_model_paths = [
            BaseModelPath(name=name, model_path=args.model)
            for name in served_model_names
        ]

        model_config = vllm_config.model_config

        supported_tasks = await engine_client.get_supported_tasks()
        logger.info("Supported_tasks: %s", supported_tasks)

        tool_server = None

        lora_modules = args.lora_modules
        default_mm_loras = (vllm_config.lora_config.default_mm_loras if vllm_config.lora_config is not None else {})
        if default_mm_loras:
            default_mm_lora_paths = [
                LoRAModulePath(
                    name=modality,
                    path=lora_path,
                ) for modality, lora_path in default_mm_loras.items()
            ]
            if args.lora_modules is None:
                lora_modules = default_mm_lora_paths
            else:
                lora_modules += default_mm_lora_paths

        state.openai_serving_models = OpenAIServingModels(
            engine_client=engine_client,
            model_config=model_config,
            base_model_paths=base_model_paths,
            lora_modules=lora_modules,
        )
        await state.openai_serving_models.init_static_loras()

        logger.info("serving models %s", state.openai_serving_models)

        state.openai_serving_chat = OpenAIServingChat(
            engine_client,
            model_config,
            state.openai_serving_models,
            args.response_role,
            request_logger=request_logger,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
            tool_parser=args.tool_call_parser,
            reasoning_parser=args.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            enable_log_outputs=args.enable_log_outputs,
            log_error_stack=args.log_error_stack,
        ) if "generate" in supported_tasks else None

        return None

    @staticmethod
    @asynccontextmanager
    async def build_engine_client(args: Namespace):
        logger.info("===============building engine client==============")
        usage_context = UsageContext.OPENAI_API_SERVER
        engine_args = AsyncEngineArgs.from_cli_args(args)
        engine_args.worker_use_ray = True
        engine_args.engine_use_ray = True

        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        logger.info("===============vllm config created==============")

        async_llm: Optional[AsyncLLM] = None
        try:
            async_llm = AsyncLLM.from_vllm_config(
                vllm_config=vllm_config,
                usage_context=usage_context,
                disable_log_requests=engine_args.disable_log_requests,
                disable_log_stats=engine_args.disable_log_stats,
            )
            logger.info("===============async_llm created==============")

            await async_llm.reset_mm_cache()
            yield async_llm
        finally:
            logger.info("===============async_llm failed==============")
            if async_llm:
                async_llm.shutdown()

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        async with self.build_engine_client(self.args) as engine_client:
            logger.info("===============get vllm config==============")
            vllm_config = await engine_client.get_vllm_config()
            logger.info("===============got vllm config==============")
            await self.init_app_state(engine_client, vllm_config, app.state, self.args)

        handler = raw_request.app.state.openai_serving_chat
        
        try:
            generator = await handler.create_chat_completion(request, raw_request)
        except Exception as e:
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                                detail=str(e)) from e
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.error.code)

        elif isinstance(generator, ChatCompletionResponse):
            return JSONResponse(content=generator.model_dump())

        return StreamingResponse(content=generator, media_type="text/event-stream")


def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )

    parser = make_arg_parser(arg_parser)
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args

app = YashaAPI.bind(parse_vllm_args({
    "model": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "tensor_parallel_size": 1,
    "max_model_len": 20000,
    "gpu_memory_utilization":0.4,
}))

serve.run(app, route_prefix="/api", name="api")




