import logging
from collections.abc import AsyncGenerator
from typing import ClassVar, cast

from starlette.requests import Request
from starlette.responses import Response
from vllm.config.model import ModelDType
from vllm.config.parallel import DistributedExecutorBackend
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.speech_to_text.serving import OpenAIServingTranscription, OpenAIServingTranslation
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from yasha.infer.infer_config import DisconnectProxy, ModelUsecase, VllmEngineConfig, YashaModelConfig
from yasha.infer.vllm.openai.serving_speech import OpenAIServingSpeech
from yasha.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    ErrorInfo,
    ErrorResponse,
    ImageGenerationRequest,
    RawSpeechResponse,
    SpeechRequest,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
)

logger = logging.getLogger("ray")


class VllmInfer:
    _vllm_usecases: ClassVar[list[ModelUsecase]] = [
        ModelUsecase.generate,
        ModelUsecase.embed,
        ModelUsecase.transcription,
        ModelUsecase.translation,
    ]

    def __init__(self, model_config: YashaModelConfig):
        self.model_config = model_config

        config_engine_kwargs = model_config.vllm_engine_kwargs.model_dump(exclude_unset=True)
        config_engine_kwargs["model"] = model_config.model

        # gpu_memory_utilization: use explicit value if set, otherwise fall back to num_gpus.
        # Only valid as gpu_memory_utilization when < 1.0; multi-GPU models default to 0.9.
        if config_engine_kwargs.get("gpu_memory_utilization") is None:
            config_engine_kwargs["gpu_memory_utilization"] = (
                model_config.num_gpus if model_config.num_gpus < 1.0 else 0.9
            )

        # distributed_executor_backend: when use_gpu is a named Ray resource (str) and
        # TP>1, the outer actor has num_gpus=0, so "mp" worker subprocesses see no GPUs.
        # Force "ray" in that case — vLLM worker actors receive VLLM_RAY_PER_WORKER_GPUS
        # (set by start.py) and claim the correct fractional GPU units themselves.
        # For all other TP>1 cases, leave the user's value (or vLLM's own default) alone.
        tp = config_engine_kwargs.get("tensor_parallel_size", 1)
        if tp > 1 and isinstance(model_config.use_gpu, str):
            explicit = config_engine_kwargs.get("distributed_executor_backend")
            if explicit not in (None, "ray"):
                logger.warning(
                    "distributed_executor_backend=%r is not supported with "
                    "tensor_parallel_size>1 and use_gpu as a named resource — overriding to 'ray'.",
                    explicit,
                )
            config_engine_kwargs["distributed_executor_backend"] = "ray"

        self.vllm_engine_kwargs: VllmEngineConfig = VllmEngineConfig(**config_engine_kwargs)
        logger.info("initialising vllm engine with args: %s", self.vllm_engine_kwargs.model_dump())

        engine_args = AsyncEngineArgs(
            model=self.vllm_engine_kwargs.model,
            tensor_parallel_size=self.vllm_engine_kwargs.tensor_parallel_size,
            max_model_len=cast("int", self.vllm_engine_kwargs.max_model_len),
            dtype=cast("ModelDType", self.vllm_engine_kwargs.dtype),
            tokenizer=self.vllm_engine_kwargs.tokenizer,
            trust_remote_code=self.vllm_engine_kwargs.trust_remote_code,
            gpu_memory_utilization=self.vllm_engine_kwargs.gpu_memory_utilization,
            distributed_executor_backend=cast(
                "DistributedExecutorBackend", self.vllm_engine_kwargs.distributed_executor_backend
            ),
            enable_log_requests=self.vllm_engine_kwargs.enable_log_requests
            if self.vllm_engine_kwargs.enable_log_requests is not None
            else False,
            quantization=self.vllm_engine_kwargs.quantization,
            kv_cache_dtype=self.vllm_engine_kwargs.kv_cache_dtype or "auto",  # type: ignore[arg-type]
            enforce_eager=self.vllm_engine_kwargs.enforce_eager or False,
        )

        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        # GPU pinning is handled by CUDA_VISIBLE_DEVICES set in ray_actor_options runtime_env.
        # The GPU is always visible as cuda:0 inside the actor — no device_config override needed.

        self.engine = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            enable_log_requests=engine_args.enable_log_requests,
            disable_log_stats=engine_args.disable_log_stats,
        )

    def __del__(self):
        try:
            if engine := getattr(self, "engine", None):
                engine.shutdown()
        except Exception:
            from yasha.metrics import RESOURCE_CLEANUP_ERRORS_TOTAL

            RESOURCE_CLEANUP_ERRORS_TOTAL.inc(tags={"model": self.model_config.name, "component": "vllm_engine"})

    async def start(self):
        logger.info("Start vllm infer for model: %s", self.model_config)
        self.vllm_config = self.engine.vllm_config
        self.supported_tasks = await self.engine.get_supported_tasks()
        logger.info("Supported_tasks: %s", self.supported_tasks)

        self.serving_chat = await self.init_serving_chat()
        self.serving_embedding = await self.init_serving_embeding()
        self.serving_transcription = await self.init_serving_transcription()
        self.serving_translation = await self.init_serving_translation()
        self.serving_speech = await self.init_serving_speech()

    async def init_serving_chat(self) -> OpenAIServingChat | None:
        logger.info("init_serving_chat: %s, %s", self.supported_tasks, self.model_config.usecase)
        if not (self.model_config.usecase is ModelUsecase.generate and "generate" in self.supported_tasks):
            return None

        models = OpenAIServingModels(
            engine_client=self.engine,
            base_model_paths=[BaseModelPath(name=self.model_config.name, model_path=self.vllm_engine_kwargs.model)],
        )

        openai_serving_render = OpenAIServingRender(
            model_config=self.engine.model_config,
            renderer=self.engine.renderer,
            io_processor=self.engine.io_processor,
            model_registry=models.registry,
            request_logger=RequestLogger(max_log_len=None),
            chat_template=None,
            chat_template_content_format=self.vllm_engine_kwargs.chat_template_content_format,
            enable_auto_tools=self.vllm_engine_kwargs.enable_auto_tool_choice is not None,
            tool_parser=self.vllm_engine_kwargs.tool_call_parser
            if self.vllm_engine_kwargs.tool_call_parser is not None
            else None,
        )

        return OpenAIServingChat(
            engine_client=self.engine,
            models=models,
            openai_serving_render=openai_serving_render,
            response_role="assistant",
            request_logger=RequestLogger(max_log_len=None),
            chat_template=None,
            chat_template_content_format=self.vllm_engine_kwargs.chat_template_content_format,
            enable_auto_tools=self.vllm_engine_kwargs.enable_auto_tool_choice is not None,
            tool_parser=self.vllm_engine_kwargs.tool_call_parser
            if self.vllm_engine_kwargs.tool_call_parser is not None
            else None,
        )

    async def init_serving_embeding(self) -> ServingEmbedding | None:
        logger.info("init_serving_embeding: %s, %s", self.supported_tasks, self.model_config.usecase)
        return (
            ServingEmbedding(
                engine_client=self.engine,
                models=OpenAIServingModels(
                    engine_client=self.engine,
                    base_model_paths=[
                        BaseModelPath(name=self.model_config.name, model_path=self.vllm_engine_kwargs.model)
                    ],
                ),
                request_logger=RequestLogger(max_log_len=None),
                chat_template=None,
                chat_template_content_format="auto",
            )
            if self.model_config.usecase is ModelUsecase.embed
            and any(task in self.supported_tasks for task in ["embed", "embedding"])
            else None
        )

    async def init_serving_transcription(self) -> OpenAIServingTranscription | None:
        logger.info("init_serving_transcription: %s, %s", self.supported_tasks, self.model_config.usecase)
        return (
            OpenAIServingTranscription(
                engine_client=self.engine,
                models=OpenAIServingModels(
                    engine_client=self.engine,
                    base_model_paths=[
                        BaseModelPath(name=self.model_config.name, model_path=self.vllm_engine_kwargs.model)
                    ],
                ),
                request_logger=RequestLogger(max_log_len=None),
            )
            if (self.model_config.usecase in [ModelUsecase.transcription, ModelUsecase.translation])
            and "transcription" in self.supported_tasks
            else None
        )

    async def init_serving_translation(self) -> OpenAIServingTranslation | None:
        logger.info("init_serving_translation: %s, %s", self.supported_tasks, self.model_config.usecase)
        return (
            OpenAIServingTranslation(
                engine_client=self.engine,
                models=OpenAIServingModels(
                    engine_client=self.engine,
                    base_model_paths=[
                        BaseModelPath(name=self.model_config.name, model_path=self.vllm_engine_kwargs.model)
                    ],
                ),
                request_logger=RequestLogger(max_log_len=None),
            )
            if (self.model_config.usecase in [ModelUsecase.transcription, ModelUsecase.translation])
            and "transcription" in self.supported_tasks
            else None
        )

    async def init_serving_speech(self) -> OpenAIServingSpeech | None:
        logger.info("init_serving_speech: %s, %s", self.supported_tasks, self.model_config.usecase)
        return (
            OpenAIServingSpeech(
                engine_client=self.engine,
                model_config=self.vllm_config.model_config,
                models=OpenAIServingModels(
                    engine_client=self.engine,
                    base_model_paths=[
                        BaseModelPath(name=self.model_config.name, model_path=self.vllm_engine_kwargs.model)
                    ],
                ),
                request_logger=RequestLogger(max_log_len=None),
                plugin=self.model_config.plugin,
            )
            if self.model_config.usecase is ModelUsecase.tts and "generate" in self.supported_tasks
            else None
        )

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | ChatCompletionResponse | AsyncGenerator[str, None]:
        if self.serving_chat is None:
            return ErrorResponse(
                error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404)
            )
        return await self.serving_chat.create_chat_completion(request, cast("Request", raw_request))

    async def create_embedding(
        self, request: EmbeddingRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | Response:
        if self.serving_embedding is None:
            return ErrorResponse(
                error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404)
            )
        return await self.serving_embedding(request, cast("Request", raw_request))

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | TranscriptionResponse | TranscriptionResponseVerbose | AsyncGenerator[str, None]:
        if self.serving_transcription is None:
            return ErrorResponse(
                error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404)
            )
        request.timestamp_granularities = []
        return await self.serving_transcription.create_transcription(audio_data, request, cast("Request", raw_request))

    async def create_translation(
        self, audio_data: bytes, request: TranslationRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | TranslationResponse | TranslationResponseVerbose | AsyncGenerator[str, None]:
        if self.serving_translation is None:
            return ErrorResponse(
                error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404)
            )
        return await self.serving_translation.create_translation(audio_data, request, cast("Request", raw_request))

    async def create_speech(
        self, request: SpeechRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | RawSpeechResponse | AsyncGenerator[str, None]:
        if self.serving_speech is None:
            return ErrorResponse(
                error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404)
            )
        return await self.serving_speech.create_speech(request, cast("Request", raw_request))

    async def create_image_generation(
        self, _request: ImageGenerationRequest, _raw_request: DisconnectProxy
    ) -> ErrorResponse:
        return ErrorResponse(
            error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404)
        )
