import logging
from collections.abc import AsyncGenerator
from typing import cast

from vllm.config.model import ModelDType
from vllm.config.parallel import DistributedExecutorBackend
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest, ChatCompletionResponse
from vllm.entrypoints.openai.speech_to_text.protocol import TranslationRequest, TranslationResponse

from yasha.infer.infer_config import DisconnectProxy, ModelUsecase, SpeechRequest, VllmEngineConfig, YashaModelConfig, RawSpeechResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.usage.usage_lib import UsageContext
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding
from vllm.entrypoints.openai.speech_to_text.serving import OpenAIServingTranscription, OpenAIServingTranslation
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.speech_to_text.protocol import TranscriptionRequest, TranscriptionResponse
from vllm.entrypoints.pooling.embed.protocol import EmbeddingRequest, EmbeddingResponse
from yasha.infer.vllm.openai.serving_speech import OpenAIServingSpeech


logger = logging.getLogger("ray")


class VllmInfer():
    _vllm_usecases = [ModelUsecase.generate, ModelUsecase.embed, ModelUsecase.transcription, ModelUsecase.translation]

    def __init__(self, model_config: YashaModelConfig):
        self.model_config = model_config

        config_engine_kwargs = model_config.vllm_engine_kwargs.model_dump() if model_config.vllm_engine_kwargs is not None else {}
        config_engine_kwargs['model'] = model_config.model

        # gpu_memory_utilization: use explicit value if set, otherwise fall back to num_gpus.
        # Only valid as gpu_memory_utilization when < 1.0; multi-GPU models default to 0.9.
        if config_engine_kwargs.get('gpu_memory_utilization') is None:
            config_engine_kwargs['gpu_memory_utilization'] = model_config.num_gpus if model_config.num_gpus < 1.0 else 0.9

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

    async def init_serving_chat(self) -> OpenAIServingChat|None:
        logger.info("init_serving_chat: %s, %s", self.supported_tasks, self.model_config.usecase)
        if not (self.model_config.usecase is ModelUsecase.generate and "generate" in self.supported_tasks):
            return None

        models = OpenAIServingModels(
            engine_client=self.engine,
            base_model_paths=[
                BaseModelPath(name=self.model_config.name, model_path=self.model_config.model)
            ]
        )

        openai_serving_render = OpenAIServingRender(
            model_config=self.engine.model_config,
            renderer=self.engine.renderer,
            io_processor=self.engine.io_processor,
            model_registry=models.registry,
            request_logger=RequestLogger(max_log_len=None),
            chat_template=None,
            chat_template_content_format=self.model_config.vllm_engine_kwargs.chat_template_content_format,
            enable_auto_tools=True if self.model_config.vllm_engine_kwargs.enable_auto_tool_choice is not None else False,
            tool_parser=self.model_config.vllm_engine_kwargs.tool_call_parser if self.model_config.vllm_engine_kwargs.tool_call_parser is not None else None,
        )

        return OpenAIServingChat(
            engine_client=self.engine,
            models=models,
            openai_serving_render=openai_serving_render,
            response_role="assistant",
            request_logger=RequestLogger(max_log_len=None),
            chat_template=None,
            chat_template_content_format=self.model_config.vllm_engine_kwargs.chat_template_content_format,
            enable_auto_tools=True if self.model_config.vllm_engine_kwargs.enable_auto_tool_choice is not None else False,
            tool_parser=self.model_config.vllm_engine_kwargs.tool_call_parser if self.model_config.vllm_engine_kwargs.tool_call_parser is not None else None,
        )

    async def init_serving_embeding(self) -> ServingEmbedding|None:
        logger.info("init_serving_embeding: %s, %s", self.supported_tasks, self.model_config.usecase)
        return ServingEmbedding(
            engine_client=self.engine,
            models=OpenAIServingModels(
                engine_client=self.engine,
                base_model_paths=[
                    BaseModelPath(name=self.model_config.name, model_path=self.model_config.model)
                ]
            ),
            request_logger=RequestLogger(max_log_len=None),
            chat_template=None,
            chat_template_content_format='auto',
        ) if self.model_config.usecase is ModelUsecase.embed and any(task in self.supported_tasks for task in ['embed', 'embedding']) else None

    async def init_serving_transcription(self) -> OpenAIServingTranscription|None:
        logger.info("init_serving_transcription: %s, %s", self.supported_tasks, self.model_config.usecase)
        return OpenAIServingTranscription(
            engine_client=self.engine,
            models=OpenAIServingModels(
                engine_client=self.engine,
                base_model_paths=[
                    BaseModelPath(name=self.model_config.name, model_path=self.model_config.model)
                ]
            ),
            request_logger=RequestLogger(max_log_len=None),
        ) if (self.model_config.usecase in [ModelUsecase.transcription, ModelUsecase.translation]) and "transcription" in self.supported_tasks else None

    async def init_serving_translation(self) -> OpenAIServingTranslation|None:
        logger.info("init_serving_translation: %s, %s", self.supported_tasks, self.model_config.usecase)
        return OpenAIServingTranslation(
            engine_client=self.engine,
            models=OpenAIServingModels(
                engine_client=self.engine,
                base_model_paths=[
                    BaseModelPath(name=self.model_config.name, model_path=self.model_config.model)
                ]
            ),
            request_logger=RequestLogger(max_log_len=None),
        ) if (self.model_config.usecase in [ModelUsecase.transcription, ModelUsecase.translation]) and "transcription" in self.supported_tasks else None

    async def init_serving_speech(self) -> OpenAIServingSpeech|None:
        logger.info("init_serving_speech: %s, %s", self.supported_tasks, self.model_config.usecase)
        return OpenAIServingSpeech(
            engine_client=self.engine,
            model_config=self.vllm_config.model_config,
            models=OpenAIServingModels(
                engine_client=self.engine,
                base_model_paths=[
                    BaseModelPath(name=self.model_config.name, model_path=self.model_config.model)
                ]
            ),
            request_logger=RequestLogger(max_log_len=None),
            plugin=self.model_config.plugin,
        ) if self.model_config.usecase is ModelUsecase.tts and "generate" in self.supported_tasks else None

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | ChatCompletionResponse | AsyncGenerator[str, None]:
        if self.serving_chat is None:
            return ErrorResponse(message="model does not support this action", type="invalid_request_error", code=404)
        return await self.serving_chat.create_chat_completion(request, raw_request)

    async def create_embedding(
        self, request: EmbeddingRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | EmbeddingResponse | AsyncGenerator:
        if self.serving_embedding is None:
            return ErrorResponse(message="model does not support this action", type="invalid_request_error", code=404)
        return await self.serving_embedding.create_embedding(request, raw_request)

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | TranscriptionResponse | AsyncGenerator:
        if self.serving_transcription is None:
            return ErrorResponse(message="model does not support this action", type="invalid_request_error", code=404)
        request.timestamp_granularities = []
        return await self.serving_transcription.create_transcription(audio_data, request, raw_request)

    async def create_translation(
        self, audio_data: bytes, request: TranslationRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | TranslationResponse | AsyncGenerator:
        if self.serving_translation is None:
            return ErrorResponse(message="model does not support this action", type="invalid_request_error", code=404)
        return await self.serving_translation.create_translation(audio_data, request, raw_request)

    async def create_speech(
        self, request: SpeechRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | RawSpeechResponse | AsyncGenerator[str, None]:
        if self.serving_speech is None:
            return ErrorResponse(message="model does not support this action", type="invalid_request_error", code=404)
        return await self.serving_speech.create_speech(request, raw_request)
