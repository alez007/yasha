import logging

from yasha.config.infer_config import YashaModelConfig

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.usage.usage_lib import UsageContext

logger = logging.getLogger("ray.serve")

class VllmInfer():
    def __init__(self, model_config: YashaModelConfig):
        logger.info("initialising vllm engine with args: %s", model_config.vllm_engine_kwargs.model_dump())

        engine_args = AsyncEngineArgs(
            model=model_config.vllm_engine_kwargs.model,
            tensor_parallel_size=model_config.vllm_engine_kwargs.tensor_parallel_size,
            max_model_len=model_config.vllm_engine_kwargs.max_model_len,
            dtype=model_config.vllm_engine_kwargs.dtype,
            tokenizer=model_config.vllm_engine_kwargs.tokenizer,
            trust_remote_code=model_config.vllm_engine_kwargs.trust_remote_code,
            gpu_memory_utilization=model_config.vllm_engine_kwargs.gpu_memory_utilization,
            distributed_executor_backend=model_config.vllm_engine_kwargs.distributed_executor_backend,
            enable_log_requests=model_config.vllm_engine_kwargs.enable_log_requests,
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
        