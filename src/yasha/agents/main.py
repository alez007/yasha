import asyncio
from pydantic import BaseModel

import ray
from ray import serve

from ray.serve.llm import LLMConfig, LLMServer, LLMRouter, build_openai_app, LLMServingArgs, build_llm_deployment
from ray.llm._internal.serve.configs.openai_api_models import ChatCompletionRequest

@serve.deployment(
    num_replicas=1,
    placement_group_bundles=[{"CPU": 2, "GPU": 1}],
    placement_group_strategy="STRICT_PACK",
)
class MainAgentChat():
    async def __init__(self):
        self.llm_config = LLMConfig(
            model_loading_config=dict(
                model_id="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
                model_source="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
            ),
            deployment_config=dict(
                autoscaling_config=dict(
                    min_replicas=1,
                    max_replicas=1,
                ),
            ),
            # You can customize the engine arguments (e.g. vLLM engine kwargs)
            engine_kwargs=dict(
                task="generate",
                tensor_parallel_size=1,
                max_model_len=20000,
                gpu_memory_utilization=0.4,
            ),
        )

        self.deployment = LLMServer.as_deployment(self.llm_config.get_serve_options(name_prefix="mainLLM:")).options(
            placement_group_bundles=[{"CPU": 1}, {"CPU": 1, "GPU": 1}], placement_group_strategy="STRICT_PACK"
        ).bind(self.llm_config)

        return None
        

    async def __call__(self, request: ChatCompletionRequest):
        # return await self.router.chat(request)
        return "hi"
