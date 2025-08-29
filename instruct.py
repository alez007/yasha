import os
import ray
from ray import serve
from ray.serve.llm import LLMConfig, LLMServer, LLMRouter, build_openai_app, LLMServingArgs, build_llm_deployment

if ray.is_initialized:
    ray.shutdown()

ray.init(
    address="ray://0.0.0.0:10001",
)

instruct_config = LLMConfig(
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

embed_config = LLMConfig(
    model_loading_config=dict(
        model_id="nomic-ai/nomic-embed-text-v1.5",
        model_source="nomic-ai/nomic-embed-text-v1.5",
    ),
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=1,
            max_replicas=1,
        ),
    ),
    # You can customize the engine arguments (e.g. vLLM engine kwargs)
    engine_kwargs=dict(
        task="embedding",
        tensor_parallel_size=1,
        enforce_eager=True,
        model_impl="transformers",
        trust_remote_code=True
    ),
)

instruct_deployment = LLMServer.as_deployment(instruct_config.get_serve_options(name_prefix="mainLLM:")).options(
    placement_group_bundles=[{"CPU": 1}, {"CPU": 1, "GPU": 1}], placement_group_strategy="STRICT_PACK"
).bind(instruct_config)

embed_deployment = LLMServer.as_deployment(embed_config.get_serve_options(name_prefix="embedLLM:")).options(
    placement_group_bundles=[{"CPU": 1}, {"CPU": 3, "GPU": 1}], placement_group_strategy="STRICT_PACK"
).bind(embed_config)

app = LLMRouter.as_deployment().bind([
    # instruct_deployment,
    embed_deployment
])

serve.run(app, route_prefix="/instruct", name="instruct")
