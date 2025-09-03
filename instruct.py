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
        model_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        model_source="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
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
        max_model_len=6000,
        gpu_memory_utilization=0.45
    ),
)

instruct_deployment = LLMServer.as_deployment(instruct_config.get_serve_options(name_prefix="mainLLM:")).options(
    placement_group_bundles=[{"CPU": 1}, {"CPU": 0, "GPU": 0.45}], placement_group_strategy="PACK"
).bind(instruct_config)

app = LLMRouter.as_deployment().bind([
    instruct_deployment,
])


print("======================starting instruct===================")
handle = serve.run(app, route_prefix="/instruct", name="instruct")
# print("======================starting instruct1===================")
# serve.run(app, route_prefix="/instruct1", name="instruct1")
