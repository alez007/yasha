from ray import init, serve
from ray.serve.llm import LLMConfig, LLMServer, LLMRouter

llm_config = LLMConfig(
    # model_loading_config=dict(
    #     model_id="Qwen3-4B",
    #     model_source="Qwen/Qwen3-4B-Instruct-2507",
    # ),
    model_loading_config=dict(
        model_id="gemma-3-1b-it",
        model_source="google/gemma-3-1b-it",
    ),
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=1, max_replicas=1,
        )
    ),
    llm_engine="vLLM",
    engine_kwargs={
        "max_model_len": 32768,
        "dtype": "bfloat16"
    },
    # accelerator_type="L4",
)

# Deploy the application
deployment = LLMServer.as_deployment(llm_config.get_serve_options(name_prefix="vLLM:")).bind(llm_config)
app = LLMRouter.as_deployment().bind([deployment])
