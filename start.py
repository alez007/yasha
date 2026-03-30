import os
import logging

_cache_dir = os.environ.get("YASHA_CACHE_DIR", "/yasha/.cache/models")
_cache_root = os.path.dirname(_cache_dir)
_cache_env_vars = {
    "HF_HOME": os.environ.get("HF_HOME", f"{_cache_root}/huggingface"),
    "VLLM_CACHE_ROOT": os.environ.get("VLLM_CACHE_ROOT", f"{_cache_root}/vllm"),
    "FLASHINFER_CACHE_DIR": os.environ.get("FLASHINFER_CACHE_DIR", f"{_cache_root}/flashinfer"),
}

import ray
from ray import serve
from ray.serve.config import HTTPOptions
from pydantic_yaml import parse_yaml_raw_as

from yasha.infer.infer_config import YashaConfig, YashaModelConfig
from yasha.infer.model_deployment import ModelDeployment
from yasha.openai.api import YashaAPI, app

logger = logging.getLogger("ray")


def build_actor_options(config: YashaModelConfig) -> dict:
    tp = config.vllm_engine_kwargs.tensor_parallel_size if config.vllm_engine_kwargs else 1

    # Multiply by tp so Ray reserves the correct total GPU units across all shards
    # (e.g. num_gpus=0.82, tp=2 → 1.64 units).  Claiming float(tp)=2.0 would leave
    # no room for the small fractional models and cause a scheduling failure.
    num_gpus = config.num_gpus * tp

    options: dict = {"num_gpus": num_gpus, "num_cpus": config.num_cpus}

    env_vars = dict(_cache_env_vars)
    if isinstance(config.use_gpu, int):
        env_vars["CUDA_VISIBLE_DEVICES"] = str(config.use_gpu)
    elif isinstance(config.use_gpu, str):
        options["resources"] = {config.use_gpu: 1}
    options["runtime_env"] = {"env_vars": env_vars}

    return options


serve.shutdown()
serve.start(
    http_options=HTTPOptions(host="0.0.0.0")
)


def yasha_app() -> serve.Application:
    _config_dir = os.path.dirname(os.path.abspath(__file__)) + "/config"
    _config_file = _config_dir + "/models.yaml"
    if not os.path.exists(_config_file):
        raise FileNotFoundError(
            f"{_config_file} not found. "
            f"Copy config/models.example.yaml to config/models.yaml and configure your models."
        )

    with open(_config_file, "r") as f:
        yml_conf: YashaConfig = parse_yaml_raw_as(YashaConfig, f)

    logger.info("Init yasha app with config: %s", yml_conf)

    model_handles = {}
    for config in yml_conf.models:
        handle = ModelDeployment.options(
            name=config.name,
            ray_actor_options=build_actor_options(config),
            max_constructor_retry_count=1
        ).bind(config)
        model_handles[config.name] = handle

    return YashaAPI.options(
        name="yasha api",
        num_replicas=1,
        ray_actor_options={"num_cpus": 1},
        max_constructor_retry_count=1
    ).bind(model_handles)


app = yasha_app()
