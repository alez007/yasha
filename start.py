import os
import signal
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
from yasha.openai.api import YashaAPI

logger = logging.getLogger("ray")


def build_actor_options(config: YashaModelConfig) -> dict:
    tp = config.vllm_engine_kwargs.tensor_parallel_size if config.vllm_engine_kwargs else 1

    if tp > 1 and isinstance(config.use_gpu, str):
        # Named resource provides scheduling exclusivity — no GPU units needed.
        # Ray requires whole numbers for num_gpus > 1, and num_gpus * tp would be
        # fractional here. VRAM is managed by each model's gpu_memory_utilization.
        num_gpus = 0
    elif tp > 1:
        # No named resource: Ray uses GPU units for placement. Must be a whole number.
        # TP models are sorted first so they are scheduled before fractional models
        # consume the pool.
        num_gpus = float(tp)
    else:
        num_gpus = config.num_gpus

    options: dict = {"num_gpus": num_gpus, "num_cpus": config.num_cpus}

    env_vars = dict(_cache_env_vars)
    if isinstance(config.use_gpu, int):
        env_vars["CUDA_VISIBLE_DEVICES"] = str(config.use_gpu)
    elif isinstance(config.use_gpu, str):
        options["resources"] = {config.use_gpu: 1}
        if tp > 1:
            # Tell vLLM's Ray executor what fraction of a GPU each worker actor
            # should claim, so they respect our memory budget rather than
            # requesting 1.0 GPU unit each (the default).
            env_vars["VLLM_RAY_PER_WORKER_GPUS"] = str(config.num_gpus)
    options["runtime_env"] = {"env_vars": env_vars}

    return options


def main():
    ray_port = os.environ.get("RAY_REDIS_PORT", "6379")
    serve.shutdown()
    ray.shutdown()
    ray.init(address=f"0.0.0.0:{ray_port}")
    serve.start(http_options=HTTPOptions(host="0.0.0.0"))

    _config_dir = os.path.dirname(os.path.abspath(__file__)) + "/config"
    _config_file = _config_dir + "/models.yaml"
    if not os.path.exists(_config_file):
        raise FileNotFoundError(
            f"{_config_file} not found. "
            f"Copy one of the example configs from config/ to config/models.yaml."
        )

    with open(_config_file, "r") as f:
        yml_conf: YashaConfig = parse_yaml_raw_as(YashaConfig, f)

    logger.info("Init yasha app with config: %s", yml_conf)

    # Schedule TP>1 models first so they claim whole GPU units before fractional
    # models consume the pool (relevant when use_gpu is not a named resource).
    sorted_models = sorted(
        yml_conf.models,
        key=lambda c: c.vllm_engine_kwargs.tensor_parallel_size if c.vllm_engine_kwargs else 1,
        reverse=True,
    )

    try:
        # Deploy models one at a time. serve.run() blocks until the deployment reaches
        # RUNNING, ensuring each model fully initialises (and releases its load-time
        # memory spike) before the next one starts.
        model_handles = {}
        for config in sorted_models:
            logger.info("Deploying model: %s", config.name)
            handle = serve.run(
                ModelDeployment.options(
                    name=config.name,
                    ray_actor_options=build_actor_options(config),
                ).bind(config),
                name=config.name,
                route_prefix=None,  # not exposed via HTTP — accessed only via handle
            )
            logger.info("Model ready: %s", config.name)
            model_handles[config.name] = handle

        logger.info("All models ready, starting API gateway...")
        serve.run(
            YashaAPI.options(
                name="yasha api",
                num_replicas=1,
                ray_actor_options={"num_cpus": 1},
            ).bind(model_handles),
            name="yasha api",
            route_prefix="/",
        )

        def _shutdown(sig, frame):
            logger.info("Shutting down...")
            serve.shutdown()
            ray.shutdown()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
        signal.pause()
    except Exception:
        logger.exception("Startup failed, shutting down serve actors...")
        serve.shutdown()
        ray.shutdown()
        raise


if __name__ == "__main__":
    main()
