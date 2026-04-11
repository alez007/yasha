import importlib
import os
import signal
import sys

import ray
from pydantic_yaml import parse_yaml_raw_as
from ray import serve
from ray.serve.config import HTTPOptions

from yasha.infer.infer_config import ModelLoader, YashaConfig, YashaModelConfig
from yasha.infer.model_deployment import ModelDeployment
from yasha.logging import configure_logging, get_logger
from yasha.openai.api import YashaAPI

_cache_dir = os.environ.get("YASHA_CACHE_DIR", "/yasha/.cache/models")
_cache_root = os.path.dirname(_cache_dir)
_cache_env_vars = {
    "HF_HOME": os.environ.get("HF_HOME", f"{_cache_root}/huggingface"),
    "VLLM_CACHE_ROOT": os.environ.get("VLLM_CACHE_ROOT", f"{_cache_root}/vllm"),
    "FLASHINFER_CACHE_DIR": os.environ.get("FLASHINFER_CACHE_DIR", f"{_cache_root}/flashinfer"),
}

logger = get_logger("startup")


def build_actor_options(config: YashaModelConfig) -> dict:
    tp = config.vllm_engine_kwargs.tensor_parallel_size if config.vllm_engine_kwargs else 1

    tp_backend = config.vllm_engine_kwargs.distributed_executor_backend if config.vllm_engine_kwargs else None

    if config.num_gpus == 0:
        num_gpus = 0
    elif tp > 1 and tp_backend == "mp":
        # mp backend: the main actor forks TP worker subprocesses, each owning one
        # physical GPU. Allocate tp whole units so Ray exposes all devices via
        # CUDA_VISIBLE_DEVICES. num_gpus is not used for Ray allocation in this path —
        # gpu_memory_utilization is passed directly to vLLM instead.
        if config.num_gpus > 0:
            logger.warning(
                "num_gpus=%s is ignored for model '%s': with vllm mp backend and "
                "tensor_parallel_size=%d, Ray GPU allocation is determined by tp.",
                config.num_gpus,
                config.name,
                tp,
            )
        num_gpus = float(tp)
    elif tp > 1:
        # ray backend: vLLM spawns tp worker Ray actors that each claim their own
        # fractional GPU. The outer actor is a coordinator only and needs no GPU
        # units. VLLM_RAY_PER_WORKER_GPUS tells vLLM what fraction each worker
        # actor should request.
        num_gpus = 0
    else:
        num_gpus = config.num_gpus

    options: dict = {"num_gpus": num_gpus, "num_cpus": config.num_cpus}

    env_vars = dict(_cache_env_vars)

    if tp > 1 and tp_backend != "mp" and config.num_gpus > 0:
        # ray backend: set per-worker GPU fraction so vLLM worker actors claim the
        # right amount. Only override when num_gpus is explicitly set; otherwise
        # let vLLM use its built-in default (0.9/worker).
        env_vars["VLLM_RAY_PER_WORKER_GPUS"] = str(config.num_gpus)

    options["runtime_env"] = {"env_vars": env_vars}

    return options


def ensure_plugin(module_name: str):
    try:
        importlib.import_module(module_name)
    except ImportError as err:
        raise RuntimeError(f"Plugin '{module_name}' is not installed. Run: uv sync --extra {module_name}") from err


def main():
    configure_logging()
    ray_cluster_address = os.environ["RAY_CLUSTER_ADDRESS"]
    ray_redis_port = os.environ["RAY_REDIS_PORT"]
    use_existing_cluster = os.environ.get("YASHA_USE_EXISTING_RAY_CLUSTER", "false").lower() == "true"
    os.environ.setdefault("RAY_GCS_RPC_TIMEOUT_S", "30")
    serve.shutdown()
    ray.shutdown()
    ray_address = f"{ray_cluster_address}:{ray_redis_port}" if use_existing_cluster else "auto"
    ray.init(address=ray_address)
    serve.start(http_options=HTTPOptions(host="0.0.0.0"))

    _config_dir = os.path.dirname(os.path.abspath(__file__)) + "/config"
    _config_file = _config_dir + "/models.yaml"
    if not os.path.exists(_config_file):
        raise FileNotFoundError(
            f"{_config_file} not found. Copy one of the example configs from config/ to config/models.yaml."
        )

    with open(_config_file) as f:
        yml_conf: YashaConfig = parse_yaml_raw_as(YashaConfig, f)

    logger.info("Init yasha app with config: %s", yml_conf)

    for config in yml_conf.models:
        if config.loader == ModelLoader.custom and config.plugin:
            ensure_plugin(config.plugin)

    # Schedule TP>1 models first so they claim whole GPU units before fractional
    # models consume the pool.
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
            model_handles[config.name] = (handle, config.usecase)

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
            logger.info("Shutting down (signal %s)...", sig)
            try:
                serve.shutdown()
            except Exception:
                logger.exception("serve.shutdown() failed")
            try:
                ray.shutdown()
            except Exception:
                logger.exception("ray.shutdown() failed")
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
        signal.pause()
    except BaseException as e:
        if isinstance(e, SystemExit):
            raise
        logger.exception("Startup failed, shutting down serve actors...")
        try:
            serve.shutdown()
        except Exception:
            logger.exception("serve.shutdown() failed during error cleanup")
        try:
            ray.shutdown()
        except Exception:
            logger.exception("ray.shutdown() failed during error cleanup")
        raise


if __name__ == "__main__":
    main()
