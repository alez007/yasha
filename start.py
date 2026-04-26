import argparse
import os
import random
import signal
import socket
import string
import sys
import time
from pathlib import Path

# Must be set BEFORE `import ray`: ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV
# is a module-level constant, evaluated at ray's import time. Leaving it on
# makes Ray auto-inject `py_executable="uv run --python X"` whenever the driver
# runs under `uv run`, which overrides the per-job virtualenv that
# runtime_env.pip creates for plugin wheels — breaking plugin imports in the
# worker. Keep the uv hook off so runtime_env.pip's py_executable survives.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "false")

import ray
from pydantic_yaml import parse_yaml_raw_as
from ray import serve
from ray.serve.config import HTTPOptions

from modelship.infer.deploy_coordinator import OperatorProbe, get_or_create_coordinator
from modelship.infer.infer_config import ModelLoader, ModelshipConfig, ModelshipModelConfig
from modelship.infer.model_deployment import ModelDeployment
from modelship.logging import configure_logging, get_logger
from modelship.openai.api import ModelshipAPI

_DEPLOY_RETRY_SLEEP_S = 2.0
_WAITING_LOG_EVERY_N_PASSES = 30  # with 2s sleep, log "still waiting" once per minute

logger = get_logger("startup")

_RAND_CHARS = string.ascii_lowercase + string.digits


def _rand_suffix(length: int = 5) -> str:
    return "".join(random.choices(_RAND_CHARS, k=length))


def _build_cache_env_vars() -> dict[str, str]:
    base_cache = os.environ.get("MSHIP_CACHE_DIR", "/.cache")
    return {
        "HF_HOME": os.environ.get("HF_HOME", f"{base_cache}/huggingface"),
        "VLLM_CACHE_ROOT": os.environ.get("VLLM_CACHE_ROOT", f"{base_cache}/vllm"),
        "FLASHINFER_CACHE_DIR": os.environ.get("FLASHINFER_CACHE_DIR", f"{base_cache}/flashinfer"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modelship — serve LLMs with Ray Serve")
    parser.add_argument("--ray-cluster-address", help="Ray cluster address (env: RAY_CLUSTER_ADDRESS)")
    parser.add_argument("--ray-redis-port", help="Ray Redis port (env: RAY_REDIS_PORT)")
    parser.add_argument("--config", help="Path to models.yaml config file (default: config/models.yaml)")
    parser.add_argument("--cache-dir", help="Model cache directory (env: MSHIP_CACHE_DIR)")
    parser.add_argument(
        "--gateway-name",
        help="Name for the API gateway app (env: MSHIP_GATEWAY_NAME, default: modelship api)",
    )
    parser.add_argument(
        "--use-existing-ray-cluster",
        action="store_true",
        default=None,
        help="Connect to an existing Ray cluster (env: MSHIP_USE_EXISTING_RAY_CLUSTER)",
    )
    parser.add_argument(
        "--log-level",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (env: MSHIP_LOG_LEVEL)",
    )
    parser.add_argument("--log-format", choices=["text", "json"], help="Log format (env: MSHIP_LOG_FORMAT)")
    parser.add_argument(
        "--log-target",
        help="Log target: 'console' (default) or syslog URI e.g. syslog://host:514, syslog+tcp://host:514 (env: MSHIP_LOG_TARGET)",
    )
    parser.add_argument(
        "--otel-endpoint",
        help="OpenTelemetry OTLP endpoint e.g. http://collector:4317 (env: OTEL_EXPORTER_OTLP_ENDPOINT)",
    )
    parser.add_argument("--no-metrics", action="store_true", default=None, help="Disable metrics (env: MSHIP_METRICS)")
    parser.add_argument("--api-keys", help="Comma-separated API keys (env: MSHIP_API_KEYS)")
    parser.add_argument(
        "--max-request-body-bytes", type=int, help="Max request body size in bytes (env: MSHIP_MAX_REQUEST_BODY_BYTES)"
    )
    parser.add_argument(
        "--openai-api-port",
        type=int,
        help="Port for the OpenAI-compatible API (env: MSHIP_OPENAI_API_PORT, default: 8000)",
    )
    parser.add_argument(
        "--redeploy",
        action="store_true",
        default=False,
        help="Tear down all existing deployments before deploying (default: additive)",
    )
    return parser.parse_args(argv)


def _apply_args_to_env(args: argparse.Namespace) -> None:
    """Set environment variables from CLI args. CLI args take precedence over env vars."""
    _arg_to_env = {
        "ray_cluster_address": "RAY_CLUSTER_ADDRESS",
        "ray_redis_port": "RAY_REDIS_PORT",
        "cache_dir": "MSHIP_CACHE_DIR",
        "log_level": "MSHIP_LOG_LEVEL",
        "log_format": "MSHIP_LOG_FORMAT",
        "log_target": "MSHIP_LOG_TARGET",
        "otel_endpoint": "OTEL_EXPORTER_OTLP_ENDPOINT",
        "api_keys": "MSHIP_API_KEYS",
        "gateway_name": "MSHIP_GATEWAY_NAME",
    }
    for attr, env_var in _arg_to_env.items():
        val = getattr(args, attr, None)
        if val is not None:
            os.environ[env_var] = val

    if args.use_existing_ray_cluster is True:
        os.environ["MSHIP_USE_EXISTING_RAY_CLUSTER"] = "true"

    if args.no_metrics is True:
        os.environ["MSHIP_METRICS"] = "false"

    if args.max_request_body_bytes is not None:
        os.environ["MSHIP_MAX_REQUEST_BODY_BYTES"] = str(args.max_request_body_bytes)

    if args.openai_api_port is not None:
        os.environ["MSHIP_OPENAI_API_PORT"] = str(args.openai_api_port)


def build_actor_options(config: ModelshipModelConfig, plugin_wheel: Path | None = None) -> dict:
    env_vars = _build_cache_env_vars()
    for log_var in ("MSHIP_LOG_LEVEL", "MSHIP_LOG_FORMAT", "MSHIP_LOG_TARGET"):
        val = os.environ.get(log_var)
        if val is not None:
            env_vars[log_var] = val

    if config.loader == ModelLoader.llama_cpp:
        if config.num_gpus > 0:
            logger.warning(
                "num_gpus=%s is ignored for model '%s': llama_cpp loader currently only supports CPU.",
                config.num_gpus,
                config.name,
            )
        num_gpus = 0
    else:
        tp = config.vllm_engine_kwargs.tensor_parallel_size if config.vllm_engine_kwargs else 1
        tp_backend = config.vllm_engine_kwargs.distributed_executor_backend if config.vllm_engine_kwargs else None

        # vLLM validates world_size against the outer actor's visible GPUs before consulting
        # the executor backend. With tp>1 and no backend set, the outer actor has 0 GPUs
        # (ray-backend path below) and ParallelConfig blows up. Default to mp so the outer
        # actor owns all tp GPUs and forks workers as subprocesses.
        if tp > 1 and tp_backend is None:
            config.vllm_engine_kwargs.distributed_executor_backend = "mp"
            tp_backend = "mp"
            logger.info(
                "Defaulting distributed_executor_backend='mp' for model '%s' (tensor_parallel_size=%d).",
                config.name,
                tp,
            )

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
            if config.num_gpus > 0:
                env_vars["VLLM_RAY_PER_WORKER_GPUS"] = str(config.num_gpus)
        else:
            num_gpus = config.num_gpus

    runtime_env: dict = {"env_vars": env_vars}
    if plugin_wheel is not None:
        # Ship the plugin to the Ray worker via runtime_env. Ray content-hashes
        # and caches the resulting per-job venv, so repeat deploys of the same
        # wheel reuse the install.
        runtime_env["pip"] = [str(plugin_wheel)]

    return {
        "num_gpus": num_gpus,
        "num_cpus": config.num_cpus,
        "runtime_env": runtime_env,
    }


def _plugin_wheel_dir() -> Path:
    return Path(os.environ.get("MSHIP_PLUGIN_WHEEL_DIR", ".build/plugin-wheels"))


def resolve_plugin_wheel(plugin: str) -> Path:
    wheel_dir = _plugin_wheel_dir()
    normalized_name = plugin.replace("-", "_")
    wheels = sorted(wheel_dir.glob(f"{normalized_name}-*.whl"))
    if not wheels:
        raise RuntimeError(
            f"No wheel found for plugin '{plugin}' (normalized: '{normalized_name}') in {wheel_dir}. "
            f"Build wheels with `make plugin-wheels` (or rebuild the Docker image), "
            f"or set MSHIP_PLUGIN_WHEEL_DIR to the directory containing them."
        )
    # Absolute path required: Ray workers run with a different cwd
    # (/tmp/ray/session_*/runtime_resources/.../exec_cwd), so a relative wheel
    # path in runtime_env.pip would fail to resolve on the worker.
    return wheels[-1].resolve()


def _make_operator_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{_rand_suffix(4)}"


def _get_existing_apps() -> set[str]:
    """Return the set of currently deployed Serve app names."""
    try:
        status = serve.status()
        return set(status.applications.keys())
    except Exception:
        return set()


def _shutdown_ray():
    """Shut down Ray Serve and Ray. Logs but swallows errors."""
    try:
        serve.shutdown()
    except Exception:
        logger.exception("serve.shutdown() failed")
    try:
        ray.shutdown()
    except Exception:
        logger.exception("ray.shutdown() failed")


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    _apply_args_to_env(args)

    configure_logging()
    ray_cluster_address = os.environ["RAY_CLUSTER_ADDRESS"]
    ray_redis_port = os.environ["RAY_REDIS_PORT"]
    use_existing_cluster = os.environ.get("MSHIP_USE_EXISTING_RAY_CLUSTER", "false").lower() == "true"
    gateway_name = os.environ.get("MSHIP_GATEWAY_NAME", "modelship api")
    os.environ.setdefault("RAY_GCS_RPC_TIMEOUT_S", "30")

    if args.redeploy:
        logger.info("--redeploy: tearing down existing deployments...")
        _shutdown_ray()

    ray_address = f"{ray_cluster_address}:{ray_redis_port}" if use_existing_cluster else "auto"
    ray.init(address=ray_address, ignore_reinit_error=True)
    openai_api_port = int(os.environ.get("MSHIP_OPENAI_API_PORT", "8000"))
    serve.start(http_options=HTTPOptions(host="0.0.0.0", port=openai_api_port))

    existing_apps = set() if args.redeploy else _get_existing_apps()
    fresh_install = gateway_name not in existing_apps
    if existing_apps:
        logger.info("Found existing deployments: %s", ", ".join(sorted(existing_apps)))
    if fresh_install and not args.redeploy:
        logger.info("No existing gateway found — treating as fresh install.")

    if args.config:
        _config_file = args.config
    else:
        _config_dir = os.path.dirname(os.path.abspath(__file__)) + "/config"
        _config_file = _config_dir + "/models.yaml"
    if not os.path.exists(_config_file):
        raise FileNotFoundError(
            f"{_config_file} not found. Copy one of the example configs from config/ to config/models.yaml."
        )

    with open(_config_file) as f:
        yml_conf: ModelshipConfig = parse_yaml_raw_as(ModelshipConfig, f)

    logger.info("Init modelship app with config: %s", yml_conf)

    # Pre-flight: resolve every referenced plugin wheel up front so a missing
    # wheel fails the whole startup before any Ray deploy is attempted.
    plugin_wheels: dict[str, Path] = {}
    for config in yml_conf.models:
        if config.loader == ModelLoader.custom and config.plugin and config.plugin not in plugin_wheels:
            plugin_wheels[config.plugin] = resolve_plugin_wheel(config.plugin)

    # Schedule TP>1 models first so they claim whole GPU units before fractional
    # models consume the pool.
    sorted_models = sorted(
        yml_conf.models,
        key=lambda c: c.vllm_engine_kwargs.tensor_parallel_size if c.vllm_engine_kwargs else 1,
        reverse=True,
    )

    # Track deployments created by this invocation: deployment_name -> model_name.
    # Used for both model registration with the gateway and cleanup on interrupt.
    deployed_this_run: dict[str, str] = {}

    def _cleanup(sig, frame):
        logger.info("Shutting down (signal %s), cleaning up deployments from this run...", sig)
        for name in reversed(deployed_this_run):
            try:
                logger.info("Deleting deployment: %s", name)
                serve.delete(name)
            except Exception:
                logger.exception("Failed to delete deployment: %s", name)
        if fresh_install:
            _shutdown_ray()
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        # Start the gateway FIRST on fresh install so /health, /v1/models, and
        # /readyz are reachable while models are still loading. Models register
        # with the gateway as they come up (incremental add_models calls).
        if fresh_install:
            logger.info("Starting API gateway...")
            serve.run(
                ModelshipAPI.options(
                    name=gateway_name,
                    num_replicas=1,
                    ray_actor_options={"num_cpus": 0},
                ).bind(),
                name=gateway_name,
                route_prefix="/",
            )
            logger.info("Gateway up — /health and /readyz now serving.")

        gateway_handle = serve.get_app_handle(gateway_name)
        try:
            gateway_handle.set_expected_models.remote([c.name for c in sorted_models]).result()
        except Exception:
            logger.exception("Failed to seed expected model list on gateway (non-fatal).")

        # Coordinator actor serialises deploys across operators on the same
        # cluster; the probe is driver-owned so Ray force-releases the lock
        # if this process dies ungracefully.
        operator_id = _make_operator_id()
        coordinator = get_or_create_coordinator()
        probe = OperatorProbe.options(num_cpus=0).remote()
        logger.info("Operator id=%s; coordinator acquired.", operator_id)

        # Retry-pass loop: each pass tries every not-yet-deployed model. Models
        # whose resources don't currently fit (or whose reservation is rejected
        # because another operator holds the lock) are skipped and retried on
        # the next pass. Placeable models deploy in configured order (TP>1 first).
        remaining = list(sorted_models)
        fatally_failed: list[tuple[str, str]] = []
        pass_count = 0
        passes_with_no_progress = 0

        while remaining:
            pass_count += 1
            made_progress = False

            for config in list(remaining):
                wheel = plugin_wheels.get(config.plugin) if config.plugin else None
                actor_opts = build_actor_options(config, plugin_wheel=wheel)
                deployment_name = f"{config.name}-{_rand_suffix()}"

                reserved, _reason = ray.get(
                    coordinator.try_reserve.remote(
                        operator_id,
                        deployment_name,
                        float(actor_opts.get("num_gpus", 0) or 0),
                        float(actor_opts.get("num_cpus", 0) or 0),
                        probe,
                    )
                )
                if not reserved:
                    continue

                try:
                    logger.info("Deploying model: %s (deployment: %s)", config.name, deployment_name)
                    deployed_this_run[deployment_name] = config.name
                    serve.run(
                        ModelDeployment.options(
                            name=deployment_name,
                            num_replicas=config.num_replicas,
                            ray_actor_options=actor_opts,
                            max_constructor_retry_count=1,
                        ).bind(config),
                        name=deployment_name,
                        route_prefix=None,
                    )
                    logger.info("Model ready: %s (deployment: %s)", config.name, deployment_name)
                    try:
                        gateway_handle.add_models.remote({deployment_name: config.name}).result()
                    except Exception:
                        logger.exception("Failed to register %s with gateway", deployment_name)
                    remaining.remove(config)
                    made_progress = True
                except Exception:
                    # Check if the deployment actively reported a fatal init error before dying
                    try:
                        fatal_err = ray.get(coordinator.pop_fatal_error.remote(config.name), timeout=2.0)
                    except Exception:
                        fatal_err = None

                    if fatal_err is not None:
                        # Re-raise to fall into the permanent skip logic instead of retrying
                        logger.error(
                            "Skipping model '%s' permanently (deployment=%s): %s",
                            config.name,
                            deployment_name,
                            fatal_err,
                        )
                        deployed_this_run.pop(deployment_name, None)
                        try:
                            serve.delete(deployment_name)
                        except Exception:
                            logger.exception("Failed to delete failed deployment: %s", deployment_name)
                        fatally_failed.append((config.name, str(fatal_err)))
                        remaining.remove(config)
                        made_progress = True
                    else:
                        logger.exception(
                            "Deploy failed for %s (deployment=%s); will retry next pass.",
                            config.name,
                            deployment_name,
                        )
                        deployed_this_run.pop(deployment_name, None)
                finally:
                    try:
                        ray.get(coordinator.release.remote(operator_id))
                    except Exception:
                        logger.exception("Failed to release coordinator lock (operator=%s)", operator_id)

            if made_progress:
                passes_with_no_progress = 0
            else:
                passes_with_no_progress += 1
                if passes_with_no_progress == 1 or passes_with_no_progress % _WAITING_LOG_EVERY_N_PASSES == 0:
                    logger.info(
                        "Waiting for capacity for %d model(s): %s",
                        len(remaining),
                        [c.name for c in remaining],
                    )

            if remaining:
                time.sleep(_DEPLOY_RETRY_SLEEP_S)

        logger.info(
            "Deploy complete. %d new deployment(s) from this run (over %d pass(es)).",
            len(deployed_this_run),
            pass_count,
        )

        if fatally_failed:
            logger.error(
                "%d model(s) failed to deploy and were skipped — fix config and restart:",
                len(fatally_failed),
            )
            for name, reason in fatally_failed:
                logger.error("  - %s: %s", name, reason)

        if fresh_install:
            # On fresh install, stay alive as the operator process.
            # Reuse _cleanup which gracefully deletes each deployment (letting actors
            # run __del__ and clean up child processes like vllm EngineCore) before
            # tearing down Ray.
            signal.signal(signal.SIGINT, _cleanup)
            signal.signal(signal.SIGTERM, _cleanup)
            signal.pause()

    except BaseException as e:
        if isinstance(e, SystemExit):
            raise
        logger.exception("Startup failed, cleaning up deployments from this run...")
        for name in reversed(deployed_this_run):
            try:
                serve.delete(name)
            except Exception:
                logger.exception("Failed to delete deployment: %s", name)
        if fresh_install:
            _shutdown_ray()
        raise


if __name__ == "__main__":
    main()
