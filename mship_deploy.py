import logging
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Must be set BEFORE `import ray`: ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV
# is a module-level constant, evaluated at ray's import time. Leaving it on
# makes Ray auto-inject `py_executable="uv run --python X"` whenever the driver
# runs under `uv run`, which overrides the per-job virtualenv that
# runtime_env.pip creates for plugin wheels — breaking plugin imports in the
# worker. Keep the uv hook off so runtime_env.pip's py_executable survives.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "false")

# Set RAY_LOG_LEVEL/RAY_SERVE_LOG_LEVEL/VLLM_LOGGING_LEVEL/TRANSFORMERS_VERBOSITY
# from MSHIP_LOG_LEVEL BEFORE `import ray` — Ray's loggers latch the env value
# at import time, so configuring them later (in configure_logging) is too late
# for the driver process. The level is env-var-only (no CLI flag) since argv
# is parsed inside main(), well after `import ray`.
from modelship.logging import propagate_lib_log_env

propagate_lib_log_env()

import ray  # noqa: E402
from pydantic_yaml import parse_yaml_raw_as  # noqa: E402
from ray import serve  # noqa: E402
from ray.serve.config import HTTPOptions  # noqa: E402
from ray.serve.schema import LoggingConfig  # noqa: E402

from modelship.actor_options import build_actor_options, resolve_plugin_wheel  # noqa: E402
from modelship.infer.deploy_coordinator import OperatorProbe, get_or_create_coordinator  # noqa: E402
from modelship.infer.infer_config import ModelLoader, ModelshipConfig, ModelshipModelConfig  # noqa: E402
from modelship.infer.model_deployment import ModelDeployment  # noqa: E402
from modelship.logging import configure_logging, get_lib_log_config, get_logger  # noqa: E402
from modelship.openai.api import ModelshipAPI  # noqa: E402
from modelship.utils import rand_suffix  # noqa: E402
from modelship.utils.cli import apply_args_to_env, parse_args  # noqa: E402

_DEPLOY_RETRY_SLEEP_S = 2.0
_WAITING_LOG_EVERY_N_PASSES = 30  # with 2s sleep, log "still waiting" once per minute
_DEFAULT_GATEWAY_NAME = "modelship api"
_DEFAULT_OPENAI_API_PORT = 8000

logger = get_logger("startup")


# ---------- small Ray / Serve helpers ----------------------------------------


def _make_operator_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{rand_suffix(4)}"


def _get_existing_apps() -> set[str]:
    """Return the set of currently deployed Serve app names."""
    try:
        return set(serve.status().applications.keys())
    except Exception:
        return set()


def _shutdown_ray() -> None:
    """Shut down Ray Serve and Ray. Logs but swallows errors."""
    for label, fn in (("serve.shutdown()", serve.shutdown), ("ray.shutdown()", ray.shutdown)):
        try:
            fn()
        except Exception:
            logger.exception("%s failed", label)


def _delete_apps_quietly(app_names) -> None:
    """Best-effort serve.delete for cleanup paths — never raises."""
    for name in app_names:
        try:
            logger.info("Deleting deployment: %s", name)
            serve.delete(name)
        except Exception:
            logger.exception("Failed to delete deployment: %s", name)


def _remove_apps(gateway_handle, app_names: list[str]) -> None:
    """Unregister the given deployment apps from the gateway (so new requests
    stop routing) and then delete them from Ray Serve. `serve.delete` drains
    in-flight requests before tearing the deployment down. The deploy
    coordinator is intentionally not involved — it gates admission, not
    teardown; freed resources show up on the next try_reserve."""
    if not app_names:
        return
    try:
        gateway_handle.remove_deployments.remote(app_names).result()
    except Exception:
        logger.exception("Failed to unregister deployments from gateway: %s", app_names)
    _delete_apps_quietly(app_names)


# ---------- startup phases ---------------------------------------------------


def _connect_ray(lib_level: int) -> None:
    ray_cluster_address = os.environ["RAY_CLUSTER_ADDRESS"]
    ray_redis_port = os.environ["RAY_REDIS_PORT"]
    use_existing_cluster = os.environ.get("MSHIP_USE_EXISTING_RAY_CLUSTER", "false").lower() == "true"
    os.environ.setdefault("RAY_GCS_RPC_TIMEOUT_S", "30")

    address = f"{ray_cluster_address}:{ray_redis_port}" if use_existing_cluster else "auto"
    ray.init(address=address, ignore_reinit_error=True, logging_level=lib_level)
    # ray.init re-sets ray.* loggers, so re-pin them after init.
    logging.getLogger("ray").setLevel(lib_level)
    logging.getLogger("ray._private.worker").setLevel(lib_level)


def _start_serve(serve_logging_config: LoggingConfig) -> None:
    port = int(os.environ.get("MSHIP_OPENAI_API_PORT", str(_DEFAULT_OPENAI_API_PORT)))
    serve.start(
        http_options=HTTPOptions(host="0.0.0.0", port=port),
        logging_config=serve_logging_config,
    )


def _resolve_config_path(arg_path: str | None) -> str:
    path = arg_path or str(Path(__file__).resolve().parent / "config" / "models.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Copy one of the example configs from config/ to config/models.yaml."
        )
    return path


def _load_yaml_config(arg_path: str | None) -> ModelshipConfig:
    with open(_resolve_config_path(arg_path)) as f:
        return parse_yaml_raw_as(ModelshipConfig, f)


def _resolve_all_plugin_wheels(yml_conf: ModelshipConfig) -> dict[str, Path]:
    """Pre-flight: resolve every referenced plugin wheel up front so a missing
    wheel fails the whole startup before any Ray deploy is attempted."""
    wheels: dict[str, Path] = {}
    for cfg in yml_conf.models:
        if cfg.loader == ModelLoader.custom and cfg.plugin and cfg.plugin not in wheels:
            wheels[cfg.plugin] = resolve_plugin_wheel(cfg.plugin)
    return wheels


@dataclass
class _DeployPlan:
    """Result of diffing models.yaml against the cluster."""

    models_to_add: list[ModelshipModelConfig]
    apps_to_remove: list[str]


def _compute_deploy_plan(
    yml_conf: ModelshipConfig,
    existing_apps: set[str],
    gateway_name: str,
    *,
    fresh_install: bool,
    reconcile: bool,
) -> _DeployPlan:
    """Diff desired (models.yaml) against deployed (serve.status). Deployment
    names are `{model_name}-{fingerprint}`, so a pure set comparison detects
    both renames and config drift: a model whose num_gpus changed gets a new
    fingerprint -> new deployment_name -> shows up as both an add and a
    remove (handled as a replace by callers)."""

    # Schedule TP>1 models first so they claim whole GPU units before fractional
    # models consume the pool.
    sorted_models = sorted(
        yml_conf.models,
        key=lambda c: c.vllm_engine_kwargs.tensor_parallel_size if c.vllm_engine_kwargs else 1,
        reverse=True,
    )

    desired_names = {c.deployment_name() for c in sorted_models}
    # `existing_apps` includes the gateway app itself; exclude it so reconcile
    # never targets the gateway for removal.
    deployed_names = existing_apps - {gateway_name}

    if reconcile:
        if fresh_install:
            logger.info("--reconcile: no existing gateway — equivalent to a fresh deploy.")
        apps_to_remove = sorted(deployed_names - desired_names)
        if apps_to_remove:
            logger.info("--reconcile: %d deployment(s) to remove: %s", len(apps_to_remove), apps_to_remove)
    else:
        apps_to_remove = []

    # In all modes, skip configs already deployed under their fingerprint —
    # makes plain re-runs idempotent instead of double-deploying.
    models_to_add = [c for c in sorted_models if c.deployment_name() not in deployed_names]
    if models_to_add:
        logger.info(
            "%d deployment(s) to add: %s",
            len(models_to_add),
            [c.deployment_name() for c in models_to_add],
        )
    return _DeployPlan(models_to_add=models_to_add, apps_to_remove=apps_to_remove)


def _start_gateway(gateway_name: str, serve_logging_config: LoggingConfig) -> None:
    logger.info("Starting API gateway...")
    serve.run(
        ModelshipAPI.options(
            name=gateway_name,
            num_replicas=1,
            ray_actor_options={"num_cpus": 0},
            logging_config=serve_logging_config,
        ).bind(),
        name=gateway_name,
        route_prefix="/",
    )
    logger.info("Gateway up — /health and /readyz now serving.")


def _seed_expected_models(gateway_handle, yml_conf: ModelshipConfig) -> None:
    # Pass the full desired set, not just models_to_add — already-deployed
    # models also count toward "ready".
    try:
        gateway_handle.set_expected_models.remote([c.name for c in yml_conf.models]).result()
    except Exception:
        logger.exception("Failed to seed expected model list on gateway (non-fatal).")


# ---------- deploy loop ------------------------------------------------------


@dataclass
class _DeployContext:
    plugin_wheels: dict[str, Path]
    coordinator: Any
    probe: Any
    operator_id: str
    gateway_handle: Any
    serve_logging_config: LoggingConfig
    deployed_this_run: dict[str, str]


def _try_reserve_and_deploy(config: ModelshipModelConfig, ctx: _DeployContext) -> tuple[str, str | None]:
    """One attempt at deploying *config*. Returns (status, detail) where status is:
    "skipped" (no progress, retry), "deployed", "transient" (deploy raised; retry),
    or "fatal" (deployment reported a permanent error; skip permanently)."""
    wheel = ctx.plugin_wheels.get(config.plugin) if config.plugin else None
    actor_opts = build_actor_options(config, plugin_wheel=wheel)
    deployment_name = config.deployment_name()

    reserved, _reason = ray.get(
        ctx.coordinator.try_reserve.remote(
            ctx.operator_id,
            deployment_name,
            float(actor_opts.get("num_gpus", 0) or 0),
            float(actor_opts.get("num_cpus", 0) or 0),
            ctx.probe,
        )
    )
    if not reserved:
        return "skipped", None

    try:
        logger.info("Deploying model: %s (deployment: %s)", config.name, deployment_name)
        ctx.deployed_this_run[deployment_name] = config.name
        serve.run(
            ModelDeployment.options(
                name=deployment_name,
                num_replicas=config.num_replicas,
                ray_actor_options=actor_opts,
                max_constructor_retry_count=1,
                logging_config=ctx.serve_logging_config,
            ).bind(config),
            name=deployment_name,
            route_prefix=None,
        )
        logger.info("Model ready: %s (deployment: %s)", config.name, deployment_name)
        try:
            ctx.gateway_handle.add_models.remote({deployment_name: config.name}).result()
        except Exception:
            logger.exception("Failed to register %s with gateway", deployment_name)
        return "deployed", None
    except Exception:
        # Did the deployment actively report a fatal init error before dying?
        try:
            fatal_err = ray.get(ctx.coordinator.pop_fatal_error.remote(deployment_name), timeout=2.0)
        except Exception:
            fatal_err = None

        ctx.deployed_this_run.pop(deployment_name, None)
        if fatal_err is not None:
            logger.error(
                "Skipping model '%s' permanently (deployment=%s): %s",
                config.name,
                deployment_name,
                fatal_err,
            )
            try:
                serve.delete(deployment_name)
            except Exception:
                logger.exception("Failed to delete failed deployment: %s", deployment_name)
            return "fatal", str(fatal_err)
        logger.exception(
            "Deploy failed for %s (deployment=%s); will retry next pass.",
            config.name,
            deployment_name,
        )
        return "transient", None
    finally:
        # Ray may already be shut down (e.g. SIGINT cleanup ran _shutdown_ray);
        # the OperatorProbe death-detection will free the lock either way once
        # the driver dies.
        if ray.is_initialized():
            try:
                ray.get(ctx.coordinator.release.remote(ctx.operator_id))
            except Exception:
                logger.exception("Failed to release coordinator lock (operator=%s)", ctx.operator_id)


def _run_deploy_loop(
    models: list[ModelshipModelConfig],
    ctx: _DeployContext,
) -> tuple[int, list[tuple[str, str]]]:
    """Retry-pass loop: each pass tries every not-yet-deployed model. Models
    whose resources don't currently fit (or whose reservation is rejected
    because another operator holds the lock) are skipped and retried on the
    next pass. Placeable models deploy in configured order (TP>1 first)."""
    remaining = list(models)
    fatally_failed: list[tuple[str, str]] = []
    pass_count = 0
    passes_with_no_progress = 0

    while remaining:
        pass_count += 1
        made_progress = False
        for config in list(remaining):
            status, detail = _try_reserve_and_deploy(config, ctx)
            if status == "deployed":
                remaining.remove(config)
                made_progress = True
            elif status == "fatal":
                fatally_failed.append((config.name, detail or ""))
                remaining.remove(config)
                made_progress = True
            # "skipped" / "transient" -> stay in `remaining` for the next pass

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

    return pass_count, fatally_failed


# ---------- main -------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    apply_args_to_env(args)

    configure_logging()
    gateway_name = os.environ.get("MSHIP_GATEWAY_NAME", _DEFAULT_GATEWAY_NAME)
    # Library log level (one step above app level). Used to silence Ray Serve's
    # system actors (controller/proxy/replica access logs) and Ray's driver
    # logger, which both ignore Python-level setLevel from the parent process.
    lib_level, lib_level_name = get_lib_log_config()
    serve_logging_config = LoggingConfig(log_level=lib_level_name)

    if args.redeploy:
        logger.info("--redeploy: tearing down existing deployments...")
        _shutdown_ray()

    _connect_ray(lib_level)
    _start_serve(serve_logging_config)

    existing_apps = set() if args.redeploy else _get_existing_apps()
    fresh_install = gateway_name not in existing_apps
    if existing_apps:
        logger.info("Found existing deployments: %s", ", ".join(sorted(existing_apps)))
    if fresh_install and not args.redeploy:
        logger.info("No existing gateway found — treating as fresh install.")

    yml_conf = _load_yaml_config(args.config)
    logger.info("Init modelship app with config: %s", yml_conf)

    plugin_wheels = _resolve_all_plugin_wheels(yml_conf)
    plan = _compute_deploy_plan(
        yml_conf, existing_apps, gateway_name, fresh_install=fresh_install, reconcile=args.reconcile
    )
    apps_to_remove = list(plan.apps_to_remove)

    # Track deployments created by this invocation: deployment_name -> model_name.
    # Shared with the SIGINT/SIGTERM cleanup handler below via closure.
    deployed_this_run: dict[str, str] = {}

    def _cleanup(sig, _frame) -> None:
        logger.info("Shutting down (signal %s), cleaning up deployments from this run...", sig)
        _delete_apps_quietly(reversed(deployed_this_run))
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
            _start_gateway(gateway_name, serve_logging_config)

        gateway_handle = serve.get_app_handle(gateway_name)
        _seed_expected_models(gateway_handle, yml_conf)

        # stop_start: drop old deployments BEFORE deploying new ones, so the
        # freed resources are available for the deploy loop. Used when the
        # cluster can't fit old + new at the same time.
        if args.replace_strategy == "stop_start":
            _remove_apps(gateway_handle, apps_to_remove)
            apps_to_remove = []

        # Coordinator actor serialises deploys across operators on the same
        # cluster; the probe is driver-owned so Ray force-releases the lock
        # if this process dies ungracefully.
        operator_id = _make_operator_id()
        coordinator = get_or_create_coordinator()
        probe = OperatorProbe.options(num_cpus=0).remote()
        logger.info("Operator id=%s; coordinator acquired.", operator_id)

        ctx = _DeployContext(
            plugin_wheels=plugin_wheels,
            coordinator=coordinator,
            probe=probe,
            operator_id=operator_id,
            gateway_handle=gateway_handle,
            serve_logging_config=serve_logging_config,
            deployed_this_run=deployed_this_run,
        )
        pass_count, fatally_failed = _run_deploy_loop(plan.models_to_add, ctx)

        logger.info(
            "Deploy complete. %d new deployment(s) from this run (over %d pass(es)).",
            len(deployed_this_run),
            pass_count,
        )

        # blue_green: drop old deployments AFTER new ones are live and
        # registered with the gateway. During the brief overlap the gateway
        # round-robins across both old and new handles for the same model,
        # so no requests are lost.
        if apps_to_remove:
            _remove_apps(gateway_handle, apps_to_remove)

        if fatally_failed:
            logger.error(
                "%d model(s) failed to deploy and were skipped — fix config and restart:",
                len(fatally_failed),
            )
            for name, reason in fatally_failed:
                logger.error("  - %s: %s", name, reason)

        if fresh_install:
            # Stay alive as the operator process. _cleanup gracefully deletes
            # each deployment (letting actors run __del__ and clean up child
            # processes like vllm EngineCore) before tearing down Ray.
            signal.pause()

    except BaseException as e:
        if isinstance(e, SystemExit):
            raise
        logger.exception("Startup failed, cleaning up deployments from this run...")
        _delete_apps_quietly(reversed(deployed_this_run))
        if fresh_install:
            _shutdown_ray()
        raise


if __name__ == "__main__":
    main()
