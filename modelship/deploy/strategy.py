import math
import time
from dataclasses import dataclass
from typing import Any

import ray
from ray import serve
from ray.serve.schema import LoggingConfig

from modelship.deploy.actor_options import build_deployment_options, total_cpu_reservation, total_gpu_reservation
from modelship.deploy.removal import delete_apps_quietly
from modelship.infer.infer_config import ModelshipConfig, ModelshipModelConfig
from modelship.infer.model_deployment import ModelDeployment
from modelship.logging import get_logger

logger = get_logger("startup")

_DEPLOY_RETRY_SLEEP_S = 2.0
# Only a pass that reached serve.run and raised consumes one; a skip never does.
_MAX_TRANSIENT_FAILURES = 3
_WAITING_LOG_EVERY_N_PASSES = 30  # with 2s sleep, log "still waiting" once per minute


@dataclass
class DeployPlan:
    """Result of diffing models.yaml against the cluster."""

    models_to_add: list[ModelshipModelConfig]
    apps_to_remove: list[str]
    # Dropped deployments with no live app to delete — only their stale coordinator
    # registry entry needs clearing, or the gateway routes to a ghost.
    registry_only_drop: list[str]


def compute_deploy_plan(
    desired_conf: ModelshipConfig,
    existing_apps: set[str],
    prev_effective_names: set[str],
    gateway_name: str,
) -> DeployPlan:
    """Diff the desired effective set against what's live. The merge verb already
    folded additive/reconcile into `desired_conf`, so this always reconciles
    live -> desired. Deployment names are `{model}-{fingerprint}`, so a set
    comparison detects renames and config drift.

    Removal is `prev_effective_names & existing_apps`: only deployments THIS
    gateway previously managed are removed, never untracked ones or another
    gateway's. An empty prev-effective set removes nothing."""

    # Sort key: footprint desc, whole-GPU before fractional, larger fraction first.
    def _gpu_footprint(c: ModelshipModelConfig) -> tuple[int, bool, float]:
        world_size = (
            c.vllm_engine_kwargs.tensor_parallel_size * c.vllm_engine_kwargs.pipeline_parallel_size
            if c.vllm_engine_kwargs
            else 1
        )
        footprint = max(world_size, math.ceil(c.num_gpus))
        fractional = 0 < c.num_gpus < 1
        return (footprint, not fractional, c.num_gpus)

    sorted_models = sorted(desired_conf.models, key=_gpu_footprint, reverse=True)

    desired_names = {c.deployment_name(gateway_name) for c in sorted_models}

    # Split the dropped set by liveness: live ones get serve.delete + a registry
    # drop, the rest a registry-only drop.
    dropped = prev_effective_names - desired_names
    apps_to_remove = sorted(dropped & existing_apps)
    registry_only_drop = sorted(dropped - existing_apps)
    if apps_to_remove:
        logger.info("Reconcile: %d deployment(s) to remove: %s", len(apps_to_remove), apps_to_remove)
    if registry_only_drop:
        logger.info(
            "Reconcile: %d stale registry entr(ies) to drop (no live app): %s",
            len(registry_only_drop),
            registry_only_drop,
        )

    # Already live under its fingerprint -> skip, so re-runs are idempotent and a
    # matching untracked deployment is adopted rather than redeployed.
    models_to_add = [c for c in sorted_models if c.deployment_name(gateway_name) not in existing_apps]
    if models_to_add:
        logger.info(
            "%d deployment(s) to add: %s",
            len(models_to_add),
            [c.deployment_name(gateway_name) for c in models_to_add],
        )
    return DeployPlan(
        models_to_add=models_to_add,
        apps_to_remove=apps_to_remove,
        registry_only_drop=registry_only_drop,
    )


@dataclass
class DeployContext:
    coordinator: Any
    replica_coordinator: Any
    probe: Any
    operator_id: str
    gateway_name: str
    serve_logging_config: LoggingConfig
    deployed_this_run: dict[str, str]


def try_reserve_and_deploy(config: ModelshipModelConfig, ctx: DeployContext) -> tuple[str, str | None]:
    """One attempt at deploying *config*. Returns (status, detail) where status is:
    "skipped" (no progress, retry), "deployed", "transient" (deploy raised; retry,
    detail is the exception), "fatal" (deployment reported a permanent error; skip
    permanently)."""
    deploy_opts = build_deployment_options(config)
    deployment_name = config.deployment_name(ctx.gateway_name)

    reserved, _reason = ray.get(
        ctx.coordinator.try_reserve.remote(
            ctx.operator_id,
            deployment_name,
            total_gpu_reservation(deploy_opts),
            total_cpu_reservation(deploy_opts),
            ctx.probe,
        )
    )
    if not reserved:
        return "skipped", None

    # Mutually exclusive, enforced at config validation — pass Serve exactly one.
    if config.autoscaling_config is not None:
        scaling_opts: dict = {"autoscaling_config": config.autoscaling_config.to_serve_dict()}
    else:
        scaling_opts = {"num_replicas": config.num_replicas}

    try:
        logger.info("Deploying model: %s (deployment: %s)", config.name, deployment_name)
        ctx.deployed_this_run[deployment_name] = config.name
        serve.run(
            ModelDeployment.options(
                name=deployment_name,
                max_constructor_retry_count=1,
                logging_config=ctx.serve_logging_config,
                **scaling_opts,
                **deploy_opts,
            ).bind(config),
            name=deployment_name,
            route_prefix=None,
        )
        logger.info("Model ready: %s (deployment: %s)", config.name, deployment_name)
        # Registering bumps the gateway's generation; replica watch loops pick the
        # deployment up from there, so the driver never pushes to them.
        try:
            ray.get(ctx.replica_coordinator.register_deployment.remote(ctx.gateway_name, deployment_name, config.name))
        except Exception:
            logger.exception("Failed to record %s in deploy registry", deployment_name)
        return "deployed", None
    except Exception as exc:
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
            # serve.run leaves the application behind when it raises.
            delete_apps_quietly([deployment_name])
            return "fatal", str(fatal_err)
        logger.exception(
            "Deploy failed for %s (deployment=%s); will retry next pass.",
            config.name,
            deployment_name,
        )
        return "transient", f"{type(exc).__name__}: {exc}"
    finally:
        # Ray may already be shut down; OperatorProbe death-detection frees the
        # lock either way once the driver dies.
        if ray.is_initialized():
            try:
                ray.get(ctx.coordinator.release.remote(ctx.operator_id))
            except Exception:
                logger.exception("Failed to release coordinator lock (operator=%s)", ctx.operator_id)


def run_deploy_loop(
    models: list[ModelshipModelConfig],
    ctx: DeployContext,
) -> tuple[int, list[tuple[ModelshipModelConfig, str]]]:
    """Retry-pass loop: each pass tries every not-yet-deployed model, in configured
    order (TP>1 first). A model skipped for resources or a held lock is retried
    indefinitely. One whose deploy raises gets `_MAX_TRANSIENT_FAILURES` attempts,
    the pass sleep doubling each time, then is given up on as fatal — without the
    cap it holds this loop, and the caller's removals and effective-config write,
    forever.

    Returns (pass_count, fatally_failed), pairing each permanently-failed config
    with its error detail. The caller logs them and keeps them in the effective
    config, so a later deploy retries."""
    remaining = list(models)
    fatally_failed: list[tuple[ModelshipModelConfig, str]] = []
    failures: dict[str, int] = {}
    pass_count = 0
    passes_with_no_progress = 0

    while remaining:
        pass_count += 1
        made_progress = False
        for config in list(remaining):
            deployment_name = config.deployment_name(ctx.gateway_name)
            status, detail = try_reserve_and_deploy(config, ctx)
            if status == "deployed":
                remaining.remove(config)
                made_progress = True
            elif status == "fatal":
                fatally_failed.append((config, detail or ""))
                remaining.remove(config)
                made_progress = True
            elif status == "transient":
                failures[deployment_name] = failures.get(deployment_name, 0) + 1
                if failures[deployment_name] >= _MAX_TRANSIENT_FAILURES:
                    logger.error(
                        "Giving up on model '%s' after %d failed attempt(s) (deployment=%s): %s",
                        config.name,
                        failures[deployment_name],
                        deployment_name,
                        detail,
                    )
                    delete_apps_quietly([deployment_name])
                    fatally_failed.append((config, detail or ""))
                    remaining.remove(config)
                    made_progress = True
            # "skipped" -> stays in `remaining` for the next pass

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
            # Back off only for models that failed; a capacity wait keeps the base cadence.
            worst = max(failures.get(c.deployment_name(ctx.gateway_name), 0) for c in remaining)
            time.sleep(_DEPLOY_RETRY_SLEEP_S * 2**worst)

    return pass_count, fatally_failed
