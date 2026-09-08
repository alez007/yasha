"""Deployment teardown, kept free of serve_utils' gateway and probe imports."""

import ray
from ray import serve

from modelship.logging import get_logger

logger = get_logger("startup")


def delete_apps_quietly(app_names) -> None:
    """Best-effort serve.delete for cleanup paths — never raises."""
    for name in app_names:
        try:
            logger.info("Deleting deployment: %s", name)
            serve.delete(name)
        except Exception:
            logger.exception("Failed to delete deployment: %s", name)


def remove_apps(app_names: list[str], replica_coordinator, gateway_name: str) -> None:
    """Drop the given deployment apps from the replica coordinator's ownership registry
    (which bumps the gateway generation so every replica's watch loop stops routing
    to them), then delete them from Ray Serve (`serve.delete` drains in-flight
    requests first)."""
    if not app_names:
        return
    try:
        ray.get([replica_coordinator.unregister_deployment.remote(gateway_name, a) for a in app_names])
    except Exception:
        logger.exception("Failed to drop deployments from registry: %s", app_names)
    delete_apps_quietly(app_names)
