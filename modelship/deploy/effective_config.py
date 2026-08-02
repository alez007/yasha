"""Per-gateway *effective config* — the durable desired-state for deploys.

Every ``mship_deploy`` invocation, whatever its mode, folds the user's input into
the gateway's effective set (additive = union; reconcile = replace), then
the deploy ALWAYS reconciles the live cluster to that effective set. Self-heal is
then just "re-run the deploy": it reads the persisted effective set and reconciles
onto an empty cluster, restoring the TRUE live set after the cluster dies — not
just whatever the last user input happened to contain.

The store holds **raw, user-equivalent model dicts**, NOT serialized validated
configs: ``ModelshipModelConfig``'s ``num_gpus``/``tensor_parallel_size``
normalization is not idempotent, so a dumped validated config fails (or silently
mutates its fingerprint) on reload. Raw input dicts reload exactly as written.

This is the deploy-domain layer over the generic ``modelship.state`` store.
"""

from typing import Literal

from modelship.infer.infer_config import ModelshipConfig, ModelshipModelConfig
from modelship.logging import get_logger
from modelship.state import StateStore

logger = get_logger("startup")

DeployMode = Literal["additive", "reconcile"]

# State-store namespace; one key per gateway: "effective/<gateway-name>".
_NAMESPACE = "effective"


def resolve_mode(*, reconcile: bool) -> DeployMode:
    """Map the CLI flags to the effective-config merge verb."""
    return "reconcile" if reconcile else "additive"


def _deployment_name(raw: dict, gateway_name: str) -> str:
    """Deployment name (name + fingerprint) for a raw model dict — the identity
    key for additive de-dup and fatal-failure eviction. Validates the dict
    (running normalization) so two raw dicts that normalize identically map to the
    same deployment."""
    return ModelshipModelConfig.model_validate(raw).deployment_name(gateway_name)


def _model_name(raw: dict) -> str:
    """Human-facing model name for a raw model dict."""
    return ModelshipModelConfig.model_validate(raw).name


def _reject_duplicate_names(raw_models: list[dict], gateway_name: str) -> None:
    """Raise if two dicts share a model name but normalize to different
    deployment_names; identical normalizations are allowed through."""
    by_name: dict[str, str] = {}
    for raw in raw_models:
        model_name = _model_name(raw)
        dep_name = _deployment_name(raw, gateway_name)
        prior = by_name.get(model_name)
        if prior is not None and prior != dep_name:
            raise ValueError(f"duplicate model name {model_name!r}: a model name maps to exactly one deployment.")
        by_name[model_name] = dep_name


def merge(
    effective_raw: list[dict],
    input_raw: list[dict],
    gateway_name: str,
    mode: DeployMode,
) -> list[dict]:
    """Fold the user's input into the effective raw model set under *mode*.

    - additive: replace-by-name — identical config (same deployment_name) is an
      idempotent skip; a different config sharing a model name replaces the
      existing entry for that name rather than joining it.
    - reconcile: input replaces the effective set entirely.

    Raises ValueError if *input_raw* itself declares two entries with the same
    model name and different configs.
    """
    _reject_duplicate_names(input_raw, gateway_name)
    if mode == "reconcile":
        return list(input_raw)

    merged = list(effective_raw)
    for d in input_raw:
        name = _deployment_name(d, gateway_name)
        if any(_deployment_name(m, gateway_name) == name for m in merged):
            continue
        model_name = _model_name(d)
        merged = [m for m in merged if _model_name(m) != model_name]
        merged.append(d)
    return merged


def deployment_names(raw_models: list[dict], gateway_name: str) -> set[str]:
    """The deployment-name set for raw model dicts — the identity set of what's
    under this gateway's effective management. Passed to the deploy plan so a
    reconcile only removes deployments that WERE effective-managed (never legacy /
    un-tracked deployments or another gateway's apps). Relies on the effective
    config being per-gateway and the gateway being folded into each fingerprint."""
    return {_deployment_name(d, gateway_name) for d in raw_models}


def to_config(raw_models: list[dict]) -> ModelshipConfig:
    """Validate raw model dicts into a ModelshipConfig for the deploy path."""
    return ModelshipConfig.model_validate({"models": raw_models})


def read_effective(store: StateStore, gateway_name: str) -> list[dict]:
    """Return the persisted effective raw model set for *gateway_name* (empty if
    none yet)."""
    data = store.get(f"{_NAMESPACE}/{gateway_name}")
    if not isinstance(data, dict):
        return []
    models = data.get("models", [])
    if not isinstance(models, list):
        logger.warning("Effective config for gateway %r has non-list 'models'; treating as empty.", gateway_name)
        return []
    return models


def write_effective(store: StateStore, gateway_name: str, raw_models: list[dict]) -> None:
    """Persist the effective raw model set for *gateway_name*."""
    store.set(f"{_NAMESPACE}/{gateway_name}", {"models": raw_models})
    logger.info("Effective config for gateway %r now has %d model(s).", gateway_name, len(raw_models))
