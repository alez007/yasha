import os
from pathlib import Path

from pydantic_yaml import parse_yaml_raw_as

from modelship.deploy.actor_options import resolve_plugin_wheel
from modelship.infer.infer_config import ModelLoader, ModelshipConfig
from modelship.infer.model_resolver import resolve_model_source
from modelship.logging import get_logger

logger = get_logger("startup")


def resolve_config_path(arg_path: str | None) -> str:
    path = arg_path or str(Path(__file__).resolve().parent.parent.parent / "config" / "models.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Copy one of the example configs from config/ to config/models.yaml."
        )
    return path


def load_yaml_config(arg_path: str | None) -> ModelshipConfig:
    with open(resolve_config_path(arg_path)) as f:
        return parse_yaml_raw_as(ModelshipConfig, f)


def resolve_all_plugin_wheels(yml_conf: ModelshipConfig) -> dict[str, Path]:
    """Pre-flight: resolve every referenced plugin wheel up front so a missing
    wheel fails the whole startup before any Ray deploy is attempted."""
    wheels: dict[str, Path] = {}
    for cfg in yml_conf.models:
        if cfg.loader == ModelLoader.custom and cfg.plugin and cfg.plugin not in wheels:
            wheels[cfg.plugin] = resolve_plugin_wheel(cfg.plugin)
    return wheels


def resolve_all_model_sources(yml_conf: ModelshipConfig) -> None:
    """Pre-flight: resolve every built-in-loader model to a local path.

    Populates `_resolved_path` on each config in place. Raises on the first
    failure (auth, missing repo, missing file, glob-no-match) so the operator
    sees the error before any Ray actor spins up.

    Plugins (`loader=custom`) are skipped — they manage their own download.

    Note: HF_HOME / VLLM_CACHE_ROOT / FLASHINFER_CACHE_DIR are set at module
    load time in mship_deploy.py — `huggingface_hub.HF_HOME` is latched at
    import, so setting them later doesn't take effect.
    """
    for cfg in yml_conf.models:
        if cfg.loader == ModelLoader.custom:
            continue
        assert cfg.model is not None  # validator guarantees this for built-in loaders
        trust_remote_code = bool(
            (cfg.vllm_engine_kwargs and cfg.vllm_engine_kwargs.trust_remote_code)
            or (cfg.transformers_config and cfg.transformers_config.trust_remote_code)
        )
        logger.info("Resolving model source for '%s': %s", cfg.name, cfg.model)
        cfg._resolved_path = resolve_model_source(cfg.model, trust_remote_code=trust_remote_code)
        logger.info("Resolved '%s' -> %s", cfg.name, cfg._resolved_path)
