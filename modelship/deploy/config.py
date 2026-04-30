import os
from pathlib import Path

from pydantic_yaml import parse_yaml_raw_as

from modelship.deploy.actor_options import resolve_plugin_wheel
from modelship.infer.infer_config import ModelLoader, ModelshipConfig


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
