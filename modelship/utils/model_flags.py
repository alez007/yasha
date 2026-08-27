"""The models.yaml keys `--model` surfaces as CLI flags.

Root-level scalars are declared by hand in `cli.py`; everything else — the
nested tuning blocks, plus any root key without a hand-written flag — is
generated from the schema here, so a new field is settable the moment it lands.

A generated flag is named for its config path, hyphenated
(`--llama-server-config.n-ctx`), and takes its value as YAML text: the same
thing you'd write after the colon in the file. Nothing is coerced or checked
on the way through — pydantic sees what YAML parsed, from either surface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

import yaml
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from modelship.utils.config_schema import ModelshipModelConfig

# The models.yaml keys with hand-written flags in cli.py. Single source of truth
# for what those flags cover; everything else in the schema is generated below.
MODEL_ARG_KEYS = (
    "name",
    "model",
    "usecase",
    "loader",
    "num_gpus",
    "num_cpus",
    "num_replicas",
    "max_ongoing_requests",
)


class _Unset:
    """Distinguishes an omitted flag from an explicit `null`, which is a real
    value for the `X | None` fields."""

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset()


@dataclass(frozen=True)
class ModelArg:
    """One generated flag and the config path it writes to."""

    path: tuple[str, ...]
    option: str
    dest: str
    metavar: str
    help: str


def _block_model(annotation: Any) -> type[BaseModel] | None:
    """The BaseModel a field holds, unwrapping `X | None`; None for a leaf."""
    candidates = get_args(annotation) if get_origin(annotation) in (Union, UnionType) else (annotation,)
    for candidate in candidates:
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def _type_label(annotation: Any) -> str:
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "").replace("NoneType", "None")


_METAVARS = {"dict": "MAP", "list": "LIST", "Literal": "VALUE", "Any": "VALUE"}


def _metavar(annotation: Any) -> str:
    """Carries the field's type, so the help text only has to carry its default."""
    base = _type_label(annotation).split(" | ")[0].split("[")[0]
    return _METAVARS.get(base, base.upper())


def _literal_choices(annotation: Any) -> str:
    for candidate in (annotation, *get_args(annotation)):
        if get_origin(candidate) is Literal:
            return " | ".join(str(v) for v in get_args(candidate)) + "; "
    return ""


def _model_arg(path: tuple[str, ...], field: FieldInfo) -> ModelArg:
    if field.is_required():
        help_text = _type_label(field.annotation)
    else:
        help_text = _literal_choices(field.annotation) + f"default: {field.get_default(call_default_factory=True)!r}"
    return ModelArg(
        path=path,
        option="--" + ".".join(path).replace("_", "-"),
        # Prefixed and dot-free so it can't collide with a hand-written dest.
        dest="m__" + "__".join(path),
        metavar=_metavar(field.annotation),
        help=help_text,
    )


def _generate() -> tuple[ModelArg, ...]:
    args: list[ModelArg] = []
    for name, field in ModelshipModelConfig.model_fields.items():
        block = _block_model(field.annotation)
        if block is None:
            if name not in MODEL_ARG_KEYS:
                args.append(_model_arg((name,), field))
            continue
        args.extend(_model_arg((name, sub_name), sub_field) for sub_name, sub_field in block.model_fields.items())
    return tuple(args)


GENERATED_MODEL_ARGS = _generate()


def _yaml_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise argparse.ArgumentTypeError(f"not valid YAML: {exc}") from exc


def add_generated_model_args(parser: argparse.ArgumentParser) -> None:
    groups: dict[str, argparse._ArgumentGroup] = {}
    for arg in GENERATED_MODEL_ARGS:
        block = arg.path[0] if len(arg.path) > 1 else ""
        group = groups.get(block)
        if group is None:
            title = f"models.yaml `{block}:` block" if block else "single-model deploy, other root keys"
            group = groups[block] = parser.add_argument_group(title)
        group.add_argument(
            arg.option, dest=arg.dest, type=_yaml_value, default=UNSET, metavar=arg.metavar, help=arg.help
        )


def set_generated_options(args: argparse.Namespace) -> tuple[str, ...]:
    """The generated flags actually passed, named as the user typed them."""
    return tuple(arg.option for arg in GENERATED_MODEL_ARGS if getattr(args, arg.dest, UNSET) is not UNSET)


def apply_generated_args(args: argparse.Namespace, raw: dict) -> None:
    """Fold the set flags into `raw` at their config paths. A block dict is
    created only when one of its keys is set, so an untouched block stays absent
    rather than empty — `model_fields_set` and the fingerprint both read it."""
    for arg in GENERATED_MODEL_ARGS:
        value = getattr(args, arg.dest, UNSET)
        if value is UNSET:
            continue
        node = raw
        for key in arg.path[:-1]:
            node = node.setdefault(key, {})
        node[arg.path[-1]] = value
