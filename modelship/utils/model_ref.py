"""Model-reference parsing. Must stay ray-free and huggingface_hub-free — the
pre-ray CLI path parses refs before the driver imports either.
"""

import os
from pathlib import Path
from typing import NamedTuple

from modelship.utils import is_pathy


class ResolvedSource(NamedTuple):
    """Result of parsing a model reference."""

    source: str  # repo_id or local path
    selector: str | None  # filename or glob pattern
    is_local: bool


def expand(s: str) -> str:
    """expanduser only for pathy strings — Path.resolve() never expands `~`,
    so this must happen here or a valid `~/...` ref 404s downstream."""
    return os.path.expanduser(s) if is_pathy(s) else s


def parse_model_ref(model: str) -> ResolvedSource:
    """Parses model string into (source, selector, is_local).

    Path-first: if the literal full string is an existing local path, treat it
    as one (covers the rare colon-in-filename case). Otherwise split on the
    first ':' — the part before is the source, the part after is the selector.

    A pathy source (starts with /, ./, or ~) is always local regardless of
    whether it exists, so a missing path fails clearly downstream instead of
    being misread as an HF repo id. `~` is expanded in the returned source."""
    expanded = expand(model)
    if is_pathy(model) and Path(expanded).exists():
        return ResolvedSource(source=expanded, selector=None, is_local=True)

    if ":" in model:
        source, selector = model.split(":", 1)
        return ResolvedSource(source=expand(source), selector=selector, is_local=is_pathy(source))

    return ResolvedSource(source=expanded, selector=None, is_local=is_pathy(model))
