"""Fetch, verify, and extract sherpa_onnx's curated model bundles into the
shared cache dir, via modelship.utils.fetch_and_extract_archive (also used by
modelship/launcher.py for llama.cpp's binary tarball)."""

import os

from modelship.infer.sherpa_onnx.registry import REGISTRY, SherpaOnnxRegistryEntry
from modelship.logging import get_logger
from modelship.utils import cache_dir, fetch_and_extract_archive

logger = get_logger("infer.sherpa_onnx.bundle")


def resolve_bundle_dir(model: str) -> tuple[str, SherpaOnnxRegistryEntry]:
    """`model` is a registry name (fetched into the shared cache) or a local
    directory path (used in place, basename must match a registry name —
    config validation already guarantees this)."""
    if model.startswith(("/", "./", "~")):
        path = os.path.expanduser(model)
        name = os.path.basename(path.rstrip("/"))
        entry = REGISTRY[name]
        validate_bundle(path, entry)
        return path, entry

    entry = REGISTRY[model]
    bundle_dir = os.path.join(cache_dir(), "sherpa_onnx", model)
    if os.path.isdir(bundle_dir):
        try:
            validate_bundle(bundle_dir, entry)
            return bundle_dir, entry
        except ValueError:
            logger.warning("cached sherpa_onnx bundle %r failed validation, re-fetching", model)

    archive_path = os.path.join(cache_dir(), "sherpa_onnx", f".{model}.tar.bz2")
    fetch_and_extract_archive(entry.tarball_url, entry.sha256, archive_path, bundle_dir)
    validate_bundle(bundle_dir, entry)
    return bundle_dir, entry


def validate_bundle(bundle_dir: str, entry: SherpaOnnxRegistryEntry) -> None:
    """Raises ValueError with an actionable message if a declared file/dir is
    missing. Used both right after a tarball extraction and directly against a
    user-supplied local directory — same check either way. Existence only:
    the tarball's own sha256 (checked before extraction) already catches a
    corrupt/truncated download."""
    if not os.path.isdir(bundle_dir):
        raise ValueError(f"sherpa_onnx bundle directory not found: {bundle_dir!r}")

    for slot, path in entry.files.items():
        _check_exists(bundle_dir, path, os.path.isfile, f"files[{slot!r}]")
    for i, path in enumerate(entry.lexicon):
        _check_exists(bundle_dir, path, os.path.isfile, f"lexicon[{i}]")
    for slot, path in entry.dirs.items():
        _check_exists(bundle_dir, path, os.path.isdir, f"dirs[{slot!r}]")


def _check_exists(bundle_dir: str, rel_path: str, check, context: str) -> None:
    if not check(os.path.join(bundle_dir, rel_path)):
        raise ValueError(f"sherpa_onnx bundle {bundle_dir!r}: missing {context} {rel_path!r}")
