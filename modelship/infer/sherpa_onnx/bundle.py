"""Fetch, verify, and extract sherpa_onnx's curated model bundles into the
shared cache dir, via modelship.utils.fetch_and_extract_archive (also used by
modelship/launcher.py for llama.cpp's binary tarball)."""

import os

from modelship.infer.sherpa_onnx.registry import REGISTRY, RegistryDir, RegistryFile, SherpaOnnxRegistryEntry
from modelship.logging import get_logger
from modelship.utils import cache_dir, fetch_and_extract_archive, verify_sha256

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
    """Raises ValueError with an actionable message on any mismatch. Used both
    right after a tarball extraction and directly against a user-supplied
    local directory — same check either way."""
    if not os.path.isdir(bundle_dir):
        raise ValueError(f"sherpa_onnx bundle directory not found: {bundle_dir!r}")

    for slot, file in entry.files.items():
        _check_file(bundle_dir, file, f"files[{slot!r}]")
    for i, file in enumerate(entry.lexicon):
        _check_file(bundle_dir, file, f"lexicon[{i}]")
    for slot, d in entry.dirs.items():
        _check_dir(bundle_dir, d, f"dirs[{slot!r}]")


def _check_file(bundle_dir: str, file: RegistryFile, context: str) -> None:
    path = os.path.join(bundle_dir, file.path)
    if not os.path.isfile(path):
        raise ValueError(f"sherpa_onnx bundle {bundle_dir!r}: missing {context} file {file.path!r}")
    actual_size = os.path.getsize(path)
    if actual_size != file.size:
        raise ValueError(
            f"sherpa_onnx bundle {bundle_dir!r}: {context} file {file.path!r} is {actual_size} bytes, "
            f"expected {file.size} (a truncated download or a `git clone` without LFS often looks like this)"
        )
    if file.sha256 is not None:
        verify_sha256(path, file.sha256)


def _check_dir(bundle_dir: str, d: RegistryDir, context: str) -> None:
    path = os.path.join(bundle_dir, d.path)
    if not os.path.isdir(path):
        raise ValueError(f"sherpa_onnx bundle {bundle_dir!r}: missing {context} directory {d.path!r}")
    actual_count = sum(len(files) for _root, _dirs, files in os.walk(path))
    if actual_count != d.file_count:
        raise ValueError(
            f"sherpa_onnx bundle {bundle_dir!r}: {context} directory {d.path!r} has {actual_count} files, "
            f"expected {d.file_count} (a half-extracted archive often looks like this)"
        )
