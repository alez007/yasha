"""Fetch, verify, and extract sherpa_onnx's curated model bundles into the
shared cache dir. Mirrors modelship/launcher.py's llama.cpp tarball
provisioning (download -> verify_sha256 -> extract), except extraction keeps
the tarball's directory structure instead of flattening to basenames."""

import contextlib
import os
import shutil
import tarfile

from modelship.infer.sherpa_onnx.registry import REGISTRY, RegistryDir, RegistryFile, SherpaOnnxRegistryEntry
from modelship.logging import get_logger
from modelship.utils import cache_dir, download, random_uuid, verify_sha256

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

    _fetch_and_extract(model, entry, bundle_dir)
    validate_bundle(bundle_dir, entry)
    return bundle_dir, entry


def _fetch_and_extract(name: str, entry: SherpaOnnxRegistryEntry, bundle_dir: str) -> None:
    root = os.path.join(cache_dir(), "sherpa_onnx")
    os.makedirs(root, exist_ok=True)
    archive_path = os.path.join(root, f".{name}.tar.bz2")

    download(entry.tarball_url, archive_path)
    try:
        verify_sha256(archive_path, entry.sha256)
    except ValueError:
        os.remove(archive_path)  # don't let a retry re-verify the same corrupt bytes
        raise

    tmp_dir = os.path.join(root, f".{name}-{random_uuid()}.tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        with tarfile.open(archive_path) as tar:
            tar.extractall(tmp_dir, filter="data")
        extracted_root = os.path.join(tmp_dir, name)
        if not os.path.isdir(extracted_root):
            raise ValueError(
                f"sherpa_onnx bundle {name!r}: extracted tarball has no top-level {name!r} directory "
                f"(found: {os.listdir(tmp_dir)})"
            )
        with contextlib.suppress(OSError):  # another replica on this node already extracted a valid bundle
            os.replace(extracted_root, bundle_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.remove(archive_path)


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
