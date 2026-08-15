"""Stdlib port of `modelship.utils.fetch_and_extract_archive`, which imports
`requests`."""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import tarfile
import urllib.request
import uuid

_SOCKET_TIMEOUT_SECONDS = 30

# Extraction filters: 3.12, backported to 3.10.12 / 3.11.4.
_HAS_EXTRACTION_FILTER = hasattr(tarfile, "data_filter")


class FetchError(RuntimeError):
    pass


def download(url: str, file_path: str, *, overwrite: bool = False) -> None:
    if not overwrite and os.path.isfile(file_path):
        return
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    tmp_path = f"{file_path}.{uuid.uuid4().hex}.tmp"
    try:
        with (
            urllib.request.urlopen(url, timeout=_SOCKET_TIMEOUT_SECONDS) as response,  # noqa: S310
            open(tmp_path, "wb") as f,
        ):
            shutil.copyfileobj(response, f, 1024 * 1024)
        os.replace(tmp_path, file_path)
    except OSError as e:
        raise FetchError(f"failed to download {url}: {e}") from e
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def verify_sha256(path: str, expected: str) -> None:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise FetchError(f"{path} failed sha256 verification: expected {expected}, got {actual}")


def _escapes(path: str) -> bool:
    """True when `path` is absolute or normalizes above its own root."""
    return os.path.isabs(path) or os.path.normpath(path).split(os.sep)[0] == ".."


def _extract(tar: tarfile.TarFile, dest: str, *, flatten: bool) -> None:
    """`filter="data"` where the interpreter has it; the checks it would have applied
    by hand where it doesn't — names and link targets stay under dest, no devices or
    fifos, and modes lose setuid/setgid/sticky and group/other write. Links stay
    allowed: the llama.cpp assets ship SONAME symlinks the binary's rpath needs."""
    members = []
    for member in tar.getmembers():
        if flatten:
            member.name = os.path.basename(member.name)
            if not member.name:
                continue
        elif _escapes(member.name):
            raise FetchError(f"refusing to extract {member.name!r} outside {dest}")

        if member.issym() or member.islnk():
            # A hardlink's target is archive-root-relative; a symlink's is relative to itself.
            base = os.path.dirname(member.name) if member.issym() else ""
            if _escapes(os.path.join(base, member.linkname)):
                raise FetchError(f"refusing to extract {member.name!r} pointing at {member.linkname!r} outside {dest}")
        elif not (member.isfile() or member.isdir()):
            raise FetchError(f"refusing to extract {member.name!r}: not a regular file, directory, or link")

        if not _HAS_EXTRACTION_FILTER:
            member.mode &= 0o755
        members.append(member)

    if _HAS_EXTRACTION_FILTER:
        tar.extractall(dest, members=members, filter="data")
    else:
        tar.extractall(dest, members=members)  # noqa: S202


def fetch_and_extract_archive(
    url: str,
    sha256: str,
    archive_path: str,
    extract_dir: str,
    *,
    flatten: bool = False,
    keep_archive: bool = False,
) -> None:
    """Extracts via a private tmp dir and an atomic replace."""
    os.makedirs(os.path.dirname(archive_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(extract_dir) or ".", exist_ok=True)

    download(url, archive_path)
    try:
        verify_sha256(archive_path, sha256)
    except FetchError:
        os.remove(archive_path)  # don't let a retry re-verify the same corrupt bytes
        raise

    tmp_dir = f"{extract_dir}.{uuid.uuid4().hex}.tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        with tarfile.open(archive_path) as tar:
            _extract(tar, tmp_dir, flatten=flatten)

        root = tmp_dir
        if not flatten:
            entries = os.listdir(tmp_dir)
            if len(entries) == 1 and os.path.isdir(os.path.join(tmp_dir, entries[0])):
                root = os.path.join(tmp_dir, entries[0])

        with contextlib.suppress(OSError):  # a concurrent extractor already won
            os.replace(root, extract_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not keep_archive:
        os.remove(archive_path)
