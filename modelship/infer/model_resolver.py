import fnmatch
import os
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download, model_info, snapshot_download
from tqdm.asyncio import tqdm_asyncio

from modelship.logging import get_logger
from modelship.utils.model_ref import ResolvedSource, parse_model_ref

__all__ = [
    "ModelDownloadError",
    "PinnedSource",
    "ResolvedSource",
    "check_model_source",
    "download_model_source",
    "parse_model_ref",
]

logger = get_logger("startup")

_MIB = 1024 * 1024
_HEARTBEAT_INTERVAL_SECONDS = 15

# Read by every worker thread hf_hub_download/snapshot_download spawns, so
# one model's files all report into a single aggregate.
_active_download: "_ModelDownloadProgress | None" = None


class _ModelDownloadProgress:
    """Aggregates byte progress across a model's files: a start line, a
    heartbeat every `_HEARTBEAT_INTERVAL_SECONDS` regardless of whether bytes
    moved (tqdm's `mininterval` only throttles `update()` calls that already
    happen; HF's 10 MiB read chunks mean those can stop firing for minutes on
    a slow link), and a completion line. Starts lazily on the first byte, so
    a fully-cached call stays silent."""

    def __init__(self, repo: str, total_bytes: int | None) -> None:
        self.repo = repo
        self.total_bytes = total_bytes
        self.downloaded_bytes = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_t = 0.0

    def add(self, n: int) -> None:
        started_thread: threading.Thread | None = None
        with self._lock:
            self.downloaded_bytes += n
            if self._thread is None:
                self._start_t = time.monotonic()
                self._thread = threading.Thread(target=self._heartbeat, daemon=True)
                started_thread = self._thread
        if started_thread is not None:
            if self.total_bytes:
                logger.info("%s: downloading (%.0f MiB total)", self.repo, self.total_bytes / _MIB)
            else:
                logger.info("%s: downloading", self.repo)
            started_thread.start()

    def _heartbeat(self) -> None:
        while not self._stop.wait(_HEARTBEAT_INTERVAL_SECONDS):
            logger.info("%s", self._describe())

    def _describe(self) -> str:
        mb, rate = self._stats()
        if self.total_bytes:
            pct = 100 * self.downloaded_bytes / self.total_bytes
            return (
                f"{self.repo}: downloading {pct:3.0f}% ({mb:.0f}/{self.total_bytes / _MIB:.0f} MiB, {rate:.1f} MiB/s)"
            )
        return f"{self.repo}: downloading {mb:.0f} MiB ({rate:.1f} MiB/s)"

    def _stats(self) -> tuple[float, float]:
        elapsed = max(time.monotonic() - self._start_t, 1e-9)
        mb = self.downloaded_bytes / _MIB
        return mb, mb / elapsed

    def finish(self, success: bool) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            if success:
                logger.info("%s: download complete (%.0f MiB, %.1f MiB/s)", self.repo, *self._stats())


class _DownloadProgressLogger(tqdm_asyncio):
    """`tqdm_class` for `hf_hub_download`/`snapshot_download`. Feeds chunks to
    the active `_ModelDownloadProgress` instead of logging itself — one
    instance exists per file, but logging is keyed by model."""

    def update(self, n: float | None = 1):
        result = super().update(n)
        if _active_download is not None and n is not None and n > 0:
            _active_download.add(int(n))
        return result

    def display(self, msg=None, pos=None):
        pass


def _matches_any_pattern(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p + "*" if p.endswith("/") else p) for p in patterns)


def _sum_sizes(files: Iterable[str], sizes_by_file: dict[str, int | None]) -> int | None:
    """Total bytes for `files`, or None if any of them is missing size metadata."""
    total = 0
    for f in files:
        size = sizes_by_file.get(f)
        if size is None:
            return None
        total += size
    return total


class ModelDownloadError(Exception):
    """Raised by `download_model_source` when a *validated* source can't be
    downloaded (network blip, transient HF error, disk full, ...). Kept
    distinct from `check_model_source`'s errors so `ModelDeployment` treats
    it as transient rather than fatal."""


def _select_patterns(repo_files: list[str], trust_remote_code: bool = False) -> list[str]:
    """Universal filter: prefer safetensors over bin/h5/onnx if present."""
    has_safetensors = any(f.endswith(".safetensors") or ".safetensors.index.json" in f for f in repo_files)

    patterns = [
        "*.json",
        "*.txt",
        "*.model",
        "tokenizer*",
        "vocab*",
        "merges*",
        "*.jinja",
        "chat_template*",
        "preprocessor_config.json",
        "generation_config.json",
        "image_processor_config.json",
        "processor_config.json",
    ]

    if trust_remote_code:
        patterns.append("*.py")
        patterns.append("**/*.py")

    if has_safetensors:
        patterns.extend(["*.safetensors", "*.safetensors.index.json", "**/*.safetensors"])
    else:
        # Fallback to bin if no safetensors
        patterns.extend(["*.bin", "*.bin.index.json", "**/*.bin"])

    return patterns


def _format_gguf_variants(repo_files: list[str]) -> str:
    """Format the GGUF files in a repo as a bullet list for error messages."""
    ggufs = sorted(f for f in repo_files if f.endswith(".gguf"))
    return "\n".join(f"  - {f}" for f in ggufs)


class PinnedSource(NamedTuple):
    """Result of `check_model_source` — enough to download the model later
    without touching the network again except for the download itself.

    `resolved_path` is set for local refs (already fully resolved). For HF
    refs it's None, and `download_filename` XOR `download_patterns` tells
    `download_model_source` whether to use `hf_hub_download` or
    `snapshot_download`; `first_shard` picks the entry-point file out of a
    multi-file snapshot. `total_bytes` sums every file that will be pulled
    (None if any is missing size metadata)."""

    resolved_path: str | None
    repo: str | None
    revision: str | None
    download_filename: str | None
    download_patterns: list[str] | None
    first_shard: str | None
    total_bytes: int | None

    @property
    def resolves_to_gguf(self) -> bool:
        """Whether this source resolves to a single `.gguf` file, without
        downloading it."""
        return self._resolves_to_extension(".gguf")

    def _resolves_to_extension(self, suffix: str) -> bool:
        if self.resolved_path is not None:
            return self.resolved_path.lower().endswith(suffix)
        filename = self.download_filename or self.first_shard
        return bool(filename and filename.lower().endswith(suffix))


def check_model_source(model_ref: str, trust_remote_code: bool = False) -> PinnedSource:
    """Driver-side: validate model_ref without fetching any weight bytes.

    - Local path: fully resolved here (existence + selector match).
    - HF repo: `repo_info` gives both the file listing (siblings, surfacing
      auth/missing-repo/selector-no-match) and the current commit SHA in one
      call, so every node downloads the same pinned revision later.
    """
    source, selector, is_local = parse_model_ref(model_ref)

    # Re-check localness in case it didn't start with / but exists (e.g. relative path)
    if not is_local and Path(source).exists():
        is_local = True

    if is_local:
        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Local path not found: {path}")

        if selector and path.is_dir():
            # If selector is provided for a local dir, try to match it
            matches = sorted(path.glob(selector))
            if not matches:
                # Try recursive if not found
                matches = sorted(path.rglob(selector))

            if not matches:
                raise FileNotFoundError(f"Selector {selector!r} matched no files in {path}")
            if len(matches) > 1:
                # Sharded weights (e.g. model-00001-of-00003.gguf): return the
                # first shard sorted alphabetically. llama.cpp auto-loads the
                # rest given the first shard's path.
                logger.info(
                    "Selector %r matched %d files in %s; returning first shard %s",
                    selector,
                    len(matches),
                    path,
                    matches[0].name,
                )
            resolved = matches[0].absolute()
            return PinnedSource(str(resolved), None, None, None, None, None, None)

        resolved = path.absolute()
        return PinnedSource(str(resolved), None, None, None, None, None, None)

    # HF Resolve. files_metadata=True surfaces each sibling's byte size, for
    # the aggregate download-progress total below.
    try:
        info = model_info(source, files_metadata=True)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch info for HF repo {source!r}: {e}") from e

    if info.siblings is None:
        raise RuntimeError(f"HF repo {source!r} returned no file listing")

    repo_files = [s.rfilename for s in info.siblings]
    sizes_by_file = {s.rfilename: s.size for s in info.siblings}
    revision = info.sha

    if selector:
        matches = sorted(fnmatch.filter(repo_files, selector))
        if not matches:
            raise FileNotFoundError(f"Selector {selector!r} matched no files in HF repo {source!r}")

        if len(matches) > 1:
            # Sharded weights (e.g. model-00001-of-00003.gguf): pull every shard
            # via snapshot_download, then return the path to the first shard so
            # loaders like llama.cpp (which want a file, not a directory) can
            # auto-load the rest.
            logger.info(
                "Selector %r matched %d files in HF repo %r; will download all shards, resolving to first %s",
                selector,
                len(matches),
                source,
                matches[0],
            )
            return PinnedSource(
                None, source, revision, None, [selector], matches[0], _sum_sizes(matches, sizes_by_file)
            )

        # Single match: download via hf_hub_download
        return PinnedSource(None, source, revision, matches[0], None, None, sizes_by_file.get(matches[0]))

    # No selector: detect a multi-variant GGUF repo and require an explicit pick.
    # This catches the common `model: org/repo-GGUF` mistake before the loader
    # silently auto-resolves to the wrong quant.
    ggufs = [f for f in repo_files if f.endswith(".gguf")]
    if len(ggufs) > 1:
        raise ValueError(
            f"HF repo {source!r} contains {len(ggufs)} GGUF variants — pick one with the `:filename` "
            f"syntax (glob supported, must match exactly one file):\n"
            f"{_format_gguf_variants(repo_files)}\n"
            f"Example: model: {source}:*Q4_K_M.gguf"
        )

    # Single GGUF in the repo: download it directly to a file path.
    # llama_server requires a file path, not a directory, so snapshot_download
    # would break it. The implicit "the only GGUF" is unambiguous.
    if len(ggufs) == 1:
        logger.info("HF repo %r has a single GGUF (%s); will resolve to its file path", source, ggufs[0])
        return PinnedSource(None, source, revision, ggufs[0], None, None, sizes_by_file.get(ggufs[0]))

    # Full snapshot with universal filter
    patterns = _select_patterns(repo_files, trust_remote_code=trust_remote_code)
    matched_files = [f for f in repo_files if _matches_any_pattern(f, patterns)]
    return PinnedSource(None, source, revision, None, patterns, None, _sum_sizes(matched_files, sizes_by_file))


def download_model_source(pinned: PinnedSource) -> str:
    """Download (or confirm already-cached) *pinned* and return its final
    absolute local path. A no-op for local refs. For HF refs,
    `hf_hub_download`/`snapshot_download` check their own cache first, so
    calling this when the files are already present is cheap — and stays
    silent, since `_ModelDownloadProgress` only starts logging once a byte
    actually arrives."""
    if pinned.resolved_path is not None:
        return pinned.resolved_path

    assert pinned.repo is not None  # PinnedSource invariant: local xor repo

    global _active_download
    progress = _ModelDownloadProgress(pinned.repo, pinned.total_bytes)
    _active_download = progress
    success = False
    try:
        if pinned.download_filename is not None:
            path = hf_hub_download(
                pinned.repo, pinned.download_filename, revision=pinned.revision, tqdm_class=_DownloadProgressLogger
            )
            success = True
            return path

        assert pinned.download_patterns is not None
        snapshot_dir = snapshot_download(
            pinned.repo,
            revision=pinned.revision,
            allow_patterns=pinned.download_patterns,
            tqdm_class=_DownloadProgressLogger,
        )
        success = True
        if pinned.first_shard is not None:
            return str(Path(snapshot_dir, pinned.first_shard).absolute())
        return snapshot_dir
    finally:
        progress.finish(success)
        _active_download = None


def resolve_model_source(model_ref: str, trust_remote_code: bool = False) -> str:
    """One-shot check + download, for callers that don't need the
    driver/actor split (e.g. the standalone benchmark entrypoint)."""
    return download_model_source(check_model_source(model_ref, trust_remote_code=trust_remote_code))
