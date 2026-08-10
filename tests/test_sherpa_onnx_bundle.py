"""resolve_bundle_dir()'s cache-hit and re-fetch-on-invalid-cache paths."""

import hashlib
import os
import tarfile
from unittest.mock import patch

from modelship.infer.sherpa_onnx import bundle
from modelship.infer.sherpa_onnx.registry import REGISTRY

_ENTRY = REGISTRY["kokoro-en-v0_19"]


def _make_valid_archive(tmp_path) -> tuple[str, str]:
    src_dir = tmp_path / "src" / "kokoro-en-v0_19"
    src_dir.mkdir(parents=True)
    (src_dir / "model.onnx").write_bytes(b"m")
    (src_dir / "tokens.txt").write_bytes(b"t")
    (src_dir / "voices.bin").write_bytes(b"v")
    data_dir = src_dir / "espeak-ng-data"
    data_dir.mkdir()
    (data_dir / "a").write_bytes(b"1")

    archive = tmp_path / "src_archive.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        tar.add(src_dir, arcname="kokoro-en-v0_19")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return str(archive), digest


def _fake_download(src_archive: str):
    def fake(url, dest):
        with open(src_archive, "rb") as f, open(dest, "wb") as out:
            out.write(f.read())

    return fake


def test_fetches_when_no_cached_bundle(tmp_path, monkeypatch):
    src_archive, digest = _make_valid_archive(tmp_path)
    monkeypatch.setattr(bundle, "cache_dir", lambda: str(tmp_path / "cache"))
    entry = _ENTRY._replace(sha256=digest)

    with (
        patch.dict(bundle.REGISTRY, {"kokoro-en-v0_19": entry}),
        patch("modelship.utils.download", side_effect=_fake_download(src_archive)),
    ):
        bundle_dir, resolved_entry = bundle.resolve_bundle_dir("kokoro-en-v0_19")

    assert resolved_entry is entry
    assert os.path.isfile(os.path.join(bundle_dir, "model.onnx"))


def test_stale_cached_bundle_is_cleared_and_refetched(tmp_path, monkeypatch):
    src_archive, digest = _make_valid_archive(tmp_path)
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(bundle, "cache_dir", lambda: str(cache_root))
    entry = _ENTRY._replace(sha256=digest)

    stale_dir = cache_root / "sherpa_onnx" / "kokoro-en-v0_19"
    stale_dir.mkdir(parents=True)
    (stale_dir / "leftover_junk.txt").write_bytes(b"junk")

    with (
        patch.dict(bundle.REGISTRY, {"kokoro-en-v0_19": entry}),
        patch("modelship.utils.download", side_effect=_fake_download(src_archive)),
    ):
        bundle_dir, _resolved_entry = bundle.resolve_bundle_dir("kokoro-en-v0_19")

    assert os.path.isfile(os.path.join(bundle_dir, "model.onnx"))
    assert not os.path.exists(os.path.join(bundle_dir, "leftover_junk.txt"))
