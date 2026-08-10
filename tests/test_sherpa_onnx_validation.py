"""validate_bundle() against synthetic bundle trees — the failure modes a real
tarball extraction or a hand-placed local directory can hit."""

import hashlib

import pytest

from modelship.infer.sherpa_onnx.bundle import validate_bundle
from modelship.infer.sherpa_onnx.registry import RegistryDir, RegistryFile, SherpaOnnxRegistryEntry


def _entry(**overrides) -> SherpaOnnxRegistryEntry:
    defaults = dict(
        tarball_url="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/fake.tar.bz2",
        sha256="0" * 64,
        family="kokoro",
        usecase="tts",
        files={
            "model": RegistryFile("model.onnx", 8),
            "tokens": RegistryFile("tokens.txt", 5, hashlib.sha256(b"hello").hexdigest()),
            "voices": RegistryFile("voices.bin", 4),
        },
        dirs={"data_dir": RegistryDir("espeak-ng-data", 2)},
        lexicon=(),
        voice_names=("af_bella",),
    )
    return SherpaOnnxRegistryEntry(**{**defaults, **overrides})


def _write_valid_bundle(root):
    (root / "model.onnx").write_bytes(b"x" * 8)
    (root / "tokens.txt").write_bytes(b"hello")
    (root / "voices.bin").write_bytes(b"x" * 4)
    data_dir = root / "espeak-ng-data"
    data_dir.mkdir()
    (data_dir / "a").write_bytes(b"1")
    (data_dir / "b").write_bytes(b"1")


def test_valid_bundle_passes(tmp_path):
    _write_valid_bundle(tmp_path)
    validate_bundle(str(tmp_path), _entry())  # no raise


def test_missing_file(tmp_path):
    _write_valid_bundle(tmp_path)
    (tmp_path / "tokens.txt").unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_bundle(str(tmp_path), _entry())


def test_wrong_size_lfs_pointer(tmp_path):
    _write_valid_bundle(tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"git-lfs pointer, not the real file")
    with pytest.raises(ValueError, match="bytes"):
        validate_bundle(str(tmp_path), _entry())


def test_short_dir_file_count(tmp_path):
    _write_valid_bundle(tmp_path)
    (tmp_path / "espeak-ng-data" / "b").unlink()
    with pytest.raises(ValueError, match="espeak-ng-data"):
        validate_bundle(str(tmp_path), _entry())


def test_bad_small_file_hash(tmp_path):
    _write_valid_bundle(tmp_path)
    (tmp_path / "tokens.txt").write_bytes(b"wrong")  # same size, different content
    with pytest.raises(ValueError, match="sha256"):
        validate_bundle(str(tmp_path), _entry())


def test_missing_lexicon_file(tmp_path):
    _write_valid_bundle(tmp_path)
    entry = _entry(lexicon=(RegistryFile("lexicon-us-en.txt", 3),))
    with pytest.raises(ValueError, match="lexicon"):
        validate_bundle(str(tmp_path), entry)


def test_missing_bundle_dir(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        validate_bundle(str(tmp_path / "does-not-exist"), _entry())
