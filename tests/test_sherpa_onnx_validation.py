"""validate_bundle() against synthetic bundle trees — the failure modes a real
tarball extraction or a hand-placed local directory can hit."""

import pytest

from modelship.infer.sherpa_onnx.bundle import validate_bundle
from modelship.infer.sherpa_onnx.registry import SherpaOnnxRegistryEntry


def _entry(**overrides) -> SherpaOnnxRegistryEntry:
    defaults = dict(
        tarball_url="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/fake.tar.bz2",
        sha256="0" * 64,
        family="kokoro",
        usecase="tts",
        files={"model": "model.onnx", "tokens": "tokens.txt", "voices": "voices.bin"},
        dirs={"data_dir": "espeak-ng-data"},
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


def test_valid_bundle_passes(tmp_path):
    _write_valid_bundle(tmp_path)
    validate_bundle(str(tmp_path), _entry())  # no raise


def test_missing_file(tmp_path):
    _write_valid_bundle(tmp_path)
    (tmp_path / "tokens.txt").unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_bundle(str(tmp_path), _entry())


def test_missing_dir(tmp_path):
    _write_valid_bundle(tmp_path)
    (tmp_path / "espeak-ng-data" / "a").unlink()
    (tmp_path / "espeak-ng-data").rmdir()
    with pytest.raises(ValueError, match="espeak-ng-data"):
        validate_bundle(str(tmp_path), _entry())


def test_missing_lexicon_file(tmp_path):
    _write_valid_bundle(tmp_path)
    entry = _entry(lexicon=("lexicon-us-en.txt",))
    with pytest.raises(ValueError, match="lexicon"):
        validate_bundle(str(tmp_path), entry)


def test_missing_bundle_dir(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        validate_bundle(str(tmp_path / "does-not-exist"), _entry())
