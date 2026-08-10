"""Driver preflight: a bare registry name has nothing to pin (the actor fetches
the tarball itself); a local directory gets validated against the registry
entry its basename names, before any actor starts."""

import pytest

from modelship.deploy.config import resolve_all_model_sources
from modelship.infer.infer_config import ModelLoader, ModelshipConfig, ModelshipModelConfig, ModelUsecase


def _cfg(model: str) -> ModelshipModelConfig:
    return ModelshipModelConfig(
        name="tts", model=model, usecase=ModelUsecase.tts, loader=ModelLoader.sherpa_onnx, num_gpus=0
    )


def test_registry_name_skips_source_check():
    cfg = _cfg("kokoro-en-v0_19")
    resolve_all_model_sources(ModelshipConfig(models=[cfg]))
    assert cfg._pinned_source is None


def test_local_dir_is_validated_at_preflight(tmp_path):
    bundle = tmp_path / "kokoro-en-v0_19"
    bundle.mkdir()
    cfg = _cfg(str(bundle))
    with pytest.raises(ValueError, match="missing"):
        resolve_all_model_sources(ModelshipConfig(models=[cfg]))


def test_local_dir_basename_must_match_a_registry_name(tmp_path):
    # Config validation (not preflight) is what catches this — a mismatched
    # basename never resolves to a registry entry in the first place.
    bad_dir = tmp_path / "not-a-registered-model"
    bad_dir.mkdir()
    with pytest.raises(ValueError, match="not a supported registry name"):
        _cfg(str(bad_dir))
