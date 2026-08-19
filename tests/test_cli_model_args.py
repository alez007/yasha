"""The `--model` flags must produce a models.yaml entry indistinguishable from the
hand-written one."""

import subprocess
import sys
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from modelship.deploy.config import resolve_input_models, validate_models
from modelship.utils.cli import MODEL_ARG_KEYS, infer_model_name, model_from_args, parse_args


def _args(*argv):
    return parse_args(list(argv))


def _raw(*argv):
    return model_from_args(_args(*argv))


class TestParityWithYaml:
    def test_fingerprint_and_fields_set_match_the_yaml_entry(self):
        yaml_entry = {
            "name": "qwen",
            "model": "lmstudio-community/Qwen3-8B-GGUF:*Q4_K_M.gguf",
            "usecase": "generate",
            "loader": "llama_server",
            "num_cpus": 3.0,
        }
        cli_entry = _raw(
            "--model",
            "lmstudio-community/Qwen3-8B-GGUF:*Q4_K_M.gguf",
            "--name",
            "qwen",
            "--usecase",
            "generate",
            "--loader",
            "llama_server",
            "--num-cpus",
            "3",
        )
        assert cli_entry == yaml_entry

        from_yaml = validate_models([yaml_entry]).models[0]
        from_cli = validate_models([cli_entry]).models[0]
        assert from_cli.fingerprint() == from_yaml.fingerprint()
        assert from_cli.model_fields_set == from_yaml.model_fields_set

    def test_unset_flags_are_omitted_not_defaulted(self):
        raw = _raw("--model", "Qwen/Qwen3-8B", "--usecase", "generate", "--loader", "vllm")
        assert raw == {"name": "qwen3-8b", "model": "Qwen/Qwen3-8B", "usecase": "generate", "loader": "vllm"}
        assert validate_models([raw]).models[0].model_fields_set == {"name", "model", "usecase", "loader"}

    def test_num_replicas_left_unset_does_not_block_autoscaling(self):
        """The check rejects num_replicas alongside autoscaling_config via
        model_fields_set — a materialized CLI default would trip it."""
        raw = _raw("--model", "Qwen/Qwen3-8B", "--usecase", "generate", "--loader", "vllm")
        merged = {**raw, "autoscaling_config": {"min_replicas": 1, "max_replicas": 3}}
        validate_models([merged])  # no raise

    def test_image_loader_still_defaults_usecase(self):
        raw = _raw("--model", "stabilityai/sdxl-turbo", "--loader", "diffusers")
        assert "usecase" not in raw
        assert validate_models([raw]).models[0].usecase.value == "image"

    def test_schema_rejects_missing_loader(self):
        with pytest.raises(ValidationError, match="loader"):
            validate_models([_raw("--model", "Qwen/Qwen3-8B", "--usecase", "generate")])

    def test_model_arg_keys_covers_every_flag_the_builder_emits(self):
        raw = _raw(
            "--model", "Qwen/Qwen3-8B",
            "--name", "n",
            "--usecase", "generate",
            "--loader", "vllm",
            "--num-gpus", "0.5",
            "--num-cpus", "2",
            "--num-replicas", "2",
            "--max-ongoing-requests", "8",
        )  # fmt: skip
        assert set(raw) == set(MODEL_ARG_KEYS)


class TestInferModelName:
    @pytest.mark.parametrize(
        ("ref", "expected"),
        [
            ("lmstudio-community/Qwen3-8B-GGUF:*Q4_K_M.gguf", "qwen3-8b"),
            ("bartowski/Llama-3.3-70B-Instruct-GGUF:*Q8_0*-of-*.gguf", "llama-3.3-70b-instruct"),
            ("Qwen/Qwen3-8B", "qwen3-8b"),
            ("nomic-ai/nomic-embed-text-v1.5-GGUF:nomic-embed-text-v1.5.Q4_K_M.gguf", "nomic-embed-text-v1.5"),
            ("/models/qwen3-8b-instruct.Q4_K_M.gguf", "qwen3-8b-instruct"),
            ("/models/Qwen2.5-0.5B-Instruct.IQ4_XS.gguf", "qwen2.5-0.5b-instruct"),
            ("/models/Qwen3-8B/", "qwen3-8b"),
            ("./weights/model.f16.gguf", "model"),
            ("kokoro-en-v0_19", "kokoro-en-v0_19"),
            ("base.en", "base.en"),
        ],
    )
    def test_inference(self, ref, expected):
        assert infer_model_name(ref) == expected

    def test_explicit_name_wins(self):
        assert _raw("--model", "Qwen/Qwen3-8B", "--name", "custom")["name"] == "custom"

    def test_uninferrable_ref_raises(self):
        with pytest.raises(ValueError, match="pass --name"):
            infer_model_name("///")

    def test_same_ref_infers_the_same_name(self):
        """Re-running a deploy must be idempotent: the name is the additive
        merge's replace-by-name key."""
        ref = "lmstudio-community/Qwen3-8B-GGUF:*Q4_K_M.gguf"
        assert infer_model_name(ref) == infer_model_name(ref)


class TestResolveInputModels:
    def test_model_flag_wins_over_the_default_config_file(self, tmp_path):
        default = tmp_path / "models.yaml"
        default.write_text("models:\n  - name: from-file\n    model: x.gguf\n    loader: llama_server\n")
        with patch("modelship.deploy.config.default_config_path", return_value=default):
            raw = resolve_input_models(_args("--model", "Qwen/Qwen3-8B", "--loader", "vllm", "--usecase", "generate"))
        assert raw is not None
        assert [m["name"] for m in raw] == ["qwen3-8b"]

    def test_config_file_is_read_when_no_model_flag(self, tmp_path):
        config = tmp_path / "models.yaml"
        config.write_text("models:\n  - name: m\n    model: x.gguf\n    loader: llama_server\n    usecase: generate\n")
        raw = resolve_input_models(_args("--config", str(config)))
        assert raw == [{"name": "m", "model": "x.gguf", "loader": "llama_server", "usecase": "generate"}]

    def test_neither_returns_none(self, tmp_path):
        with patch("modelship.deploy.config.default_config_path", return_value=tmp_path / "nope.yaml"):
            assert resolve_input_models(_args()) is None

    def test_both_is_rejected_at_parse_time(self, tmp_path, capsys):
        config = tmp_path / "models.yaml"
        config.write_text("models: []\n")
        with pytest.raises(SystemExit) as exc:
            _args("--model", "Qwen/Qwen3-8B", "--config", str(config))
        assert exc.value.code == 2
        assert "mutually exclusive" in capsys.readouterr().err

    def test_both_is_rejected_for_a_hand_built_namespace(self, tmp_path):
        args = _args("--config", str(tmp_path / "models.yaml"))
        args.model = "Qwen/Qwen3-8B"
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_input_models(args)


class TestFailsBeforeRay:
    def test_bad_model_flag_exits_without_importing_ray(self):
        code = (
            "import sys; from modelship.launcher import _cmd_deploy\n"
            "try:\n"
            "    _cmd_deploy(['--model', 'Qwen/Qwen3-8B'])\n"
            "except SystemExit as e:\n"
            "    print('exit', e.code, 'ray' in sys.modules)\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "exit 1 False"


class TestIntegrationCliRouting:
    """conftest sends a lone CLI-expressible model through --model. Pin the entries
    that take that route: a nested block added to one fails here, not silently."""

    def test_single_model_call_sites_stay_cli_expressible(self):
        from tests.conftest import MODEL_CONFIGS, cli_expressible

        routed = {"chat-llama-server-plain", "chat-llama-server-gpu", "embed-model-llama-server"}
        assert {name for name in routed if cli_expressible(MODEL_CONFIGS[name])} == routed

    def test_nested_blocks_are_not_cli_expressible(self):
        from tests.conftest import MODEL_CONFIGS, cli_expressible

        assert not cli_expressible(MODEL_CONFIGS["chat-capable"])
        assert not cli_expressible(MODEL_CONFIGS["autoscale-llama"])

    def test_flags_round_trip_back_to_the_config(self):
        """conftest builds argv from a MODEL_CONFIGS entry; parsing it back must
        reproduce that entry."""
        from tests.conftest import MODEL_CONFIGS, _model_flags

        config = MODEL_CONFIGS["chat-llama-server-gpu"]
        assert model_from_args(_args(*_model_flags(config))) == config
