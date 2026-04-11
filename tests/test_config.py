"""Tests for Modelship model configuration parsing and validation."""

import pytest
from pydantic import ValidationError

from modelship.infer.infer_config import (
    ModelLoader,
    ModelshipConfig,
    ModelshipModelConfig,
    ModelUsecase,
    VllmEngineConfig,
)


class TestModelshipModelConfig:
    def test_minimal_vllm_model(self):
        config = ModelshipModelConfig(
            name="test-llm",
            model="some-org/some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
        )
        assert config.name == "test-llm"
        assert config.loader == ModelLoader.vllm
        assert config.num_gpus == 0
        assert config.num_cpus == 0.1

    def test_custom_loader_requires_plugin(self):
        with pytest.raises(ValidationError, match="loader='custom' requires plugin"):
            ModelshipModelConfig(
                name="test-tts",
                model="some-model",
                usecase=ModelUsecase.tts,
                loader=ModelLoader.custom,
            )

    def test_custom_loader_with_plugin(self):
        config = ModelshipModelConfig(
            name="test-tts",
            model="some-model",
            usecase=ModelUsecase.tts,
            loader=ModelLoader.custom,
            plugin="kokoro",
        )
        assert config.plugin == "kokoro"

    def test_custom_loader_plugin_only(self):
        config = ModelshipModelConfig(
            name="test-tts",
            model="some-model",
            usecase=ModelUsecase.tts,
            loader=ModelLoader.custom,
            plugin="kokoro",
        )
        assert config.plugin == "kokoro"

    def test_model_required(self):
        with pytest.raises(ValidationError, match="Field required"):
            ModelshipModelConfig(
                name="test-llm",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.vllm,
            )

    def test_loader_required(self):
        with pytest.raises(ValidationError, match="Field required"):
            ModelshipModelConfig(
                name="test-llm",
                model="some-model",
                usecase=ModelUsecase.generate,
            )

    def test_gpu_allocation_fraction(self):
        config = ModelshipModelConfig(
            name="test-llm",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=0.70,
        )
        assert config.num_gpus == 0.70

    def test_all_usecases_valid(self):
        for usecase in ModelUsecase:
            config = ModelshipModelConfig(
                name=f"test-{usecase.value}",
                model="some-model",
                usecase=usecase,
                loader=ModelLoader.vllm,
            )
            assert config.usecase == usecase

    def test_all_loaders_valid(self):
        for loader in ModelLoader:
            kwargs = {"name": "test", "model": "some-model", "usecase": ModelUsecase.generate}
            if loader == ModelLoader.custom:
                kwargs["plugin"] = "test-plugin"
            config = ModelshipModelConfig(loader=loader, **kwargs)
            assert config.loader == loader


class TestVllmEngineConfig:
    def test_defaults(self):
        config = VllmEngineConfig()
        assert config.tensor_parallel_size == 1
        assert config.dtype == "auto"
        assert config.gpu_memory_utilization == 0.9
        assert config.trust_remote_code is False

    def test_custom_values(self):
        config = VllmEngineConfig(
            tensor_parallel_size=2,
            max_model_len=12288,
            enable_auto_tool_choice=True,
            tool_call_parser="llama3_json",
        )
        assert config.tensor_parallel_size == 2
        assert config.max_model_len == 12288
        assert config.enable_auto_tool_choice is True
        assert config.tool_call_parser == "llama3_json"


class TestModelshipConfig:
    def test_multi_model_config(self):
        config = ModelshipConfig(
            models=[
                ModelshipModelConfig(
                    name="llm",
                    model="some-org/some-llm",
                    usecase=ModelUsecase.generate,
                    loader=ModelLoader.vllm,
                    num_gpus=0.70,
                ),
                ModelshipModelConfig(
                    name="tts",
                    model="some-model",
                    usecase=ModelUsecase.tts,
                    loader=ModelLoader.custom,
                    plugin="kokoro",
                    num_gpus=0.05,
                ),
            ]
        )
        assert len(config.models) == 2
        assert config.models[0].name == "llm"
        assert config.models[1].name == "tts"

    def test_empty_models_list(self):
        config = ModelshipConfig(models=[])
        assert len(config.models) == 0

    def test_duplicate_names_allowed(self):
        config = ModelshipConfig(
            models=[
                ModelshipModelConfig(
                    name="kokoro",
                    model="hexgrad/Kokoro-82M",
                    usecase=ModelUsecase.tts,
                    loader=ModelLoader.custom,
                    plugin="kokoro",
                    num_gpus=0.07,
                ),
                ModelshipModelConfig(
                    name="kokoro",
                    model="hexgrad/Kokoro-82M",
                    usecase=ModelUsecase.tts,
                    loader=ModelLoader.custom,
                    plugin="kokoro",
                    num_gpus=0,
                ),
            ]
        )
        assert len(config.models) == 2
        assert config.models[0].name == config.models[1].name == "kokoro"


class TestNumReplicas:
    def test_default_num_replicas(self):
        config = ModelshipModelConfig(
            name="test",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
        )
        assert config.num_replicas == 1

    def test_custom_num_replicas(self):
        config = ModelshipModelConfig(
            name="test",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_replicas=3,
        )
        assert config.num_replicas == 3
