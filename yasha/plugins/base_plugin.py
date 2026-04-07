from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Literal, Protocol

from vllm.config.model import ModelConfig
from vllm.engine.protocol import EngineClient

from yasha.infer.infer_config import YashaModelConfig
from yasha.openai.protocol import ErrorResponse, RawSpeechResponse


class BasePluginVllm(ABC):
    @abstractmethod
    def __init__(self, engine_client: EngineClient, model_config: ModelConfig):
        pass

    @abstractmethod
    async def generate(
        self, input: str, voice: str, request_id: str, stream_format: Literal["sse", "audio"]
    ) -> RawSpeechResponse | AsyncGenerator[str, None] | ErrorResponse:
        pass

    def max_context_length(self) -> int | None:
        return None


class BasePluginTransformers(ABC):
    @abstractmethod
    def __init__(self, model_name: str, device: str):
        pass

    @abstractmethod
    async def generate(
        self, input: str, voice: str, request_id: str, stream_format: Literal["sse", "audio"]
    ) -> RawSpeechResponse | AsyncGenerator[str, None] | ErrorResponse:
        pass

    def max_context_length(self) -> int | None:
        return None


class BasePlugin(ABC):
    @abstractmethod
    def __init__(self, model_config: YashaModelConfig):
        pass

    @abstractmethod
    async def start(self):
        pass

    @abstractmethod
    async def generate(
        self, input: str, voice: str, request_id: str, stream_format: Literal["sse", "audio"]
    ) -> RawSpeechResponse | AsyncGenerator[str, None] | ErrorResponse:
        pass

    def max_context_length(self) -> int | None:
        return None


class PluginProtoVllm(Protocol):
    ModelPlugin: type[BasePluginVllm]


class PluginProtoTransformers(Protocol):
    ModelPlugin: type[BasePluginTransformers]


class PluginProto(Protocol):
    ModelPlugin: type[BasePlugin]
