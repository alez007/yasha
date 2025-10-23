from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Literal, Union, Protocol
from vllm.engine.protocol import EngineClient
from vllm.config import ModelConfig
from yasha.infer.infer_config import SpeechResponse, SpeechRequest, RawSpeechResponse
from vllm.entrypoints.openai.protocol import ErrorInfo, ErrorResponse

class BasePlugin(ABC):
    @abstractmethod
    def __init__(self, engine_client: EngineClient, model_config: ModelConfig):
        pass

    @abstractmethod
    async def generate(self, input: str, voice: str, request_id: str, stream_format: Literal["sse", "audio"]) -> Union[RawSpeechResponse, AsyncGenerator[str, None],
        ErrorResponse]:
        pass

class PluginProto(Protocol):
    ModelPlugin: type[BasePlugin]
