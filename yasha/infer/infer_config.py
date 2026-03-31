import asyncio
import ray
from typing import Any, Literal
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from pydantic import BaseModel, Field, model_validator
from enum import Enum

from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel
from fastapi import Request
from starlette.datastructures import Headers, State


class ModelUsecase(str, Enum):
    generate = 'generate'
    embed = 'embed'
    transcription = 'transcription'
    translation = 'translation'
    tts = 'tts'


class ModelLoader(str, Enum):
    vllm = 'vllm'
    transformers = 'transformers'
    custom = 'custom'


class VllmEngineConfig(BaseModel):
    model: str = ""
    tensor_parallel_size: int = 1
    max_model_len: int|None = None
    dtype: str = "auto"
    tokenizer: str|None = None
    trust_remote_code: bool = False
    gpu_memory_utilization: float = 0.9  # overridden by num_gpus when not explicitly set in config
    distributed_executor_backend: str|None = None
    task: str = "auto"
    model_impl: str|None = None
    enable_log_requests: bool|None = False
    kv_cache_dtype: str|None = None
    quantization: str|None = None
    enable_auto_tool_choice: bool|None = None
    tool_call_parser: str|None = None
    chat_template_content_format: ChatTemplateContentFormatOption = "auto"
    enforce_eager: bool|None = None


class TransformersConfig(BaseModel):
    device: str = "cpu"


class YashaModelConfig(BaseModel):
    name: str
    model: str|None = None
    usecase: ModelUsecase
    loader: ModelLoader = ModelLoader.vllm
    plugin: str|None = None              # only meaningful for loader='custom', silently ignored otherwise
    num_gpus: float = 0
    num_cpus: float = 0.1
    use_gpu: int|str|None = None
    vllm_engine_kwargs: VllmEngineConfig = Field(default_factory=VllmEngineConfig)
    transformers_config: TransformersConfig|None = None
    plugin_config: dict[str, Any]|None = None  # plugin devs parse this themselves

    @model_validator(mode='after')
    def check_model_or_plugin(self):
        if self.model is None and self.plugin is None:
            raise ValueError('model and plugin fields cannot be both empty')
        if self.loader in (ModelLoader.vllm, ModelLoader.transformers) and self.model is None:
            raise ValueError(f"loader='{self.loader}' requires model to be set")
        return self

    @model_validator(mode='after')
    def check_custom_requires_plugin(self):
        if self.loader == ModelLoader.custom and self.plugin is None:
            raise ValueError("loader='custom' requires plugin to be set")
        return self

    @model_validator(mode='after')
    def check_use_gpu_int_incompatible_with_tp(self):
        if isinstance(self.use_gpu, int):
            tp = self.vllm_engine_kwargs.tensor_parallel_size if self.vllm_engine_kwargs else 1
            if tp > 1:
                raise ValueError(
                    "use_gpu: int pins to a single GPU via CUDA_VISIBLE_DEVICES — "
                    "incompatible with tensor_parallel_size > 1. "
                    "Use use_gpu: str (Ray custom resource) or omit use_gpu."
                )
        return self


class YashaConfig(BaseModel):
    models: list[YashaModelConfig]


@ray.remote(num_cpus=0)
class DisconnectEvent:
    """Ray actor that holds a disconnect flag — shareable across process boundaries."""
    def __init__(self):
        self._set = False

    def set(self):
        self._set = True

    def is_set(self) -> bool:
        return self._set


class RequestWatcher:
    """Watches a FastAPI Request for client disconnect and signals via a Ray actor event."""
    def __init__(self, raw_request: Request):
        self._request = raw_request
        self._event = DisconnectEvent.remote()
        asyncio.create_task(self._watch())

    async def _watch(self):
        while True:
            if await self._request.is_disconnected():
                await self._event.set.remote()  # type: ignore[attr-defined]
                break
            await asyncio.sleep(0.1)

    @property
    def event(self):
        return self._event


class DisconnectProxy:
    """
    Stands in for a FastAPI Request inside model deployment actors.

    The real FastAPI Request cannot cross Ray process boundaries — it holds a live
    TCP socket and ASGI callables that are not serializable. Instead, the gateway
    extracts the serializable parts (headers as a plain dict, disconnect signal via
    DisconnectEvent Ray actor) and passes those to the model deployment. DisconnectProxy
    reconstructs them into the interface that vllm expects from a raw_request:

      - raw_request.headers.get(...)     → Starlette Headers built from the dict
      - await raw_request.is_disconnected() → polls the DisconnectEvent Ray actor

    Any additional attributes vllm reads from raw_request in future should be added here.
    """
    def __init__(self, event, headers: dict):
        self._event = event
        self.headers = Headers(headers=headers)
        self.state = State()  # vllm writes per-request state here; initialized empty, lives in the actor process

    async def is_disconnected(self) -> bool:
        return await self._event.is_set.remote()


class SpeechRequest(OpenAIBaseModel):
    input: str = Field(..., description="The text to generate audio for")
    model: str = Field(
        ...,
        description="The model to use for generation.",
    )
    voice: str = Field(
        ...,
        description="The voice to use for generation.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. Select a value from 0.25 to 4.0.",
    )
    stream_format: Literal["sse", "audio"] = Field(
        default="audio",
        description="The stream format to return the audio in.",
    )


class SpeechResponse(OpenAIBaseModel):
    audio: str|None = Field(default=None, description="The generated audio data encoded in base 64")
    type: Literal["speech.audio.delta", "speech.audio.done"] = Field(
        ...,
        description="Type of audio chunk",
    )


class RawSpeechResponse(BaseModel):
    audio: bytes = Field(..., description="full audio file bytes")
    media_type: Literal["audio/wav"] = Field(default="audio/wav", description="audio bytes media type")
