import asyncio
from enum import StrEnum
from typing import Any, Literal

import ray
from fastapi import Request
from pydantic import BaseModel, Field, model_validator
from starlette.datastructures import Headers, State

ChatTemplateContentFormatOption = Literal["auto", "string", "openai"]


class ModelUsecase(StrEnum):
    generate = "generate"
    embed = "embed"
    transcription = "transcription"
    translation = "translation"
    tts = "tts"
    image = "image"


class ModelLoader(StrEnum):
    vllm = "vllm"
    transformers = "transformers"
    diffusers = "diffusers"
    custom = "custom"


class VllmEngineConfig(BaseModel):
    model: str = ""
    tensor_parallel_size: int = 1
    max_model_len: int | None = None
    dtype: str = "auto"
    tokenizer: str | None = None
    trust_remote_code: bool = False
    gpu_memory_utilization: float = 0.9  # overridden by num_gpus when not explicitly set in config
    distributed_executor_backend: str | None = None
    task: str = "auto"
    model_impl: str | None = None
    enable_log_requests: bool | None = False
    kv_cache_dtype: str | None = None
    quantization: str | None = None
    enable_auto_tool_choice: bool | None = None
    tool_call_parser: str | None = None
    chat_template_content_format: ChatTemplateContentFormatOption = "auto"
    enforce_eager: bool | None = None


class TransformersConfig(BaseModel):
    device: str = "cpu"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    pipeline_kwargs: dict[str, Any] = Field(default_factory=dict)


class DiffusersConfig(BaseModel):
    torch_dtype: str = "float16"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5


class ModelshipModelConfig(BaseModel):
    name: str
    model: str
    usecase: ModelUsecase
    loader: ModelLoader
    plugin: str | None = None  # only meaningful for loader='custom'
    num_gpus: float = 0
    num_cpus: float = 0.1
    num_replicas: int = 1
    vllm_engine_kwargs: VllmEngineConfig = Field(default_factory=VllmEngineConfig)
    transformers_config: TransformersConfig | None = None
    diffusers_config: DiffusersConfig | None = None
    plugin_config: dict[str, Any] | None = None  # plugin devs parse this themselves

    @model_validator(mode="after")
    def check_custom_requires_plugin(self):
        if self.loader == ModelLoader.custom and self.plugin is None:
            raise ValueError("loader='custom' requires plugin to be set")
        return self


class ModelshipConfig(BaseModel):
    models: list[ModelshipModelConfig]


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

    def __init__(self, raw_request: Request, model: str = "", endpoint: str = ""):
        self._request = raw_request
        self._event = DisconnectEvent.remote()
        self._model = model
        self._endpoint = endpoint
        self._task = asyncio.create_task(self._watch())

    async def _watch(self):
        from modelship.metrics import CLIENT_DISCONNECTS_TOTAL

        while True:
            if await self._request.is_disconnected():
                CLIENT_DISCONNECTS_TOTAL.inc(tags={"model": self._model, "endpoint": self._endpoint})
                await self._event.set.remote()  # type: ignore[attr-defined]
                break
            await asyncio.sleep(0.1)

    def stop(self):
        """Cancel the watch task and kill the Ray actor. Call when the request is fully handled."""
        self._task.cancel()
        ray.kill(self._event)

    @property
    def event(self):
        return self._event


class RawRequestProxy:
    """
    Stands in for a FastAPI Request inside model deployment actors.

    The real FastAPI Request cannot cross Ray process boundaries — it holds a live
    TCP socket and ASGI callables that are not serializable. Instead, the gateway
    extracts the serializable parts (headers as a plain dict, disconnect signal via
    DisconnectEvent Ray actor) and passes those to the model deployment. RawRequestProxy
    reconstructs them into the interface that vllm expects from a raw_request:

      - raw_request.headers.get(...)     → Starlette Headers built from the dict
      - await raw_request.is_disconnected() → polls the DisconnectEvent Ray actor

    Any additional attributes vllm reads from raw_request in future should be added here.
    """

    def __init__(self, event, headers: dict, request_id: str | None = None):
        self._event = event
        self.headers = Headers(headers=headers)
        self.state = State()  # vllm writes per-request state here; initialized empty, lives in the actor process
        self.request_id = request_id

    async def is_disconnected(self) -> bool:
        return await self._event.is_set.remote()
