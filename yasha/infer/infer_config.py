from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator
from enum import Enum

from vllm.entrypoints.openai.protocol import OpenAIBaseModel

class ModelUsecase(str, Enum):
    generate = 'generate'
    embed = 'embed'
    transcription = 'transcription'
    translation = 'translation'
    tts = 'tts'

class VllmEngineConfig(BaseModel):
    model: str = ""
    tensor_parallel_size: int = 1
    max_model_len: int|None = None
    dtype: str = "bfloat16"
    tokenizer: str|None = None
    trust_remote_code: bool = False
    gpu_memory_utilization: float = 0.9
    distributed_executor_backend: str|None = None
    task: str = "auto"
    model_impl: str|None = None
    enable_log_requests: bool|None = False
    kv_cache_dtype: str|None = None
    quantization: str|None = None

class TransformersConfig(BaseModel):
    device: str = "cpu"

class PluginConfig(BaseModel):
    pass
    
class YashaModelConfig(BaseModel):
    name: str
    model: str|None = None
    usecase: ModelUsecase
    plugin: str|None = None
    use_vllm: bool = True
    vllm_engine_kwargs: VllmEngineConfig|None = None
    transformers_config: TransformersConfig|None = None
    plugin_config: PluginConfig|None = None

    @model_validator(mode='after')
    def check_model_or_plugin(self):
        if self.model is None and self.plugin is None:
            raise ValueError('model and plugin fields cannot be both empty')
        return self

class YashaConfig(BaseModel):
    models: list[YashaModelConfig]

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
    
    