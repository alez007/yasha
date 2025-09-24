from typing import Any
from pydantic import BaseModel, Field, model_validator
from enum import Enum

class ModelUsecase(str, Enum):
    generate = 'generate'
    embed = 'embed'
    transcription = 'transcription'
    translation = 'translation'

class VllmEngineConfig(BaseModel):
    model: str = ""
    tensor_parallel_size: int = 1
    max_model_len: int|None = None
    dtype: str = "bfloat16"
    tokenizer: str|None = None
    trust_remote_code: bool = False
    gpu_memory_utilization: float = 0.9
    distributed_executor_backend: str = "mp"
    task: str = "auto"
    model_impl: str|None = None
    enable_log_requests: bool|None = False

    
    
class YashaModelConfig(BaseModel):
    name: str
    model: str
    usecase: ModelUsecase
    use_vllm: bool = True
    vllm_engine_kwargs: VllmEngineConfig|None = None

class YashaConfig(BaseModel):
    models: list[YashaModelConfig]