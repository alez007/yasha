from typing import Dict, Optional, List
from pydantic import BaseModel, Field

class VllmEngineConfig(BaseModel):
    model: str
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    dtype: str = "bfloat16"
    tokenizer: str = Field(default_factory=lambda data: data['model'])
    trust_remote_code: bool = False
    gpu_memory_utilization: float = 0.9
    distributed_executor_backend: str = "mp"
    task: str = "auto"
    model_impl: Optional[str] = None
    enable_log_requests: Optional[bool] = False
    
class YashaModelConfig(BaseModel):
    name: str
    vllm_engine_kwargs: VllmEngineConfig

class YashaConfig(BaseModel):
    models: List[YashaModelConfig]