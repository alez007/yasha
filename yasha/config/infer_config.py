from pydantic import BaseModel, Field

class VllmEngineConfig(BaseModel):
    model: str
    tensor_parallel_size: int = 1
    max_model_len: int = 2000
    dtype: str = "bfloat16"
    tokenizer: str = Field(default_factory=lambda data: data['model'])
    trust_remote_code: bool = True
    gpu_memory_utilization: float = 0.9
    distributed_executor_backend: str = "mp"
    

class YashaModelConfig(BaseModel):
    name: str
    vllm_engine_kwargs: VllmEngineConfig