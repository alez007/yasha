from typing import Literal, Union, cast
import os
from vllm.config.parallel import DistributedExecutorBackend
from vllm.engine.protocol import EngineClient
from vllm.config.model import ModelConfig, ModelDType
from transformers import AutoTokenizer
import torch
from vllm import AsyncEngineArgs, SamplingParams
import logging
from collections.abc import AsyncGenerator
import base64

from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from yasha.infer.infer_config import SpeechResponse, SpeechRequest, RawSpeechResponse, VllmEngineConfig, YashaModelConfig
from vllm.entrypoints.openai.protocol import ErrorInfo, ErrorResponse
import wave
import io
import numpy as np
from yasha.plugins.base_plugin import BasePlugin

logger = logging.getLogger("ray")

class ModelPlugin(BasePlugin):
    def __init__(self, model_config: YashaModelConfig):
        self.model_config = model_config

        config_engine_kwargs = model_config.vllm_engine_kwargs.model_dump() if model_config.vllm_engine_kwargs is not None else {}
        config_engine_kwargs['model'] = model_config.model
        vllm_engine_kwargs: VllmEngineConfig = VllmEngineConfig(**config_engine_kwargs)
        logger.info("initialising vllm engine with args: %s", vllm_engine_kwargs.model_dump())

        engine_args = AsyncEngineArgs(
            model=vllm_engine_kwargs.model,
            tensor_parallel_size=vllm_engine_kwargs.tensor_parallel_size,
            max_model_len=vllm_engine_kwargs.max_model_len,
            dtype=cast(ModelDType, vllm_engine_kwargs.dtype),
            tokenizer=vllm_engine_kwargs.tokenizer,
            trust_remote_code=vllm_engine_kwargs.trust_remote_code,
            gpu_memory_utilization=vllm_engine_kwargs.gpu_memory_utilization,
            distributed_executor_backend=cast(DistributedExecutorBackend, vllm_engine_kwargs.distributed_executor_backend),
            enable_log_requests=vllm_engine_kwargs.enable_log_requests if vllm_engine_kwargs.enable_log_requests is not None else False,
        )
        # engine_args.engine_use_ray = True

        usage_context = UsageContext.OPENAI_API_SERVER
        engine_config = engine_args.create_engine_config(usage_context=usage_context)

        self.engine = AsyncLLM.from_vllm_config(
            vllm_config=engine_config,
            usage_context=usage_context,
            enable_log_requests=engine_args.enable_log_requests,
            disable_log_stats=engine_args.disable_log_stats,
        )
    
    async def start(self):
        vllm_config = await self.engine.get_vllm_config()

        from orpheus_tts.decoder import tokens_decoder

        self.tokens_decoder = tokens_decoder
        tokenizer_path = vllm_config.model_config.tokenizer
        self.tokenizer = self._load_tokenizer(tokenizer_path)

    def _load_tokenizer(self, tokenizer_path):
        """Load tokenizer from local path or HuggingFace hub"""
        try:
            # Check if tokenizer_path is a local directory
            if os.path.isdir(tokenizer_path):
                return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            else:
                return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print(f"Falling back to default tokenizer")
            return AutoTokenizer.from_pretrained("gpt2")

    def _format_prompt(self, prompt: str, voice: str):
        # adapted_prompt = f"{voice}: {prompt}"
        # prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
        # start_token = torch.tensor([[ 128259]], dtype=torch.int64)
        # end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        # all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        # prompt_string = self.tokenizer.decode(all_input_ids[0])
        # return prompt_string

        formatted_prompt = f"{voice}: {prompt}"
    
        # Add special token markers for the Orpheus-FASTAPI
        special_start = "<|audio|>"  # Using the additional_special_token from config
        special_end = "<|eot_id|>"   # Using the eos_token from config
        
        return f"{special_start}{formatted_prompt}{special_end}"

    async def generate(self, input: str, voice: str, request_id: str, stream_format: Literal["sse", "audio"]) -> Union[RawSpeechResponse, AsyncGenerator[str, None],
        ErrorResponse]:
        logger.info("started generation: %s with voice: %s", input, voice)
        if stream_format=="sse":
            return self.generate_sse(input, voice, request_id)
        else:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                total_frames = 0
                async for audio_bytes in self.generate_audio_bytes_async(input, voice, request_id):
                    logger.info("got some audio bytes for wav: %s", audio_bytes)   # PCM16 bytes from SNAC
                    frame_count = len(audio_bytes) // (wf.getsampwidth() * wf.getnchannels())
                    total_frames += frame_count
                    wf.writeframes(audio_bytes)
                
                duration = total_frames / wf.getframerate()
                logger.info("audio has total duration of %s seconds", duration)
            
            return RawSpeechResponse(audio=buf.getvalue(), media_type="audio/wav")

        
    async def generate_sse(self, input: str, voice: str, request_id: str) -> AsyncGenerator[str, None]:
        # tokens_decoder is YOUR async decoder from the snippet
        async for audio_bytes in self.generate_audio_bytes_async(input, voice, request_id):
            logger.info("got some audio bytes: %s", audio_bytes)   # PCM16 bytes from SNAC
            encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
            event_data = SpeechResponse(
                audio=encoded_audio,
                type="speech.audio.delta"
            )
            yield f"data: {event_data.model_dump_json()}\n\n"
        
        completion_event=SpeechResponse(audio=None, type="speech.audio.done")
        yield f"data: {completion_event.model_dump_json()}\n\n"

    async def generate_audio_bytes_async(self, input: str, voice: str, request_id: str) -> AsyncGenerator[bytes, None]:
        async for audio_bytes in self.tokens_decoder(self.generate_tokens_async(input, voice, request_id)):
            yield audio_bytes

    async def generate_tokens_async(self, input: str, voice: str, request_id: str):
        prompt_string = self._format_prompt(input, voice)
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.8,
            max_tokens=1200,
            stop_token_ids=[49158],
            repetition_penalty=1.3,
        )

        async for result in self.engine.generate(
            prompt=prompt_string,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            chunk = result.outputs[0].text  # cumulative
            if chunk:
                yield chunk

    
    
