from typing import Literal, Union
import os
from vllm.engine.protocol import EngineClient
from vllm.config.model import ModelConfig
from transformers import AutoTokenizer
import torch
from vllm import SamplingParams
import logging
from collections.abc import AsyncGenerator
import base64
from yasha.infer.infer_config import SpeechResponse, SpeechRequest, RawSpeechResponse, YashaModelConfig
from vllm.entrypoints.openai.protocol import ErrorInfo, ErrorResponse
import wave
import io
import numpy as np
from yasha.plugins.base_plugin import BasePlugin
from transformers import BarkModel, BarkProcessor
from yasha.infer.infer_config import SpeechResponse, SpeechRequest, RawSpeechResponse
from collections.abc import AsyncGenerator
import struct

logger = logging.getLogger("ray")

class ModelPlugin(BasePlugin):
    def __init__(self, model_config: YashaModelConfig):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = BarkModel.from_pretrained(pretrained_model_name_or_path=model_config.model).to(self.device)
        self.processor = BarkProcessor.from_pretrained(model_config.model)
    
    async def start(self):
        pass
    
    async def generate(self, input: str, voice: str, request_id: str, stream_format: Literal["sse", "audio"]) -> RawSpeechResponse | AsyncGenerator[str, None] | ErrorResponse:
        logger.info("started generation: %s with voice: %s to device %s", input, voice, self.device)

        inputs = self.processor(input, voice_preset=voice).to(device=self.device)
        speech_output = self.model.generate(**inputs).cpu().numpy()
        logger.info("speech_output 1: %s", speech_output)
        
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            logger.info("got some audio bytes for wav: %s", speech_output)   # PCM16 bytes from SNAC
            wf.writeframes(speech_output)
                

        return RawSpeechResponse(audio=buf.getvalue(), media_type="audio/wav")
        # if stream_format=="sse":
        #     return self.generate_sse(input, voice, request_id)
        # else:
        #     buf = io.BytesIO()
        #     with wave.open(buf, "wb") as wf:
        #         wf.setnchannels(1)
        #         wf.setsampwidth(2)
        #         wf.setframerate(24000)
        #         async for audio_bytes in self.generate_audio_bytes_async(input, voice, request_id):
        #             logger.info("got some audio bytes for wav: %s", audio_bytes)   # PCM16 bytes from SNAC
        #             wf.writeframes(audio_bytes)
            
        #     return RawSpeechResponse(audio=buf.getvalue(), media_type="audio/wav")

        
    

    
    
