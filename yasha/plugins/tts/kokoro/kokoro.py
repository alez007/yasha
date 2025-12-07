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

from vllm.entrypoints.openai.serving_models import create_error_response
from yasha.infer.infer_config import SpeechResponse, SpeechRequest, RawSpeechResponse, YashaModelConfig
from vllm.entrypoints.openai.protocol import ErrorInfo, ErrorResponse
import wave
import io
import numpy as np
from yasha.plugins.base_plugin import BasePlugin
from transformers import BarkModel, BarkProcessor
from yasha.infer.infer_config import SpeechResponse, SpeechRequest, RawSpeechResponse
from collections.abc import AsyncGenerator
from scipy.io.wavfile import write as write_wav
from kokoro import KPipeline, KModel

logger = logging.getLogger("ray")

class ModelPlugin(BasePlugin):
    def __init__(self, model_config: YashaModelConfig):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipeline = KPipeline(lang_code='a', repo_id=model_config.model, device=self.device)
        
    
    async def start(self):
        pass
    
    async def generate(self, input: str, voice: str, request_id: str, stream_format: Literal["sse", "audio"]) -> RawSpeechResponse | AsyncGenerator[str, None] | ErrorResponse:
        logger.info("started generation: %s with voice: %s to device %s", input, voice, self.device)

        generator = self.pipeline(input, voice=voice)

        buf = io.BytesIO()
        for i, (gs, ps, audio_bytes) in enumerate(generator):
            logger.info("got some audio bytes for wav: %s", audio_bytes)
            write_wav(buf, rate=24000, data=audio_bytes)
            

        return RawSpeechResponse(audio=buf.getvalue(), media_type="audio/wav")

        
    

    
    
