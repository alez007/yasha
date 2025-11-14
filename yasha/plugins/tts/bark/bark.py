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
        
        sample_rate = self.model.generation_config.sample_rate

        if stream_format=="sse":
            return create_error_response("sse stream format not supported")
        else:
            buf = io.BytesIO()
            speech_output = self.model.generate(**inputs).cpu().numpy().squeeze()
            write_wav(buf, rate=sample_rate, data=speech_output)

            return RawSpeechResponse(audio=buf.getvalue(), media_type="audio/wav")

        
    

    
    
