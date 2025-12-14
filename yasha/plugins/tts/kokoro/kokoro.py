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
from kokoro_onnx import Kokoro
import onnxruntime as ort
from yasha.utils import download

logger = logging.getLogger("ray")

class ModelPlugin(BasePlugin):
    def __init__(self, model_config: YashaModelConfig):
        providers = ort.get_available_providers()
        logger.info("available providers: %s", providers)
        
        download("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx", "/yasha/plugins/tts/kokoro/kokoro-v1.0.onnx")
        download("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin", "/yasha/plugins/tts/kokoro/voices-v1.0.bin")

        self.kokoro = Kokoro("/yasha/plugins/tts/kokoro/kokoro-v1.0.onnx", "/yasha/plugins/tts/kokoro/voices-v1.0.bin")
        
    
    async def start(self):
        pass
    
    async def generate(self, input: str, voice: str, request_id: str, stream_format: Literal["sse", "audio"]) -> RawSpeechResponse | AsyncGenerator[str, None] | ErrorResponse:
        logger.info("started generation: %s with voice: %s", input, voice)

        buf = io.BytesIO()
        async for audio_bytes, sample_rate in self.kokoro.create_stream(input, voice=voice, speed=1.0, lang="en-us"):
            logger.info("got some audio bytes for wav: %s", audio_bytes)
            write_wav(buf, rate=sample_rate, data=audio_bytes)
            

        return RawSpeechResponse(audio=buf.getvalue(), media_type="audio/wav")

        
    

    
    
