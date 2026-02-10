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

from vllm.entrypoints.openai.models.serving import create_error_response
from yasha.infer.infer_config import SpeechResponse, SpeechRequest, RawSpeechResponse, YashaModelConfig
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
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
from yasha.utils import download, tmp_dir

logger = logging.getLogger("ray")


class ModelPlugin(BasePlugin):
    def __init__(self, model_config: YashaModelConfig):
        logger.info(f"onnxruntime device: {ort.get_device()}")
        logger.info(f"available providers: {ort.get_available_providers()}")

        plugin_dir = f"{tmp_dir()}/kokoro"

        if not os.path.exists(plugin_dir):
            os.mkdir(plugin_dir)
        
        model_path = f"{plugin_dir}/kokoro-v1.0.onnx"
        voices_path = f"{plugin_dir}/voices-v1.0.bin"
        download("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx", model_path)
        download("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin", voices_path)

        self.kokoro = Kokoro(model_path, voices_path)
        
    
    async def start(self):
        pass
    
    async def generate(self, input: str, voice: str, request_id: str, stream_format: Literal["sse", "audio"]) -> RawSpeechResponse | AsyncGenerator[str, None] | ErrorResponse:
        logger.info("started generation: %s with voice: %s", input, voice)

        if stream_format=="sse":
            return self.generate_sse(input, voice, request_id)
        else:
            buf = io.BytesIO()
            async for audio_bytes, sample_rate in self.kokoro.create_stream(input, voice=voice, speed=1.0, lang="en-us"):
                logger.info("got some audio bytes (sample rate %s) for input: %s", sample_rate, input)
                write_wav(buf, rate=sample_rate, data=audio_bytes)

            return RawSpeechResponse(audio=buf.getvalue(), media_type="audio/wav")
       
        
    async def generate_sse(self, input: str, voice: str, request_id: str) -> AsyncGenerator[str, None]:
        async for audio_bytes, sample_rate in self.kokoro.create_stream(input, voice=voice, speed=1.0, lang="en-us"):
            logger.info("got some audio bytes (sample rate %s) for input: %s", sample_rate, input)
            encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
            event_data = SpeechResponse(
                audio=encoded_audio,
                type="speech.audio.delta"
            )
            yield f"data: {event_data.model_dump_json()}\n\n"
            
        completion_event=SpeechResponse(audio=None, type="speech.audio.done")
        yield f"data: {completion_event.model_dump_json()}\n\n"

        
    

    
    
