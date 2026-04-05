"""
Kokoro ONNX TTS plugin for Yasha.

Selected voices (full list in voices-v1.0.bin):

    American Female: af_heart, af_bella, af_nicole, af_sarah, af_sky
    American Male:   am_adam, am_michael
    British Female:  bf_emma, bf_isabella
    British Male:    bm_george, bm_lewis

Plugin config options (via plugin_config in models.yaml):

    onnx_provider: ONNX execution provider (default: "CUDAExecutionProvider")
                   Other options: "CPUExecutionProvider", "TensorrtExecutionProvider"
    sample_rate:   Resample output to this rate in Hz (default: model native ~24000)

Example request:

    curl http://localhost:8000/v1/audio/speech \\
      -H "Content-Type: application/json" \\
      -d '{"model": "kokoro", "input": "Hello world", "voice": "af_heart", "response_format": "wav"}' \\
      --output speech.wav
"""

import base64
import io
import logging
import os
from collections.abc import AsyncGenerator
from typing import Literal

import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro
from scipy.io.wavfile import write as write_wav
from scipy.signal import resample_poly
from vllm.entrypoints.openai.engine.protocol import ErrorResponse

from yasha.infer.infer_config import RawSpeechResponse, SpeechResponse, YashaModelConfig
from yasha.plugins.base_plugin import BasePlugin
from yasha.utils import cache_dir, download

logger = logging.getLogger("ray")


class ModelPlugin(BasePlugin):
    def __init__(self, model_config: YashaModelConfig):
        logger.info("onnxruntime device: %s", ort.get_device())
        logger.info("available providers: %s", ort.get_available_providers())

        plugin_dir = f"{cache_dir()}/kokoro"
        os.makedirs(plugin_dir, exist_ok=True)

        model_path = f"{plugin_dir}/kokoro-v1.0.onnx"
        voices_path = f"{plugin_dir}/voices-v1.0.bin"
        download(
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
            model_path,
        )
        download(
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
            voices_path,
        )

        onnx_provider = (model_config.plugin_config or {}).get("onnx_provider", "CUDAExecutionProvider")
        os.environ["ONNX_PROVIDER"] = onnx_provider
        logger.info("ONNX_PROVIDER=%s", onnx_provider)
        self.kokoro = Kokoro(model_path, voices_path)
        logger.info("kokoro session providers: %s", self.kokoro.sess.get_providers())
        self.target_sample_rate: int | None = (model_config.plugin_config or {}).get("sample_rate")

    def __del__(self):
        try:
            if kokoro := getattr(self, "kokoro", None):
                del kokoro
                self.kokoro = None
        except Exception:
            pass

    async def start(self):
        pass

    def _resample(self, audio: np.ndarray, from_rate: int) -> tuple[np.ndarray, int]:
        if self.target_sample_rate is None or from_rate == self.target_sample_rate:
            return audio, from_rate
        from math import gcd

        g = gcd(self.target_sample_rate, from_rate)
        audio = resample_poly(audio, self.target_sample_rate // g, from_rate // g).astype(audio.dtype)
        return audio, self.target_sample_rate

    async def generate(
        self, input: str, voice: str, request_id: str, stream_format: Literal["sse", "audio"]
    ) -> RawSpeechResponse | AsyncGenerator[str, None] | ErrorResponse:
        logger.info("started generation: %s with voice: %s", input, voice)

        if stream_format == "sse":
            return self.generate_sse(input, voice, request_id)
        else:
            chunks = []
            sample_rate = self.target_sample_rate
            async for audio_bytes, sr in self.kokoro.create_stream(input, voice=voice, speed=1.0, lang="en-us"):
                audio_bytes, sample_rate = self._resample(audio_bytes, sr)
                chunks.append(audio_bytes)

            logger.info("got %d chunks (sample rate %s) for input: %s", len(chunks), sample_rate, input)
            audio = np.concatenate(chunks)
            if np.issubdtype(audio.dtype, np.floating):
                audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            buf = io.BytesIO()
            write_wav(buf, rate=sample_rate, data=audio)
            return RawSpeechResponse(audio=buf.getvalue(), media_type="audio/wav")

    async def generate_sse(self, input: str, voice: str, request_id: str) -> AsyncGenerator[str, None]:
        async for audio_bytes, sample_rate in self.kokoro.create_stream(input, voice=voice, speed=1.0, lang="en-us"):
            audio_bytes, sample_rate = self._resample(audio_bytes, sample_rate)
            logger.info("got some audio bytes (sample rate %s) for input: %s", sample_rate, input)
            encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
            event_data = SpeechResponse(audio=encoded_audio, type="speech.audio.delta")
            yield f"data: {event_data.model_dump_json()}\n\n"

        completion_event = SpeechResponse(audio=None, type="speech.audio.done")
        yield f"data: {completion_event.model_dump_json()}\n\n"
