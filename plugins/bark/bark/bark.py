"""
Bark TTS plugin for Yasha.

Voice presets follow the pattern: v2/<lang>_speaker_<0-9>

    English:  v2/en_speaker_0  … v2/en_speaker_9
    Chinese:  v2/zh_speaker_0  … v2/zh_speaker_9
    French:   v2/fr_speaker_0  … v2/fr_speaker_9
    German:   v2/de_speaker_0  … v2/de_speaker_9
    Spanish:  v2/es_speaker_0  … v2/es_speaker_9
    Hindi:    v2/hi_speaker_0  … v2/hi_speaker_9
    Italian:  v2/it_speaker_0  … v2/it_speaker_9
    Japanese: v2/ja_speaker_0  … v2/ja_speaker_9
    Korean:   v2/ko_speaker_0  … v2/ko_speaker_9
    Polish:   v2/pl_speaker_0  … v2/pl_speaker_9
    Portuguese: v2/pt_speaker_0 … v2/pt_speaker_9
    Russian:  v2/ru_speaker_0  … v2/ru_speaker_9
    Turkish:  v2/tr_speaker_0  … v2/tr_speaker_9

Example request:

    curl http://localhost:8000/v1/audio/speech \\
      -H "Content-Type: application/json" \\
      -d '{"model": "bark", "input": "Hello world", "voice": "v2/en_speaker_6", "response_format": "wav"}' \\
      --output speech.wav
"""

import io
from collections.abc import AsyncGenerator
from typing import Literal

import torch
from scipy.io.wavfile import write as write_wav
from transformers import BarkModel, BarkProcessor

from yasha.infer.infer_config import YashaModelConfig
from yasha.logging import get_logger
from yasha.openai.protocol import ErrorResponse, RawSpeechResponse, create_error_response
from yasha.plugins.base_plugin import BasePlugin

logger = get_logger("plugin.bark")


class ModelPlugin(BasePlugin):
    def __init__(self, model_config: YashaModelConfig):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = BarkModel.from_pretrained(pretrained_model_name_or_path=model_config.model).to(self.device)  # type: ignore[arg-type]
        self.processor = BarkProcessor.from_pretrained(model_config.model)

    def __del__(self):
        try:
            if model := getattr(self, "model", None):
                del model
                self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    async def start(self):
        pass

    async def generate(
        self, input: str, voice: str, request_id: str, stream_format: Literal["sse", "audio"]
    ) -> RawSpeechResponse | AsyncGenerator[str, None] | ErrorResponse:
        logger.info("started generation: %s with voice: %s to device %s", input, voice, self.device)

        if stream_format == "sse":
            return create_error_response("sse stream format not supported")

        inputs = self.processor(input, voice_preset=voice).to(device=self.device)
        sample_rate = getattr(self.model.generation_config, "sample_rate", 24000)  # type: ignore[union-attr]
        speech_output = self.model.generate(**inputs).cpu().numpy().squeeze()  # type: ignore[call-arg]

        buf = io.BytesIO()
        write_wav(buf, rate=sample_rate, data=speech_output)
        return RawSpeechResponse(audio=buf.getvalue(), media_type="audio/wav")
