"""
Bark TTS plugin for Modelship.

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

from collections.abc import AsyncGenerator

import torch
from transformers import BarkModel, BarkProcessor

from modelship.infer.infer_config import ModelshipModelConfig
from modelship.logging import get_logger
from modelship.openai.protocol import ErrorResponse, RawSpeechResponse, create_error_response
from modelship.plugins.base_plugin import BasePlugin
from modelship.utils.audio import to_wav

logger = get_logger("plugin.bark")


class ModelPlugin(BasePlugin):
    def __init__(self, model_config: ModelshipModelConfig):
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

    async def create_speech(
        self,
        input: str,
        voice: str | None = None,
        speed: float | None = None,
        stream: bool = False,
        request_id: str | None = None,
    ) -> RawSpeechResponse | AsyncGenerator[tuple[bytes, int], None] | ErrorResponse:
        logger.info("started generation: %s with voice: %s to device %s", input, voice, self.device)

        if stream:
            return create_error_response("bark does not support streaming")

        inputs = self.processor(input, voice_preset=voice).to(device=self.device)
        sample_rate = getattr(self.model.generation_config, "sample_rate", 24000)  # type: ignore[union-attr]
        speech_output = self.model.generate(**inputs).cpu().numpy().squeeze()  # type: ignore[call-arg]

        return RawSpeechResponse(audio=to_wav(speech_output, sample_rate), media_type="audio/wav")
