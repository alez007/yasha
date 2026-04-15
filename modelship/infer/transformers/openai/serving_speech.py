import struct

import numpy as np
from transformers import Pipeline

from modelship.infer.base_serving import OpenAIServing
from modelship.infer.infer_config import RawRequestProxy, TransformersConfig
from modelship.logging import TRACE, get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    RawSpeechResponse,
    SpeechRequest,
    create_error_response,
)
from modelship.utils import base_request_id

logger = get_logger("infer.transformers.speech")


def _audio_to_wav(audio: np.ndarray, sampling_rate: int) -> bytes:
    """Convert a float32 numpy audio array to WAV bytes."""
    if audio.ndim > 1:
        audio = audio.squeeze()

    pcm = (audio * 32767).astype(np.int16)
    data_bytes = pcm.tobytes()

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data_bytes),
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        1,  # mono
        sampling_rate,
        sampling_rate * 2,  # byte rate
        2,  # block align
        16,  # bits per sample
        b"data",
        len(data_bytes),
    )

    return header + data_bytes


class OpenAIServingSpeech(OpenAIServing):
    request_id_prefix = "tts"

    def __init__(self, pipeline: Pipeline, model_name: str, config: TransformersConfig):
        self.pipeline = pipeline
        self.model_name = model_name
        self.config = config

    async def warmup(self) -> None:
        logger.info("Warming up TTS model: %s", self.model_name)
        await self.run_in_executor(self._run, "warmup")
        logger.info("Warmup TTS done for %s", self.model_name)

    async def create_speech(
        self, request: SpeechRequest, raw_request: RawRequestProxy
    ) -> RawSpeechResponse | ErrorResponse:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info("speech request %s", request_id)
        logger.log(TRACE, "speech request %s: input=%r", request_id, request.input)

        try:
            result = await self.run_in_executor(self._run, request.input)
        except Exception:
            logger.exception("speech inference failed for %s", request_id)
            return create_error_response("speech inference failed")

        wav_bytes = _audio_to_wav(result["audio"], result["sampling_rate"])
        logger.log(
            TRACE,
            "speech response %s: audio_bytes=%d, sample_rate=%d",
            request_id,
            len(wav_bytes),
            result["sampling_rate"],
        )
        return RawSpeechResponse(audio=wav_bytes)

    def _run(self, text: str) -> dict:
        return self.pipeline(text, **self.config.pipeline_kwargs)  # type: ignore[return-value]
