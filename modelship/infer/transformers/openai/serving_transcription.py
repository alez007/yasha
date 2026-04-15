import io

import librosa
import numpy as np
import soundfile as sf
from transformers import Pipeline

from modelship.infer.base_infer import MINIMAL_WAV
from modelship.infer.base_serving import OpenAIServing
from modelship.infer.infer_config import RawRequestProxy, TransformersConfig
from modelship.logging import TRACE, get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
    create_error_response,
)
from modelship.utils import base_request_id

logger = get_logger("infer.transformers.transcription")


def _decode_audio(audio_data: bytes, target_sr: int) -> tuple[dict, int]:
    """Decode audio bytes and resample to *target_sr*.

    Returns the pipeline input dict and the duration in whole seconds.
    """
    samples, source_sr = sf.read(io.BytesIO(audio_data), dtype="float32")
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    duration_seconds = int(len(samples) / source_sr)
    if source_sr != target_sr:
        samples = librosa.resample(samples, orig_sr=source_sr, target_sr=target_sr)
    return {"raw": samples, "sampling_rate": target_sr}, duration_seconds


class OpenAIServingTranscription(OpenAIServing):
    request_id_prefix = "asr"

    def __init__(self, pipeline: Pipeline, model_name: str, config: TransformersConfig):
        self.pipeline = pipeline
        self.model_name = model_name
        self.config = config

    @property
    def _target_sr(self) -> int:
        return self.pipeline.feature_extractor.sampling_rate

    async def warmup(self) -> None:
        logger.info("Warming up transcription model: %s", self.model_name)
        audio_input, _ = _decode_audio(MINIMAL_WAV, self._target_sr)
        await self.run_in_executor(self._run, audio_input, None)
        logger.info("Warmup transcription done for %s", self.model_name)

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: RawRequestProxy
    ) -> TranscriptionResponse | TranscriptionResponseVerbose | ErrorResponse:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        language = request.language if hasattr(request, "language") else None
        logger.info("transcription request %s", request_id)
        logger.log(
            TRACE, "transcription request %s: language=%s, audio_bytes=%d", request_id, language, len(audio_data)
        )

        try:
            audio_input, duration_seconds = _decode_audio(audio_data, self._target_sr)
            result = await self.run_in_executor(self._run, audio_input, language)
        except Exception:
            logger.exception("transcription inference failed for %s", request_id)
            return create_error_response("transcription inference failed")

        text = result["text"].strip()
        logger.log(TRACE, "transcription response %s: text=%r, duration=%ds", request_id, text, duration_seconds)

        from vllm.entrypoints.openai.speech_to_text.protocol import TranscriptionUsageAudio

        return TranscriptionResponse(
            text=text,
            usage=TranscriptionUsageAudio(seconds=duration_seconds),
        )

    def _run(self, audio_input: dict, language: str | None = None) -> dict:
        kwargs = {**self.config.pipeline_kwargs}
        if language:
            kwargs.setdefault("generate_kwargs", {})
            kwargs["generate_kwargs"]["language"] = language
        return self.pipeline(audio_input, **kwargs)  # type: ignore[return-value]


class OpenAIServingTranslation(OpenAIServing):
    request_id_prefix = "translate"

    def __init__(self, pipeline: Pipeline, model_name: str, config: TransformersConfig):
        self.pipeline = pipeline
        self.model_name = model_name
        self.config = config

    @property
    def _target_sr(self) -> int:
        return self.pipeline.feature_extractor.sampling_rate

    async def warmup(self) -> None:
        logger.info("Warming up translation model: %s", self.model_name)
        audio_input, _ = _decode_audio(MINIMAL_WAV, self._target_sr)
        await self.run_in_executor(self._run, audio_input)
        logger.info("Warmup translation done for %s", self.model_name)

    async def create_translation(
        self, audio_data: bytes, request: TranslationRequest, raw_request: RawRequestProxy
    ) -> TranslationResponse | TranslationResponseVerbose | ErrorResponse:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info("translation request %s", request_id)
        logger.log(TRACE, "translation request %s: audio_bytes=%d", request_id, len(audio_data))

        try:
            audio_input, _ = _decode_audio(audio_data, self._target_sr)
            result = await self.run_in_executor(self._run, audio_input)
        except Exception:
            logger.exception("translation inference failed for %s", request_id)
            return create_error_response("translation inference failed")

        text = result["text"].strip()
        logger.log(TRACE, "translation response %s: text=%r", request_id, text)

        return TranslationResponse(text=text)

    def _run(self, audio_input: dict) -> dict:
        kwargs = {**self.config.pipeline_kwargs}
        kwargs.setdefault("generate_kwargs", {})
        kwargs["generate_kwargs"]["task"] = "translate"
        return self.pipeline(audio_input, **kwargs)  # type: ignore[return-value]
