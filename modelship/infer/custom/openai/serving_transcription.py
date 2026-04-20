from modelship.infer.infer_config import RawRequestProxy
from modelship.logging import TRACE, get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionUsageAudio,
    TranslationRequest,
    TranslationResponse,
    create_error_response,
)
from modelship.plugins.base_plugin import BasePlugin
from modelship.utils import base_request_id

logger = get_logger("infer.custom.transcription")


class OpenAIServingTranscription:
    request_id_prefix = "asr"

    def __init__(self, serving_engine: BasePlugin | None):
        self.serving_engine = serving_engine

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: RawRequestProxy
    ) -> TranscriptionResponse | ErrorResponse:
        if self.serving_engine is None:
            return create_error_response("transcription model is not yet accessible")

        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info("transcription request %s", request_id)
        logger.log(
            TRACE,
            "transcription request %s: language=%s, audio_bytes=%d",
            request_id,
            request.language,
            len(audio_data),
        )

        result = await self.serving_engine.create_transcription(
            audio_data=audio_data,
            language=request.language,
            prompt=request.prompt or None,
            temperature=request.temperature,
            request_id=request_id,
        )
        if isinstance(result, ErrorResponse):
            return result

        logger.log(TRACE, "transcription response %s: text=%r", request_id, result.text)
        return TranscriptionResponse(
            text=result.text,
            usage=TranscriptionUsageAudio(seconds=int(result.duration_seconds or 0)),
        )


class OpenAIServingTranslation:
    request_id_prefix = "translate"

    def __init__(self, serving_engine: BasePlugin | None):
        self.serving_engine = serving_engine

    async def create_translation(
        self, audio_data: bytes, request: TranslationRequest, raw_request: RawRequestProxy
    ) -> TranslationResponse | ErrorResponse:
        if self.serving_engine is None:
            return create_error_response("translation model is not yet accessible")

        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info("translation request %s", request_id)
        logger.log(TRACE, "translation request %s: audio_bytes=%d", request_id, len(audio_data))

        result = await self.serving_engine.create_translation(
            audio_data=audio_data,
            prompt=request.prompt or None,
            temperature=request.temperature,
            request_id=request_id,
        )
        if isinstance(result, ErrorResponse):
            return result

        logger.log(TRACE, "translation response %s: text=%r", request_id, result.text)
        return TranslationResponse(text=result.text)
