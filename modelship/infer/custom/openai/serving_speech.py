import base64
from collections.abc import AsyncGenerator

from modelship.infer.infer_config import RawRequestProxy
from modelship.logging import get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    RawSpeechResponse,
    SpeechRequest,
    SpeechResponse,
    create_error_response,
)
from modelship.plugins.base_plugin import BasePlugin
from modelship.utils import base_request_id

logger = get_logger("infer.custom.speech")


class OpenAIServingSpeech:
    request_id_prefix = "tts"

    """Handles speech requests"""

    def __init__(self, serving_engine: BasePlugin | None):
        self.serving_engine = serving_engine

    async def create_speech(
        self, request: SpeechRequest, raw_request: RawRequestProxy
    ) -> RawSpeechResponse | AsyncGenerator[str, None] | ErrorResponse:
        if self.serving_engine is None:
            return create_error_response("tts model is not yet accessible")

        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        stream_sse = "stream_format" in request.model_fields_set and request.stream_format == "sse"

        result = await self.serving_engine.create_speech(
            input=request.input,
            voice=request.voice,
            speed=request.speed,
            stream=stream_sse,
            request_id=request_id,
        )

        if isinstance(result, ErrorResponse | RawSpeechResponse):
            return result

        return _sse_stream(result)


async def _sse_stream(chunks: AsyncGenerator[tuple[bytes, int], None]) -> AsyncGenerator[str, None]:
    async for pcm, _sample_rate in chunks:
        audio_b64 = base64.b64encode(pcm).decode("ascii")
        event = SpeechResponse(type="speech.audio.delta", audio=audio_b64)
        yield f"data: {event.model_dump_json()}\n\n"
    done = SpeechResponse(type="speech.audio.done")
    yield f"data: {done.model_dump_json()}\n\n"
