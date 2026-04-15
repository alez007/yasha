from collections.abc import AsyncGenerator

from modelship.infer.infer_config import RawRequestProxy
from modelship.logging import get_logger
from modelship.openai.protocol import ErrorResponse, RawSpeechResponse, SpeechRequest, create_error_response
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

        return await self.serving_engine.generate(request.input, request.voice, request_id, request.stream_format)
