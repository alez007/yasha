import logging
from collections.abc import AsyncGenerator

from fastapi import Request
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.utils import create_error_response

from yasha.infer.infer_config import (
    RawSpeechResponse,
    SpeechRequest,
)
from yasha.plugins.base_plugin import BasePlugin
from yasha.utils import base_request_id

logger = logging.getLogger("ray")


class OpenAIServingSpeech:
    request_id_prefix = "tts"

    """Handles speech requests"""

    def __init__(self, serving_engine: BasePlugin | None):
        self.serving_engine = serving_engine

    async def create_speech(
        self, request: SpeechRequest, raw_request: Request
    ) -> RawSpeechResponse | AsyncGenerator[str, None] | ErrorResponse:
        if self.serving_engine is None:
            return create_error_response("tts model is not yet accessible")

        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"

        return await self.serving_engine.generate(request.input, request.voice, request_id, request.stream_format)
