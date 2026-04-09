import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from yasha.infer.infer_config import DisconnectProxy, YashaModelConfig
from yasha.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    ErrorInfo,
    ErrorResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    RawSpeechResponse,
    SpeechRequest,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
)

logger = logging.getLogger("ray")

_NOT_SUPPORTED = ErrorResponse(
    error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404)
)


class BaseInfer(ABC):
    def __init__(self, model_config: YashaModelConfig):
        self.model_config = model_config
        self.max_context_length: int | None = None

    def _get_memory_fraction(self) -> float | None:
        """Return the GPU memory fraction if explicitly set and < 1.0, otherwise None."""
        if self.model_config.num_gpus > 0 and self.model_config.num_gpus < 1.0:
            return self.model_config.num_gpus
        return None

    def _set_max_context_length(self, length: int | None) -> None:
        self.max_context_length = length
        logger.info("max_context_length for %s: %s", self.model_config.name, self.max_context_length)

    @abstractmethod
    async def start(self) -> None: ...

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | ChatCompletionResponse | AsyncGenerator[str, None]:
        return _NOT_SUPPORTED

    async def create_embedding(self, request: EmbeddingRequest, raw_request: DisconnectProxy) -> ErrorResponse:
        return _NOT_SUPPORTED

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | TranscriptionResponse | TranscriptionResponseVerbose | AsyncGenerator[str, None]:
        return _NOT_SUPPORTED

    async def create_translation(
        self, audio_data: bytes, request: TranslationRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | TranslationResponse | TranslationResponseVerbose | AsyncGenerator[str, None]:
        return _NOT_SUPPORTED

    async def create_speech(
        self, request: SpeechRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | RawSpeechResponse | AsyncGenerator[str, None]:
        return _NOT_SUPPORTED

    async def create_image_generation(
        self, request: ImageGenerationRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | ImageGenerationResponse:
        return _NOT_SUPPORTED
