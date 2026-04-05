"""
Centralised re-exports of the OpenAI-compatible protocol models used for
FastAPI request/response validation.

Every backend (vllm, transformers, custom) and the API gateway import from
here instead of reaching into vllm internals directly.  If the upstream
module paths change, or we decide to define our own models, only this file
needs updating.
"""

from typing import Literal

from pydantic import BaseModel, Field

# -- chat completion --------------------------------------------------------
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

# -- engine / base ----------------------------------------------------------
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    OpenAIBaseModel,
)

# -- speech-to-text ---------------------------------------------------------
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
)

# -- embeddings -------------------------------------------------------------
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingRequest,
    EmbeddingResponse,
)

# -- utilities --------------------------------------------------------------
from vllm.entrypoints.utils import create_error_response


# -- text-to-speech (not yet provided by vllm) ------------------------------
class SpeechRequest(OpenAIBaseModel):
    input: str = Field(..., description="The text to generate audio for")
    model: str = Field(
        ...,
        description="The model to use for generation.",
    )
    voice: str = Field(
        ...,
        description="The voice to use for generation.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. Select a value from 0.25 to 4.0.",
    )
    stream_format: Literal["sse", "audio"] = Field(
        default="audio",
        description="The stream format to return the audio in.",
    )


class SpeechResponse(OpenAIBaseModel):
    audio: str | None = Field(default=None, description="The generated audio data encoded in base 64")
    type: Literal["speech.audio.delta", "speech.audio.done"] = Field(
        ...,
        description="Type of audio chunk",
    )


class RawSpeechResponse(BaseModel):
    audio: bytes = Field(..., description="full audio file bytes")
    media_type: Literal["audio/wav"] = Field(default="audio/wav", description="audio bytes media type")


__all__ = [
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ErrorInfo",
    "ErrorResponse",
    "OpenAIBaseModel",
    "RawSpeechResponse",
    "SpeechRequest",
    "SpeechResponse",
    "TranscriptionRequest",
    "TranscriptionResponse",
    "TranscriptionResponseVerbose",
    "TranslationRequest",
    "TranslationResponse",
    "TranslationResponseVerbose",
    "create_error_response",
]
