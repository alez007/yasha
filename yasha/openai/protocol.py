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


# -- image generation (not yet provided by vllm) ---------------------------
class ImageGenerationRequest(OpenAIBaseModel):
    model: str = Field(..., description="The model to use for image generation.")
    prompt: str = Field(..., description="A text description of the desired image(s).")
    n: int = Field(default=1, ge=1, le=10, description="The number of images to generate.")
    size: str = Field(default="512x512", description="The size of the generated images in WxH format.")
    response_format: Literal["b64_json"] = Field(
        default="b64_json",
        description="The format in which the generated images are returned.",
    )
    num_inference_steps: int | None = Field(default=None, description="Override default inference steps.")
    guidance_scale: float | None = Field(default=None, description="Override default guidance scale.")


class ImageObject(OpenAIBaseModel):
    b64_json: str = Field(..., description="The base64-encoded JSON of the generated image.")
    revised_prompt: str | None = Field(default=None, description="The prompt that was used to generate the image.")


class ImageGenerationResponse(OpenAIBaseModel):
    created: int = Field(..., description="The Unix timestamp of when the response was created.")
    data: list[ImageObject] = Field(..., description="The list of generated images.")


__all__ = [
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ErrorInfo",
    "ErrorResponse",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "ImageObject",
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
