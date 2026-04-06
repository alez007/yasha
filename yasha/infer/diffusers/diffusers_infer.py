import logging
from typing import cast

import torch
from starlette.requests import Request

from yasha.infer.diffusers.openai.serving_image import OpenAIServingImage
from yasha.infer.infer_config import DiffusersConfig, DisconnectProxy, ModelUsecase, YashaModelConfig
from yasha.openai.protocol import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ErrorInfo,
    ErrorResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    SpeechRequest,
    TranscriptionRequest,
    TranslationRequest,
)

logger = logging.getLogger("ray")

_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class DiffusersInfer:
    def __init__(self, model_config: YashaModelConfig):
        self.model_config = model_config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available() and model_config.num_gpus < 1.0:
            torch.cuda.set_per_process_memory_fraction(model_config.num_gpus)

    def __del__(self):
        try:
            if pipeline := getattr(self, "_pipeline", None):
                del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    async def start(self):
        from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image

        config = self.model_config.diffusers_config or DiffusersConfig()
        dtype = _TORCH_DTYPES.get(config.torch_dtype, torch.float16)

        logger.info(
            "Loading diffusers pipeline: %s (dtype=%s, device=%s)",
            self.model_config.model,
            config.torch_dtype,
            self.device,
        )
        self._pipeline = AutoPipelineForText2Image.from_pretrained(
            self.model_config.model,
            torch_dtype=dtype,
        ).to(self.device)

        self.serving_image: OpenAIServingImage | None = (
            OpenAIServingImage(pipeline=self._pipeline, config=config)
            if self.model_config.usecase is ModelUsecase.image
            else None
        )

    async def create_chat_completion(
        self, _request: ChatCompletionRequest, _raw_request: DisconnectProxy
    ) -> ErrorResponse:
        return ErrorResponse(
            error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404)
        )

    async def create_embedding(self, _request: EmbeddingRequest, _raw_request: DisconnectProxy) -> ErrorResponse:
        return ErrorResponse(
            error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404)
        )

    async def create_transcription(
        self, _audio_data: bytes, _request: TranscriptionRequest, _raw_request: DisconnectProxy
    ) -> ErrorResponse:
        return ErrorResponse(
            error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404)
        )

    async def create_translation(
        self, _audio_data: bytes, _request: TranslationRequest, _raw_request: DisconnectProxy
    ) -> ErrorResponse:
        return ErrorResponse(
            error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404)
        )

    async def create_speech(self, _request: SpeechRequest, _raw_request: DisconnectProxy) -> ErrorResponse:
        return ErrorResponse(
            error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404)
        )

    async def create_image_generation(
        self, request: ImageGenerationRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | ImageGenerationResponse:
        if self.serving_image is None:
            return ErrorResponse(
                error=ErrorInfo(message="model does not support this action", type="invalid_request_error", code=404)
            )
        return await self.serving_image.create_image_generation(request, cast("Request", raw_request))
