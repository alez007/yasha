import logging
from typing import cast

import torch
from starlette.requests import Request

from yasha.infer.base_infer import BaseInfer
from yasha.infer.diffusers.openai.serving_image import OpenAIServingImage
from yasha.infer.infer_config import DiffusersConfig, DisconnectProxy, ModelUsecase, YashaModelConfig
from yasha.openai.protocol import (
    ErrorResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
)

logger = logging.getLogger("ray")

_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class DiffusersInfer(BaseInfer):
    def __init__(self, model_config: YashaModelConfig):
        super().__init__(model_config)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        mem_frac = self._get_memory_fraction()
        if torch.cuda.is_available() and mem_frac is not None:
            torch.cuda.set_per_process_memory_fraction(mem_frac)

    def __del__(self):
        try:
            if pipeline := getattr(self, "_pipeline", None):
                del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            from yasha.metrics import RESOURCE_CLEANUP_ERRORS_TOTAL

            RESOURCE_CLEANUP_ERRORS_TOTAL.inc(tags={"model": self.model_config.name, "component": "diffusers_pipeline"})

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
        ).to(device=self.device, dtype=dtype)

        tokenizer = getattr(self._pipeline, "tokenizer", None)
        if tokenizer is not None:
            self._set_max_context_length(getattr(tokenizer, "model_max_length", None))

        self.serving_image: OpenAIServingImage | None = (
            OpenAIServingImage(pipeline=self._pipeline, config=config)
            if self.model_config.usecase is ModelUsecase.image
            else None
        )

    async def warmup(self) -> None:
        if self.serving_image is None:
            return
        logger.info("Warming up diffusers model: %s", self.model_config.name)
        request = ImageGenerationRequest(
            model=self.model_config.name,
            prompt="warmup",
            n=1,
            size="64x64",
            num_inference_steps=1,
            guidance_scale=0.0,
        )
        await self.create_image_generation(request, DisconnectProxy(None, {}))
        logger.info("Warmup image generation done for %s", self.model_config.name)

    async def create_image_generation(
        self, request: ImageGenerationRequest, raw_request: DisconnectProxy
    ) -> ErrorResponse | ImageGenerationResponse:
        if self.serving_image is None:
            return await super().create_image_generation(request, raw_request)
        return await self.serving_image.create_image_generation(request, cast("Request", raw_request))
