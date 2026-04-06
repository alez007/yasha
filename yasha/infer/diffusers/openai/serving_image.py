import asyncio
import base64
import io
import logging
import time

from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from fastapi import Request

from yasha.infer.infer_config import DiffusersConfig
from yasha.openai.protocol import (
    ErrorResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageObject,
    create_error_response,
)
from yasha.utils import base_request_id

logger = logging.getLogger("ray")


class OpenAIServingImage:
    request_id_prefix = "img"

    def __init__(self, pipeline: AutoPipelineForText2Image, config: DiffusersConfig):
        self.pipeline = pipeline
        self.config = config

    async def create_image_generation(
        self, request: ImageGenerationRequest, raw_request: Request
    ) -> ImageGenerationResponse | ErrorResponse:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info(
            "image generation request %s: prompt=%r, n=%d, size=%s", request_id, request.prompt, request.n, request.size
        )

        try:
            width, height = _parse_size(request.size)
        except ValueError as e:
            return create_error_response(str(e))

        steps = request.num_inference_steps or self.config.num_inference_steps
        guidance = request.guidance_scale or self.config.guidance_scale

        loop = asyncio.get_event_loop()
        images = await loop.run_in_executor(
            None,
            lambda: (
                self.pipeline(  # type: ignore[reportCallIssue]
                    prompt=request.prompt,
                    num_images_per_prompt=request.n,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                ).images
            ),
        )

        data = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data.append(ImageObject(b64_json=b64, revised_prompt=request.prompt))

        return ImageGenerationResponse(created=int(time.time()), data=data)


def _parse_size(size: str) -> tuple[int, int]:
    parts = size.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid size format '{size}', expected WxH (e.g. '512x512')")
    w, h = int(parts[0]), int(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"Width and height must be positive, got {w}x{h}")
    if w % 8 != 0 or h % 8 != 0:
        raise ValueError(f"Width and height must be multiples of 8, got {w}x{h}")
    return w, h
