from transformers import Pipeline

from modelship.infer.base_serving import OpenAIServing
from modelship.infer.infer_config import RawRequestProxy, TransformersConfig
from modelship.logging import TRACE, get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    RawSpeechResponse,
    SpeechRequest,
    create_error_response,
)
from modelship.utils import base_request_id
from modelship.utils.audio import to_wav

logger = get_logger("infer.transformers.speech")


class OpenAIServingSpeech(OpenAIServing):
    request_id_prefix = "tts"

    def __init__(self, pipeline: Pipeline, model_name: str, config: TransformersConfig):
        self.pipeline = pipeline
        self.model_name = model_name
        self.config = config

    async def warmup(self) -> None:
        logger.info("Warming up TTS model: %s", self.model_name)
        await self.run_in_executor(self._run, "warmup")
        logger.info("Warmup TTS done for %s", self.model_name)

    async def create_speech(
        self, request: SpeechRequest, raw_request: RawRequestProxy
    ) -> RawSpeechResponse | ErrorResponse:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info("speech request %s", request_id)
        logger.log(TRACE, "speech request %s: input=%r", request_id, request.input)

        try:
            result = await self.run_in_executor(self._run, request.input)
        except Exception:
            logger.exception("speech inference failed for %s", request_id)
            return create_error_response("speech inference failed")

        wav_bytes = to_wav(result["audio"], result["sampling_rate"])
        logger.log(
            TRACE,
            "speech response %s: audio_bytes=%d, sample_rate=%d",
            request_id,
            len(wav_bytes),
            result["sampling_rate"],
        )
        return RawSpeechResponse(audio=wav_bytes)

    def _run(self, text: str) -> dict:
        return self.pipeline(text, **self.config.pipeline_kwargs)  # type: ignore[return-value]
