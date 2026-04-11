"""
Orpheus TTS plugin for Yasha.

Voices:

    zoe, zac, jess, leo, mia, julia, leah

Plugin config options (via plugin_config in models.yaml):

    None — all generation parameters are fixed (temp=0.6, top_p=0.8, max_tokens=1200)

Example models.yaml entry:

    - name: "orpheus"
      model: "canopylabs/orpheus-tts-0.1-finetune-prod"
      usecase: "tts"
      loader: "custom"
      plugin: "orpheus"
      num_gpus: 0.4
      plugin_config:
        max_model_len: 2048
        tokenizer: "canopylabs/orpheus-tts-0.1-finetune-prod"

Example request:

    curl http://localhost:8000/v1/audio/speech \\
      -H "Content-Type: application/json" \\
      -d '{"model": "orpheus", "input": "Hello world", "voice": "zoe", "response_format": "wav"}' \\
      --output speech.wav
"""

import base64
import io
import os
import wave
from collections.abc import AsyncGenerator
from typing import Literal

import numpy as np
import torch
from snac import SNAC  # type: ignore[import-unresolved]
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from yasha.infer.infer_config import YashaModelConfig
from yasha.logging import get_logger
from yasha.openai.protocol import ErrorResponse, RawSpeechResponse, SpeechResponse
from yasha.plugins.base_plugin import BasePlugin

logger = get_logger("plugin.orpheus")

_snac_device = os.environ.get("SNAC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
_snac_model: SNAC | None = None


def _get_snac_model() -> SNAC:
    """Lazily load and cache the SNAC 24kHz model on first call."""
    global _snac_model
    if _snac_model is None:
        _snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(_snac_device)
    return _snac_model


def _turn_token_into_id(token_string: str, index: int) -> int | None:
    """Parse a vLLM output chunk and extract the SNAC codec token ID.

    The model emits tokens as strings like ``<custom_token_X>``. Each position
    within a 7-token frame maps to a different SNAC codebook level, so the raw
    token number is adjusted by subtracting a per-position offset
    (``(index % 7) * 4096``) to recover the original codebook index.

    Returns None if the chunk contains no recognisable custom token.
    """
    token_string = token_string.strip()
    last_token_start = token_string.rfind("<custom_token_")
    if last_token_start == -1:
        return None
    last_token = token_string[last_token_start:]
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            return int(last_token[14:-1]) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    return None


def _convert_to_audio(multiframe: list[int]) -> bytes | None:
    """Decode a window of SNAC token IDs into raw PCM audio bytes.

    SNAC 24kHz uses three hierarchical codebooks per frame. Within each
    7-token frame the layout is::

        [c0, c1_a, c2_a, c2_b, c1_b, c2_c, c2_d]

    where c0 feeds codebook 0 (1 token/frame), c1_* feeds codebook 1
    (2 tokens/frame), and c2_* feeds codebook 2 (4 tokens/frame). The
    tokens are deinterleaved into those three tensors before being passed
    to ``SNAC.decode``.

    The decoded waveform is sliced to samples 2048-4096 to discard edge
    artefacts, then converted to signed 16-bit PCM and returned as bytes.
    Returns None if any token is out of the valid 0-4096 range.
    """
    if len(multiframe) < 7:
        return None

    num_frames = len(multiframe) // 7
    frame = multiframe[: num_frames * 7]

    codes_0 = torch.tensor([frame[7 * j] for j in range(num_frames)], device=_snac_device, dtype=torch.int32)
    codes_1 = torch.tensor(
        [frame[7 * j + k] for j in range(num_frames) for k in (1, 4)], device=_snac_device, dtype=torch.int32
    )
    codes_2 = torch.tensor(
        [frame[7 * j + k] for j in range(num_frames) for k in (2, 3, 5, 6)], device=_snac_device, dtype=torch.int32
    )

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
    if any(torch.any(c < 0) or torch.any(c > 4096) for c in codes):
        return None

    with torch.inference_mode():
        audio_hat = _get_snac_model().decode(codes)

    audio_np = audio_hat[:, :, 2048:4096].detach().cpu().numpy()
    return (audio_np * 32767).astype(np.int16).tobytes()


async def _tokens_decoder(token_gen: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
    """Convert a stream of vLLM token strings into a stream of PCM audio chunks.

    Consumes ``token_gen`` and accumulates valid SNAC token IDs in a buffer.
    Audio conversion is triggered every 7 tokens once at least 28 tokens have
    been collected (4 full frames), using a rolling window of the last 28
    tokens. Yields raw PCM bytes for each converted window.
    """
    buffer: list[int] = []
    count = 0
    async for token_string in token_gen:
        token = _turn_token_into_id(token_string, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            if count % 7 == 0 and count > 27:
                audio_samples = _convert_to_audio(buffer[-28:])
                if audio_samples is not None:
                    yield audio_samples


class ModelPlugin(BasePlugin):
    def __init__(self, model_config: YashaModelConfig):
        self.model_config = model_config

        plugin_config = model_config.plugin_config or {}
        max_model_len = plugin_config.get("max_model_len", 2048)
        tokenizer = plugin_config.get("tokenizer", model_config.model)

        logger.info(
            "initialising vllm engine for model: %s, max_model_len: %s, tokenizer: %s",
            model_config.model,
            max_model_len,
            tokenizer,
        )

        engine_args = AsyncEngineArgs(
            model=model_config.model,  # type: ignore[arg-type]
            max_model_len=max_model_len,
            tokenizer=tokenizer,
            gpu_memory_utilization=model_config.num_gpus if model_config.num_gpus < 1.0 else 0.9,
        )

        usage_context = UsageContext.OPENAI_API_SERVER
        engine_config = engine_args.create_engine_config(usage_context=usage_context)

        self.engine = AsyncLLM.from_vllm_config(
            vllm_config=engine_config,
            usage_context=usage_context,
            enable_log_requests=engine_args.enable_log_requests,
            disable_log_stats=engine_args.disable_log_stats,
        )

    def __del__(self):
        try:
            if engine := getattr(self, "engine", None):
                engine.shutdown()
        except Exception:
            pass

    async def start(self):
        vllm_config = self.engine.vllm_config
        tokenizer_path = vllm_config.model_config.tokenizer
        self.tokenizer = self._load_tokenizer(tokenizer_path)

    def _load_tokenizer(self, tokenizer_path):
        try:
            if os.path.isdir(tokenizer_path):
                return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            else:
                return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            logger.warning("Error loading tokenizer %s, falling back to gpt2: %s", tokenizer_path, e)
            return AutoTokenizer.from_pretrained("gpt2")

    def _format_prompt(self, prompt: str, voice: str) -> str:
        formatted_prompt = f"{voice}: {prompt}"
        return f"<|audio|>{formatted_prompt}<|eot_id|>"

    async def generate(
        self, input: str, voice: str, request_id: str, stream_format: Literal["sse", "audio"]
    ) -> RawSpeechResponse | AsyncGenerator[str, None] | ErrorResponse:
        logger.info("started generation: %s with voice: %s", input, voice)
        if stream_format == "sse":
            return self.generate_sse(input, voice, request_id)
        else:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                total_frames = 0
                async for audio_bytes in self.generate_audio_bytes_async(input, voice, request_id):
                    logger.debug("audio chunk bytes=%d", len(audio_bytes))
                    frame_count = len(audio_bytes) // (wf.getsampwidth() * wf.getnchannels())
                    total_frames += frame_count
                    wf.writeframes(audio_bytes)

            duration = total_frames / 24000
            logger.info("audio has total duration of %s seconds", duration)
            return RawSpeechResponse(audio=buf.getvalue(), media_type="audio/wav")

    async def generate_sse(self, input: str, voice: str, request_id: str) -> AsyncGenerator[str, None]:
        async for audio_bytes in self.generate_audio_bytes_async(input, voice, request_id):
            logger.debug("audio chunk bytes=%d", len(audio_bytes))
            encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
            event_data = SpeechResponse(audio=encoded_audio, type="speech.audio.delta")
            yield f"data: {event_data.model_dump_json()}\n\n"

        completion_event = SpeechResponse(audio=None, type="speech.audio.done")
        yield f"data: {completion_event.model_dump_json()}\n\n"

    async def generate_audio_bytes_async(self, input: str, voice: str, request_id: str) -> AsyncGenerator[bytes, None]:
        async for audio_bytes in _tokens_decoder(self.generate_tokens_async(input, voice, request_id)):
            yield audio_bytes

    async def generate_tokens_async(self, input: str, voice: str, request_id: str):
        prompt_string = self._format_prompt(input, voice)
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.8,
            max_tokens=1200,
            stop_token_ids=[49158],
            repetition_penalty=1.3,
        )

        async for result in self.engine.generate(
            prompt=prompt_string,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            chunk = result.outputs[0].text
            if chunk:
                yield chunk
