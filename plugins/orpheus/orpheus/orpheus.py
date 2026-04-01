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

from typing import Literal
import os
import wave
import io
import logging
import base64
from collections.abc import AsyncGenerator

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.usage.usage_lib import UsageContext
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from transformers import AutoTokenizer

from yasha.infer.infer_config import SpeechResponse, RawSpeechResponse, YashaModelConfig
from yasha.plugins.base_plugin import BasePlugin

logger = logging.getLogger("ray")


class ModelPlugin(BasePlugin):
    def __init__(self, model_config: YashaModelConfig):
        self.model_config = model_config

        plugin_config = model_config.plugin_config or {}
        max_model_len = plugin_config.get("max_model_len", 2048)
        tokenizer = plugin_config.get("tokenizer", model_config.model)

        logger.info("initialising vllm engine for model: %s, max_model_len: %s, tokenizer: %s", model_config.model, max_model_len, tokenizer)

        engine_args = AsyncEngineArgs(
            model=model_config.model,
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

        from orpheus_tts.decoder import tokens_decoder

        self.tokens_decoder = tokens_decoder
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

    async def generate(self, input: str, voice: str, request_id: str, stream_format: Literal["sse", "audio"]) -> RawSpeechResponse | AsyncGenerator[str, None] | ErrorResponse:
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
                    logger.info("got some audio bytes for wav: %s", audio_bytes)
                    frame_count = len(audio_bytes) // (wf.getsampwidth() * wf.getnchannels())
                    total_frames += frame_count
                    wf.writeframes(audio_bytes)

            duration = total_frames / 24000
            logger.info("audio has total duration of %s seconds", duration)
            return RawSpeechResponse(audio=buf.getvalue(), media_type="audio/wav")

    async def generate_sse(self, input: str, voice: str, request_id: str) -> AsyncGenerator[str, None]:
        async for audio_bytes in self.generate_audio_bytes_async(input, voice, request_id):
            logger.info("got some audio bytes: %s", audio_bytes)
            encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
            event_data = SpeechResponse(audio=encoded_audio, type="speech.audio.delta")
            yield f"data: {event_data.model_dump_json()}\n\n"

        completion_event = SpeechResponse(audio=None, type="speech.audio.done")
        yield f"data: {completion_event.model_dump_json()}\n\n"

    async def generate_audio_bytes_async(self, input: str, voice: str, request_id: str) -> AsyncGenerator[bytes, None]:
        async for audio_bytes in self.tokens_decoder(self.generate_tokens_async(input, voice, request_id)):
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
