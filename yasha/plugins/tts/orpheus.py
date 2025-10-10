from typing import Literal, Union
import os
from vllm.engine.protocol import EngineClient
from vllm.config import ModelConfig
from transformers import AutoTokenizer
import torch
from vllm import SamplingParams
import logging
from collections.abc import AsyncGenerator
import base64
from yasha.infer.infer_config import SpeechResponse, SpeechRequest, RawSpeechResponse
from vllm.entrypoints.openai.protocol import ErrorInfo, ErrorResponse
import wave
import io

logger = logging.getLogger("ray")

class OrpheusTTSPlugin:
    def __init__(self, engine_client: EngineClient, model_config: ModelConfig):
        from orpheus_tts.decoder import tokens_decoder

        self.tokens_decoder = tokens_decoder
        self.engine_client = engine_client
        tokenizer_path = model_config.tokenizer
        self.tokenizer = self._load_tokenizer(tokenizer_path)

    def _load_tokenizer(self, tokenizer_path):
        """Load tokenizer from local path or HuggingFace hub"""
        try:
            # Check if tokenizer_path is a local directory
            if os.path.isdir(tokenizer_path):
                return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            else:
                return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print(f"Falling back to default tokenizer")
            return AutoTokenizer.from_pretrained("gpt2")

    def _format_prompt(self, prompt: str, voice: str):
        adapted_prompt = f"{voice}: {prompt}"
        prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
        start_token = torch.tensor([[ 128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        prompt_string = self.tokenizer.decode(all_input_ids[0])
        return prompt_string

    async def generate(self, input: str, voice: str, request_id: str, stream_format: Literal["sse", "audio"]) -> Union[RawSpeechResponse, AsyncGenerator[str, None],
        ErrorResponse]:
        logger.info("started generation: %s with voice: %s", input, voice)
        if stream_format=="sse":
            return self.generate_sse(input, voice, request_id)
        else:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                async for audio_bytes in self.generate_audio_bytes_async(input, voice, request_id):
                    logger.info("got some audio bytes for wav: %s", audio_bytes)   # PCM16 bytes from SNAC
                    wf.writeframes(audio_bytes)
            
            return RawSpeechResponse(audio=buf.getvalue(), media_type="audio/wav")

        
    async def generate_sse(self, input: str, voice: str, request_id: str) -> AsyncGenerator[str, None]:
        # tokens_decoder is YOUR async decoder from the snippet
        async for audio_bytes in self.generate_audio_bytes_async(input, voice, request_id):
            logger.info("got some audio bytes: %s", audio_bytes)   # PCM16 bytes from SNAC
            encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
            event_data = SpeechResponse(
                audio=encoded_audio,
                type="speech.audio.delta"
            )
            yield f"data: {event_data.model_dump_json()}\n\n"
        
        completion_event=SpeechResponse(audio=None, type="speech.audio.done")
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

        async for result in self.engine_client.generate(
            prompt=prompt_string,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            chunk = result.outputs[0].text  # cumulative
            if chunk:
                yield chunk

    
    
