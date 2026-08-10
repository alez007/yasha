import base64
from collections.abc import AsyncGenerator

from modelship.openai.protocol import SpeechResponse


async def sse_stream_speech(chunks: AsyncGenerator[tuple[bytes, int], None]) -> AsyncGenerator[str, None]:
    """Frame a stream of `(pcm16_bytes, sample_rate)` chunks as
    `speech.audio.delta`/`speech.audio.done` SSE events. Shared by every
    loader that streams TTS output."""
    async for pcm, _sample_rate in chunks:
        audio_b64 = base64.b64encode(pcm).decode("ascii")
        event = SpeechResponse(type="speech.audio.delta", audio=audio_b64)
        yield f"data: {event.model_dump_json()}\n\n"
    done = SpeechResponse(type="speech.audio.done")
    yield f"data: {done.model_dump_json()}\n\n"
