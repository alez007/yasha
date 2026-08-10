"""End-to-end coverage for the sherpa_onnx loader against a real sherpa-onnx
binding and a real downloaded kokoro bundle. `test_sherpa_onnx_infer.py` covers
the same logic against a stub."""

import io
import json
import wave

import httpx
import pytest

import openai
from modelship.infer.sherpa_onnx.registry import REGISTRY

OPENAI_API_BASE = "http://localhost:8000/v1"
_VOICE_NAMES = REGISTRY["kokoro-en-v0_19"].voice_names


def _post_speech(**fields) -> httpx.Response:
    return httpx.post(f"{OPENAI_API_BASE}/audio/speech", json={"model": "tts-model", **fields}, timeout=120)


def _sse_events(body: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ") and line[len("data: ") :].strip() not in ("", "[DONE]")
    ]


def _wav_frames_and_rate(audio: bytes) -> tuple[int, int]:
    with wave.open(io.BytesIO(audio)) as w:
        return w.getnframes(), w.getframerate()


@pytest.mark.integration
@pytest.mark.sherpa_onnx
class TestSherpaOnnx:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("tts-model")

    def test_speech_returns_valid_wav_audio(self, client):
        response = client.audio.speech.create(
            model="tts-model", voice="af_bella", input="Hello from an integration test."
        )
        audio = response.content
        assert audio.startswith(b"RIFF") and b"WAVE" in audio[:16]
        with wave.open(io.BytesIO(audio)) as w:
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getnframes() > 0

    def test_multiple_voices_produce_distinct_audio(self, client):
        # A sample across the registry, not all 11 — enough to prove sid
        # resolution actually varies the speaker, not exhaustive coverage.
        sample = (_VOICE_NAMES[0], _VOICE_NAMES[len(_VOICE_NAMES) // 2], _VOICE_NAMES[-1])
        clips = [
            client.audio.speech.create(model="tts-model", voice=voice, input="The quick brown fox.").content
            for voice in sample
        ]
        assert len({clips[0], clips[1], clips[2]}) == len(sample)

    def test_numeric_voice_is_accepted(self, client):
        # sherpa's kokoro model isn't byte-deterministic call-to-call (stochastic
        # vocoder), so this can't assert equality against the named-voice
        # request — only that the digit path round-trips through the API to
        # real inference instead of being rejected somewhere along the way.
        audio = client.audio.speech.create(model="tts-model", voice="0", input="Numeric voice.").content
        assert audio.startswith(b"RIFF")
        assert _wav_frames_and_rate(audio)[0] > 0

    def test_unknown_voice_returns_error(self, client):
        with pytest.raises(openai.BadRequestError, match="unknown voice"):
            client.audio.speech.create(model="tts-model", voice="not-a-real-voice", input="hi")

    def test_speed_parameter_changes_duration(self, client):
        normal = client.audio.speech.create(
            model="tts-model", voice="af_bella", input="This sentence has a normal speaking speed.", speed=1.0
        ).content
        fast = client.audio.speech.create(
            model="tts-model", voice="af_bella", input="This sentence has a normal speaking speed.", speed=2.0
        ).content
        normal_frames, normal_rate = _wav_frames_and_rate(normal)
        fast_frames, fast_rate = _wav_frames_and_rate(fast)
        assert (fast_frames / fast_rate) < (normal_frames / normal_rate)

    def test_streaming_sse_speech(self):
        response = _post_speech(voice="af_bella", input="Streaming synthesis test.", stream_format="sse")
        assert response.status_code == 200
        events = _sse_events(response.text)
        assert events, "expected at least one SSE event"
        assert events[-1]["type"] == "speech.audio.done"
        deltas = [e for e in events[:-1] if e["type"] == "speech.audio.delta"]
        assert deltas, "expected at least one speech.audio.delta event"
        assert all(e["audio"] for e in deltas)

    def test_response_format_is_accepted_but_audio_is_always_wav(self):
        # SherpaOnnxInfer's RawSpeechResponse always carries media_type
        # audio/wav; response_format is accepted by the request schema but
        # not currently used to transcode the output.
        response = _post_speech(voice="af_bella", input="Format check.", response_format="mp3")
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content.startswith(b"RIFF")
