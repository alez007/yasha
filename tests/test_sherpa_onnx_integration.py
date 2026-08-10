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


def _transcribe(client, audio: bytes, tmp_path) -> str:
    audio_file = tmp_path / "speech.wav"
    audio_file.write_bytes(audio)
    with open(audio_file, "rb") as f:
        return client.audio.transcriptions.create(model="stt-cpp-model", file=f).text.lower()


@pytest.mark.integration
@pytest.mark.sherpa_onnx
class TestSherpaOnnx:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("tts-model", "stt-cpp-model")

    def test_multiple_voices_produce_distinct_audio(self, client):
        # A sample across the registry, not all 11 — enough to prove sid
        # resolution actually varies the speaker, not exhaustive coverage.
        sample = (_VOICE_NAMES[0], _VOICE_NAMES[len(_VOICE_NAMES) // 2], _VOICE_NAMES[-1])
        clips = [
            client.audio.speech.create(model="tts-model", voice=voice, input="The quick brown fox.").content
            for voice in sample
        ]
        assert len({clips[0], clips[1], clips[2]}) == len(sample)

    def test_speech_is_intelligible_via_transcription(self, client, tmp_path):
        # Structural WAV checks alone don't prove the audio is clean speech
        # rather than silence/noise that happens to be shaped like a WAV file
        # — round-trip it through a real STT model (whispercpp, CPU-only so
        # it can run alongside sherpa_onnx without a GPU) to confirm it is.
        audio = client.audio.speech.create(
            model="tts-model", voice="af_bella", input="The wizard counted seventeen purple bananas."
        ).content
        text = _transcribe(client, audio, tmp_path)
        assert "wizard" in text
        assert "banana" in text

    def test_different_voices_are_all_intelligible(self, client, tmp_path):
        sample = (_VOICE_NAMES[0], _VOICE_NAMES[len(_VOICE_NAMES) // 2], _VOICE_NAMES[-1])
        for voice in sample:
            audio = client.audio.speech.create(
                model="tts-model", voice=voice, input="Open the door and turn on the light."
            ).content
            text = _transcribe(client, audio, tmp_path)
            assert "door" in text or "light" in text, f"voice {voice!r} transcribed to unintelligible text: {text!r}"

    def test_numeric_voice_is_accepted(self, client, tmp_path):
        # sherpa's kokoro model isn't byte-deterministic call-to-call (stochastic
        # vocoder), so this can't assert equality against the named-voice
        # request — only that the digit path round-trips through the API to
        # real, intelligible inference instead of being rejected somewhere
        # along the way.
        audio = client.audio.speech.create(model="tts-model", voice="0", input="Numeric voice selection.").content
        assert "numeric" in _transcribe(client, audio, tmp_path)

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
