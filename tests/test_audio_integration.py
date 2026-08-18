"""Cross-loader audio round trip: sherpa_onnx TTS output fed back in as
vllm-whisper STT input, through the modelship gateway."""

import pytest


@pytest.mark.integration
class TestAudio:
    """tts-model and stt-model are both small enough to share a class;
    test_audio_transcription re-uses the TTS output as its input."""

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("tts-model", "stt-model")

    def test_audio_speech(self, client):
        response = client.audio.speech.create(model="tts-model", voice="af_bella", input="Hello from integration test")
        assert len(response.content) > 1000

    def test_audio_transcription(self, client, tmp_path):
        audio_data = client.audio.speech.create(
            model="tts-model", voice="af_bella", input="This is a test transcription."
        ).content

        audio_file = tmp_path / "test_audio.mp3"
        audio_file.write_bytes(audio_data)

        with open(audio_file, "rb") as f:
            transcription = client.audio.transcriptions.create(model="stt-model", file=f)
        assert "test" in transcription.text.lower()
