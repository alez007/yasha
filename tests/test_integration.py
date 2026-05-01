import os
import signal
import subprocess
import time

import httpx
import pytest
import yaml

from openai import OpenAI

OPENAI_API_BASE = "http://localhost:8000/v1"


@pytest.fixture(scope="session")
def mship_cluster(tmp_path_factory):
    """Starts a Ray cluster and mship_deploy in the background and waits for it to be ready."""
    # Create a session-unique config path
    tmp_dir = tmp_path_factory.mktemp("mship_integration")
    config_path = tmp_dir / "integration-models.yaml"

    # Ensure Ray is stopped first to start fresh
    subprocess.run(["ray", "stop", "--force"], check=False)
    subprocess.run(["ray", "start", "--head", "--dashboard-host=0.0.0.0", "--disable-usage-stats"], check=True)

    # Ensure config exists
    config = {
        "models": [
            {
                "name": "chat-capable",
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "usecase": "generate",
                "loader": "vllm",  # Use vLLM for reliable tool calling
                "num_gpus": 0.1,
            },
            {
                "name": "chat-limited",
                "model": "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf",
                "usecase": "generate",
                "loader": "llama_cpp",
                "num_cpus": 1,
            },
            {
                "name": "embed-model",
                "model": "nomic-ai/nomic-embed-text-v1.5",
                "usecase": "embed",
                "loader": "transformers",
                "num_cpus": 1,
            },
            {
                "name": "stt-model",
                "model": "openai/whisper-tiny",
                "usecase": "transcription",
                "loader": "transformers",
                "num_cpus": 1,
            },
            {
                "name": "tts-model",
                "model": "hexgrad/Kokoro-82M",
                "usecase": "tts",
                "loader": "custom",
                "plugin": "kokoroonnx",
                "num_cpus": 1,
                "plugin_config": {"onnx_provider": "CPUExecutionProvider"},
            },
        ]
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Start deployment
    proc = subprocess.Popen(
        ["uv", "run", "mship_deploy.py", "--config", str(config_path), "--redeploy"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    # Wait for ready
    start_time = time.time()
    timeout = 900  # 15 minutes (vLLM can be slow to init)
    ready = False

    while time.time() - start_time < timeout:
        try:
            resp = httpx.get(f"{OPENAI_API_BASE}/models")
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if len(models) >= 5:
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(15)

    if not ready:
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "timeout", "timeout"
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        subprocess.run(["ray", "stop", "--force"], check=False)
        pytest.fail(f"Deployment failed to become ready within timeout.\nSTDOUT: {stdout}\nSTDERR: {stderr}")

    yield proc

    # Cleanup
    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

    subprocess.run(["ray", "stop", "--force"], check=False)


@pytest.fixture(scope="session")
def client(mship_cluster):
    return OpenAI(base_url=OPENAI_API_BASE, api_key="not-needed")


@pytest.mark.integration
def test_list_models(client):
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    assert "chat-capable" in model_ids
    assert "chat-limited" in model_ids
    assert "embed-model" in model_ids
    assert "stt-model" in model_ids
    assert "tts-model" in model_ids


@pytest.mark.integration
def test_chat_completion(client):
    completion = client.chat.completions.create(
        model="chat-capable", messages=[{"role": "user", "content": "Hello!"}], max_tokens=10
    )
    assert completion.choices[0].message.content
    assert completion.model == "chat-capable"


@pytest.mark.integration
def test_chat_streaming(client):
    stream = client.chat.completions.create(
        model="chat-capable",
        messages=[{"role": "user", "content": "Tell me a short story."}],
        max_tokens=20,
        stream=True,
    )
    chunks = []
    for chunk in stream:
        if chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
    assert len(chunks) > 0


@pytest.mark.integration
def test_tool_calling_success(client):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]
    completion = client.chat.completions.create(
        model="chat-capable",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=tools,
        tool_choice="required",
    )
    assert completion.choices[0].message.tool_calls
    assert completion.choices[0].message.tool_calls[0].function.name == "get_weather"


@pytest.mark.integration
def test_tool_calling_unsupported_loader(client):
    """Verifies that loaders without tool support (like llama_cpp) don't return tool calls."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    # Loader currently ignores the tools param
    completion = client.chat.completions.create(
        model="chat-limited", messages=[{"role": "user", "content": "Weather in London?"}], tools=tools
    )
    assert not completion.choices[0].message.tool_calls


@pytest.mark.integration
def test_embeddings(client):
    response = client.embeddings.create(model="embed-model", input=["Hello world", "Modelship is great"])
    assert len(response.data) == 2
    assert len(response.data[0].embedding) > 0


@pytest.mark.integration
def test_audio_speech(client):
    response = client.audio.speech.create(model="tts-model", voice="af_bella", input="Hello from integration test")
    # response.content is the binary audio data
    assert len(response.content) > 1000


@pytest.mark.integration
def test_audio_transcription(client, tmp_path):
    # Generate audio first using TTS
    audio_data = client.audio.speech.create(
        model="tts-model", voice="af_bella", input="This is a test transcription."
    ).content

    audio_file = tmp_path / "test_audio.mp3"
    audio_file.write_bytes(audio_data)

    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(model="stt-model", file=f)
    assert "test" in transcription.text.lower()
