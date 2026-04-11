# Home Assistant Integration

Modelship exposes an OpenAI-compatible API, but Home Assistant's built-in voice assistant pipeline does not speak the OpenAI protocol directly. Instead, it uses the **Wyoming protocol** — a lightweight, purpose-built protocol for local voice services (STT, TTS, wake word).

To bridge the two, Modelship uses [wyoming-openai](https://github.com/roryeckel/wyoming_openai): a small Docker image that sits between Home Assistant and Modelship, translating Wyoming requests into OpenAI API calls.

## Setup

Run `wyoming-openai` alongside Modelship, pointing it at Modelship's API endpoint:

```bash
docker run -it -p 10300:10300 \
  -e WYOMING_URI="tcp://0.0.0.0:10300" \
  -e WYOMING_LOG_LEVEL="INFO" \
  -e WYOMING_LANGUAGES="en" \
  -e STT_OPENAI_URL="http://<modelship-host>:8000/v1" \
  -e STT_MODELS="whisper" \
  -e STT_STREAMING_MODELS="whisper" \
  -e STT_BACKEND="OPENAI" \
  -e TTS_OPENAI_URL="http://<modelship-host>:8000/v1" \
  -e TTS_MODELS="kokoro" \
  -e TTS_STREAMING_MODELS="kokoro" \
  -e TTS_VOICES="af_heart" \
  -e TTS_SPEED="1.0" \
  -e TTS_BACKEND="OPENAI" \
  ghcr.io/roryeckel/wyoming_openai:latest
```

Replace `<modelship-host>` with the address of your Modelship instance:

- `localhost` or `0.0.0.0` — if running on the same machine
- A Docker container name — if both containers share a Docker network
- A domain or IP — if Modelship is exposed over the network or internet

Then in Home Assistant, add a **Wyoming** integration pointing to `<host>:10300`.

The `STT_MODELS` and `TTS_MODELS` values must match the `name` fields in your `models.yaml`.
