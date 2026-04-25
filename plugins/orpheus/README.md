# Orpheus TTS Plugin

Text-to-speech using [Orpheus TTS](https://github.com/canopylabs/orpheus-tts) by Canopy Labs. Uses vLLM internally for token generation and SNAC for audio decoding.

## Installation

For local development:

```bash
uv sync --extra orpheus
```

For Docker, no extra setup is needed — plugins referenced in `models.yaml` are loaded automatically from pre-built wheels via Ray's `runtime_env`.

## Configuration

```yaml
models:
  - name: orpheus
    model: canopylabs/orpheus-tts-0.1-finetune-prod
    usecase: tts
    loader: custom
    plugin: orpheus
    num_gpus: 0.4
    plugin_config:
      max_model_len: 2048
      tokenizer: canopylabs/orpheus-tts-0.1-finetune-prod
```

### plugin_config Options

| Option | Type | Default | Description |
|---|---|---|---|
| `max_model_len` | int | `2048` | Maximum model sequence length for vLLM engine |
| `tokenizer` | string | same as `model` | Tokenizer path or HuggingFace ID |

## Voices

`zoe`, `zac`, `jess`, `leo`, `mia`, `julia`, `leah`

## Example Request

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "orpheus", "input": "Hello world", "voice": "zoe", "response_format": "wav"}' \
  --output speech.wav
```

Supports both single-response (`wav`) and SSE streaming modes.
