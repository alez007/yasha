# Kokoro ONNX TTS Plugin

ONNX-based text-to-speech using [Kokoro](https://github.com/thewh1teagle/kokoro-onnx). Fast, lightweight, supports GPU and CPU inference.

## Installation

```bash
uv sync --extra kokoroonnx
```

Or via Docker:

```
MSHIP_PLUGINS=kokoroonnx
```

## Requirements

- `espeak-ng` must be installed on the system (`apt-get install -y espeak-ng`)

## Configuration

```yaml
models:
  - name: kokoro
    model: hexgrad/Kokoro-82M
    usecase: tts
    loader: custom
    plugin: kokoroonnx
    num_gpus: 0.07
    plugin_config:
      onnx_provider: CUDAExecutionProvider
```

### plugin_config Options

| Option | Type | Default | Description |
|---|---|---|---|
| `onnx_provider` | string | `CUDAExecutionProvider` | ONNX Runtime execution provider. Options: `CUDAExecutionProvider`, `CPUExecutionProvider`, `TensorrtExecutionProvider` |
| `sample_rate` | int | model native (~24000) | Resample output audio to this rate in Hz |

### CPU-only Example

```yaml
models:
  - name: kokoro
    model: hexgrad/Kokoro-82M
    usecase: tts
    loader: custom
    plugin: kokoroonnx
    num_gpus: 0
    num_cpus: 1
    plugin_config:
      onnx_provider: CPUExecutionProvider
```

## Voices

| Region | Voices |
|---|---|
| American Female | `af_heart`, `af_bella`, `af_nicole`, `af_sarah`, `af_sky` |
| American Male | `am_adam`, `am_michael` |
| British Female | `bf_emma`, `bf_isabella` |
| British Male | `bm_george`, `bm_lewis` |

Full list available in `voices-v1.0.bin` (downloaded automatically).

## Example Request

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Hello world", "voice": "af_heart", "response_format": "wav"}' \
  --output speech.wav
```

Supports both single-response (`wav`) and SSE streaming modes.
