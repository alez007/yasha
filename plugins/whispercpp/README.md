# whisper.cpp STT Plugin

Speech-to-text using [whisper.cpp](https://github.com/ggerganov/whisper.cpp) via [pywhispercpp](https://github.com/absadiki/pywhispercpp). CPU-only, no PyTorch — ideal for low-resource hosts (Intel N100 mini-PCs, ARM boards) where a full transformers stack is too heavy.

## Installation

For local development:

```bash
uv sync --extra whispercpp
```

For Docker, no extra setup is needed — plugins referenced in `models.yaml` are loaded automatically from pre-built wheels via Ray's `runtime_env`.

## Configuration

```yaml
models:
  - name: whisper
    model: base.en
    usecase: transcription
    loader: custom
    plugin: whispercpp
    num_gpus: 0
    num_cpus: 1
    plugin_config:
      n_threads: 2
```

### plugin_config Options

| Option | Type | Default | Description |
|---|---|---|---|
| `models_dir` | string | `<plugins_dir>/whispercpp` | Directory to store/load ggml model files |
| `n_threads` | int | pywhispercpp default | CPU threads for inference |

## Models

Models are specified by their whisper.cpp short name and downloaded automatically on first use. Pick the smallest model that meets your accuracy needs:

| Model | Size | Notes |
|---|---|---|
| `tiny.en` / `tiny` | ~75 MB | Fastest, lowest accuracy |
| `base.en` / `base` | ~150 MB | Recommended for English on a mini-PC |
| `small.en` / `small` | ~500 MB | Better accuracy, ~2× slower than base |
| `medium.en` / `medium` | ~1.5 GB | High accuracy, significant CPU load |
| `large-v3` | ~3 GB | Best accuracy, not recommended for CPU-only boxes |

`.en` variants are English-only and noticeably faster. Drop the suffix for multilingual support.

## Example Request

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file=@audio.wav \
  -F model=whisper
```

Also supports `/v1/audio/translations` — sets `translate: true` to produce English output from non-English audio.
