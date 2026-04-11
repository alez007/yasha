# Plugin Development

Plugins are Python packages that extend Modelship with new TTS backends. Each plugin is a self-contained uv workspace package with its own dependencies, installed on demand.

## How plugins work

When `loader: custom` is set in `models.yaml`, Modelship imports the module named by `plugin` and expects it to expose a `ModelPlugin` class extending `BasePlugin`.

`BasePlugin` requires three methods:

| Method | Description |
|---|---|
| `__init__(model_config: ModelshipModelConfig)` | Initialize the plugin; store config and set up state |
| `async start()` | Load models into memory; called once before the first request |
| `async generate(input, voice, request_id, stream_format)` | Generate audio; return a `RawSpeechResponse` or an SSE `AsyncGenerator[str, None]` |

## Creating a plugin

### 1. Create the package structure

```
plugins/
  myplugin/
    pyproject.toml
    myplugin/
      __init__.py
      myplugin.py
```

### 2. Write `pyproject.toml`

```toml
[project]
name = "myplugin"
version = "0.1.0"
dependencies = [
    "modelship",
    # your plugin's dependencies
]
```

### 3. Implement `ModelPlugin`

```python
# plugins/myplugin/myplugin/myplugin.py
from collections.abc import AsyncGenerator
from typing import Literal

from modelship.plugins.base_plugin import BasePlugin
from modelship.infer.infer_config import ModelshipModelConfig
from modelship.openai.protocol import ErrorResponse, RawSpeechResponse


class ModelPlugin(BasePlugin):
    def __init__(self, model_config: ModelshipModelConfig):
        self.model_name = model_config.model
        self.config = model_config.plugin_config or {}

    async def start(self):
        # load your model here
        pass

    async def generate(
        self,
        input: str,
        voice: str,
        request_id: str,
        stream_format: Literal["sse", "audio"],
    ) -> RawSpeechResponse | AsyncGenerator[str, None] | ErrorResponse:
        audio_bytes = b"..."  # your synthesis here
        return RawSpeechResponse(audio=audio_bytes)
```

### 4. Export `ModelPlugin` from `__init__.py`

```python
# plugins/myplugin/myplugin/__init__.py
from myplugin.myplugin import ModelPlugin

__all__ = ["ModelPlugin"]
```

### 5. Register the extra in the root `pyproject.toml`

```toml
[project.optional-dependencies]
myplugin = ["myplugin"]
```

### 6. Install and configure

```bash
uv sync --extra myplugin
```

In `models.yaml`:

```yaml
- name: myplugin
  usecase: tts
  loader: custom
  plugin: myplugin
  num_gpus: 0.1
```

## SSE streaming

To support streaming, yield SSE-formatted strings instead of returning a `RawSpeechResponse`:

```python
async def generate(self, input, voice, request_id, stream_format):
    if stream_format == "sse":
        async def _stream():
            for chunk in self._synthesize_chunks(input):
                import base64, json
                b64 = base64.b64encode(chunk).decode()
                yield f"data: {json.dumps({'type': 'speech.audio.delta', 'audio': b64})}\n\n"
            yield f"data: {json.dumps({'type': 'speech.audio.done'})}\n\n"
        return _stream()
    else:
        audio = self._synthesize_full(input)
        return RawSpeechResponse(audio=audio)
```

## Submitting to this repo

Open a PR adding:
- `plugins/myplugin/` with your package
- One line in root `pyproject.toml` optional extras
