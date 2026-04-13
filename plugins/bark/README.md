# Bark TTS Plugin

Text-to-speech using [Bark](https://github.com/suno-ai/bark) by Suno. Supports multiple languages and speaker presets.

## Installation

```bash
uv sync --extra bark
```

Or via Docker:

```
MSHIP_PLUGINS=bark
```

## Configuration

```yaml
models:
  - name: bark
    model: suno/bark
    usecase: tts
    loader: custom
    plugin: bark
    num_gpus: 0.30
```

### plugin_config Options

None. Bark has no plugin-specific configuration options.

## Voices

Voice presets follow the pattern `v2/<lang>_speaker_<0-9>`:

| Language | Preset pattern |
|---|---|
| English | `v2/en_speaker_0` ... `v2/en_speaker_9` |
| Chinese | `v2/zh_speaker_0` ... `v2/zh_speaker_9` |
| French | `v2/fr_speaker_0` ... `v2/fr_speaker_9` |
| German | `v2/de_speaker_0` ... `v2/de_speaker_9` |
| Spanish | `v2/es_speaker_0` ... `v2/es_speaker_9` |
| Hindi | `v2/hi_speaker_0` ... `v2/hi_speaker_9` |
| Italian | `v2/it_speaker_0` ... `v2/it_speaker_9` |
| Japanese | `v2/ja_speaker_0` ... `v2/ja_speaker_9` |
| Korean | `v2/ko_speaker_0` ... `v2/ko_speaker_9` |
| Polish | `v2/pl_speaker_0` ... `v2/pl_speaker_9` |
| Portuguese | `v2/pt_speaker_0` ... `v2/pt_speaker_9` |
| Russian | `v2/ru_speaker_0` ... `v2/ru_speaker_9` |
| Turkish | `v2/tr_speaker_0` ... `v2/tr_speaker_9` |

## Example Request

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "bark", "input": "Hello world", "voice": "v2/en_speaker_6", "response_format": "wav"}' \
  --output speech.wav
```

Single-response mode only. SSE streaming is not supported.
