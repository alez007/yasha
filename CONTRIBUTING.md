# Contributing to Modelship

Thanks for your interest in contributing to Modelship! This document covers the basics for getting started.

## Development Setup

The recommended way to develop is via the VS Code Dev Container. See [docs/development.md](docs/development.md) for full instructions.

**Quick version:**

1. Install Docker + NVIDIA Container Toolkit
2. Set `HF_TOKEN` and `MSHIP_PLUGINS` environment variables
3. Open the project in VS Code and select **Dev Containers: Reopen in Container**
4. Inside the container:

```bash
uv sync --extra dev
ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-cpus=2 --num-gpus=1
uv run start.py
```

## Code Style

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [Pyright](https://github.com/microsoft/pyright) for type checking. Both run in CI on every pull request.

```bash
ruff check .          # lint
ruff format .         # format
pyright               # type check
```

Configuration is in `pyproject.toml`. Key settings: Python 3.12, 120 character line length.

## Submitting Changes

1. Fork the repository and create a branch from `main`
2. Make your changes
3. Ensure `ruff check`, `ruff format --check`, and `pyright` pass
4. Open a pull request against `main` with a clear description of what changed and why

## Writing Plugins

If you're contributing a TTS backend, see [docs/plugins.md](docs/plugins.md) for the plugin architecture and development guide. Plugins live in `plugins/` as isolated uv workspace packages.

## Reporting Issues

Use [GitHub Issues](https://github.com/alez007/modelship/issues). For bugs, include:

- GPU model and VRAM
- Your `models.yaml` configuration
- Docker and NVIDIA driver versions
- Relevant logs from the container

## Security

For security vulnerabilities, see [SECURITY.md](SECURITY.md) for responsible disclosure instructions.
