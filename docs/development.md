# Development

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (optional — only required for GPU development)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension

## Quick start (Dev Container)

The recommended way to develop Modelship is with VS Code Dev Containers. The configuration in `.devcontainer/` builds the dev image, mounts the repo, forwards ports, and installs all required extensions automatically.

1. Set required environment variables on your host:

   ```bash
   export HF_TOKEN=your_token_here
   export MSHIP_PLUGINS=kokoro  # optional — comma-separated list of plugins to install
   ```

2. Open the repo in VS Code and run **Dev Containers: Reopen in Container** from the command palette (`Ctrl+Shift+P` / `Cmd+Shift+P`).

3. Once inside the container, sync dependencies and start the Ray head node:

   ```bash
   # Sync project deps (add --extra <plugin> for each plugin you need)
   uv sync --extra dev

   # Start the Ray head node
   ray start --head \
     --port=${RAY_REDIS_PORT} \
     --dashboard-host=0.0.0.0 \
     --num-cpus=${RAY_HEAD_CPU_NUM} \
     --num-gpus=${RAY_HEAD_GPU_NUM} \
     --disable-usage-stats
   ```

4. Start the server:

   ```bash
   uv run start.py
   ```

> **Why the extra steps?** The Dev Container overrides the image's default `CMD` (which normally runs `start.sh` to sync deps and start Ray). Inside a Dev Container you need to run these steps manually.

The Dev Container automatically:
- Builds the dev image from `Dockerfile` (target: `dev`)
- Bind-mounts the repo to `/modelship` for live editing
- Forwards ports `8000` (API) and `8265` (Ray Dashboard)
- Installs extensions: Ruff, Python, Pyright, and Claude Code
- Configures the Python interpreter and linting to use the container's venv at `/.venv`

### Environment variables

The following environment variables are set in the dev image with sensible defaults. Override them in your host shell or in `.devcontainer/devcontainer.json` under `remoteEnv` as needed:

| Variable | Default | Description |
|---|---|---|
| `RAY_REDIS_PORT` | `6379` | Ray GCS port |
| `RAY_CLUSTER_ADDRESS` | `ray://0.0.0.0` | Ray cluster address |
| `RAY_HEAD_CPU_NUM` | `2` | CPUs allocated to Ray head |
| `RAY_HEAD_GPU_NUM` | `1` | GPUs allocated to Ray head (set to `0` for CPU-only development) |
| `MSHIP_CACHE_DIR` | `/.cache` | Model cache directory |
| `MSHIP_USE_EXISTING_RAY_CLUSTER` | `false` | Set to `true` to skip starting a Ray head node |

### Installing plugin dependencies for IntelliSense

Plugin packages (e.g. `kokoro_onnx`, `onnxruntime`) are only installed when their extra is enabled. If you're working on a plugin and want full IntelliSense, sync the extras inside the container:

```bash
uv sync --extra kokoro --extra dev
```

## Manual setup (without Dev Containers)

If you prefer not to use Dev Containers, you can build and run the dev image directly.

### Building the dev image

**GPU (Standard):**
```bash
docker build -t modelship_dev --target dev .
```

**CPU (Lightweight):**
```bash
docker build -t modelship_dev_cpu --target dev -f Dockerfile.cpu .
```

### Running with live source mounting

The dev image does not bake in source files. Mount the repo root so changes take effect without rebuilding:

**GPU:**
```bash
docker run -it --rm --shm-size=8g --gpus all \
  -e HF_TOKEN=your_token_here \
  -e MSHIP_PLUGINS=kokoro \
  -e RAY_HEAD_GPU_NUM=1 \
  --mount type=bind,src=./,dst=/modelship \
  -p 8000:8000 modelship_dev
```

**CPU:**
```bash
docker run -it --rm --shm-size=8g \
  -e HF_TOKEN=your_token_here \
  -e RAY_HEAD_GPU_NUM=0 \
  --mount type=bind,src=./,dst=/modelship \
  -p 8000:8000 modelship_dev_cpu
```

The container's entrypoint (`start.sh`) automatically syncs dependencies (including any plugins listed in `MSHIP_PLUGINS`), starts the Ray head node, and drops into a shell. Then start the server:

```bash
uv run start.py
```

## Production Builds

To build the production images locally:

**GPU:**
```bash
docker build -t modelship:latest --target prod .
```

**CPU:**
```bash
docker build -t modelship:latest-cpu --target prod -f Dockerfile.cpu .
```

## Ports

| Port | Service |
|---|---|
| `8000` | API |
| `8265` | Ray dashboard |
