# Development

## Building the dev image

```bash
docker build -t yasha_dev -f Dockerfile.dev .
```

## Running with live source mounting

The dev image does not bake in source files. Mount the repo root so changes take effect without rebuilding:

```bash
docker run -it --rm --shm-size=8g --gpus all \
  -e HF_TOKEN=your_token_here \
  --mount type=bind,src=./,dst=/yasha \
  -p 8265:8265 -p 8000:8000 yasha_dev
```

Inside the container, start the server:

```bash
uv run start.py
```

## Attaching to the container

Use VS Code Dev Containers (or any IDE that supports remote container attachment) to attach to the running container for a full development experience with IntelliSense and debugging.

## Ports

| Port | Service |
|---|---|
| `8000` | API |
| `8265` | Ray dashboard |
