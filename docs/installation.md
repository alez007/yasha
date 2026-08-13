# Installation

## Requirements

- **Docker** (or Python 3.12.10 with `uv` for local development — `uv` fetches this
  exact version automatically, matching the repo's pinned `requires-python`)
- **NVIDIA GPU** (optional) — 16 GB+ VRAM recommended for a full stack (LLM +
  TTS + STT + embeddings) via vLLM; 8 GB is sufficient for lighter setups.
  Not required when using the vLLM or llama.cpp backends on CPU
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)**
  — required only when running GPU models in Docker
- **HuggingFace token** for gated models

## Image variants

Three images are published from a single `Dockerfile`, all under
`ghcr.io/modelship-ai/modelship`:

| Variant | Tag suffix | Platforms | Purpose |
|---|---|---|---|
| Thin | *(none)* — bare tag | amd64, arm64 | Control/coordinator image, no torch/vllm. Runs the driver/head role only — cannot serve models by itself |
| CUDA | `-cuda` | amd64 | GPU node image (vLLM, Diffusers, llama.cpp GPU offload) |
| CPU | `-cpu` | amd64, arm64 | CPU node image (vLLM CPU backend, llama.cpp, stable_diffusion_cpp) |

Floating tags (`:latest`, `:latest-cuda`, `:latest-cpu`) are single-node
only — multi-node deployments must pin every node to the same `X.Y.Z`
(`-cuda`/`-cpu`) tag, or Ray refuses to form the cluster across mismatched
versions.

## Running a single node

CPU, no GPU required:

```bash
docker run --rm --shm-size=8g \
  -v ./models.yaml:/modelship/config/models.yaml \
  -v ./models-cache:/.cache \
  -p 8000:8000 \
  ghcr.io/modelship-ai/modelship:latest-cpu
```

GPU, with the NVIDIA Container Toolkit installed:

```bash
docker run --rm --shm-size=8g --gpus all \
  -e HF_TOKEN=your_token_here \
  -v ./models.yaml:/modelship/config/models.yaml \
  -v ./models-cache:/.cache \
  -p 8000:8000 \
  ghcr.io/modelship-ai/modelship:latest-cuda
```

See [Quickstart](quickstart.md) for a full copy-pasteable `models.yaml` and
walkthrough.

!!! tip
    Always set `--shm-size=8g` (or higher) — Ray falls back to slower
    disk-backed storage instead of `/dev/shm` if the container's shared
    memory is too small for the object store.

## Native install (Apple Silicon)

No Docker required — `mship` installs directly and runs its own Ray head with
full Metal GPU offload for `llama_server` (GGUF chat/embeddings/vision) and
`stable_diffusion_cpp` (image generation). Install Xcode Command Line Tools
first: `[metal]` compiles `stable-diffusion-cpp-python` from source on first
install, which fails partway through with a raw compiler error if no
compiler is present. `xcode-select -p` checks whether you already have it.

```bash
xcode-select --install          # first-time only; skip if already installed
uv tool install "mship[metal]"
mship deploy --config models.yaml
```

`uv tool install` auto-fetches the pinned Python 3.12.10 interpreter;
`pip install "mship[metal]"` works too if that exact version is already
present (same Xcode CLI Tools prerequisite applies). The first install
compiles stable-diffusion.cpp and takes a few minutes — that's expected, not
a hang.

## Native install (Linux, CPU)

`mship[cpu]` is torch-free — `llama_server` (auto-provisioned, same as
Metal), `whispercpp`, `sherpa_onnx`, and `stable_diffusion_cpp` all run
without pulling in CUDA. Install `build-essential`/`cmake` first —
`stable-diffusion-cpp-python` compiles from source on first install, same
requirement as Xcode CLI Tools on Metal.

```bash
sudo apt-get install -y build-essential cmake   # first-time only
uv tool install "mship[cpu]"
mship deploy --config models.yaml
```

For CPU vLLM (`loader: vllm`, `num_gpus: 0`), add the `vllm-cpu` extra —
also install `libnuma1`/`openssl` and pass both indexes:

```bash
sudo apt-get install -y libnuma1 openssl   # first-time only
uv tool install "mship[cpu,vllm-cpu]" \
  --index https://download.pytorch.org/whl/cpu \
  --index https://wheels.vllm.ai/0.26.0/cpu \
  --index-strategy unsafe-best-match
```

## Native install (Linux, CUDA)

`mship[cuda]` installs outside Docker on any Linux host with an NVIDIA driver.
Unlike `vllm-cpu` it needs **no `--index` flags** — PyPI's default `torch`
wheel is already the CUDA 13.0 build.

Beyond `build-essential`/`cmake` (the same source-compile step as `[cpu]`), it
needs `ninja-build` and NVIDIA's `nvcc`: flashinfer JIT-compiles its sampling
kernel at model-load time, so without a CUDA toolkit the vLLM engine dies
during init with `RuntimeError: Could not find nvcc and default
cuda_home='/usr/local/cuda' doesn't exist`.

```bash
# first-time only
sudo apt-get install -y build-essential cmake ninja-build

# NVIDIA apt repo (Ubuntu 24.04) — CUDA 13.0, matching the -cuda image's pin
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
  | sudo gpg --dearmor -o /usr/share/keyrings/cuda-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
  | sudo tee /etc/apt/sources.list.d/cuda.list
sudo apt-get update && sudo apt-get install -y \
  cuda-nvcc-13-0 cuda-cuobjdump-13-0 libcurand-dev-13-0

uv tool install "mship[cuda]"
mship deploy --config models.yaml
```

`nvcc` does not need to be on `PATH` — the packages create the
`/usr/local/cuda` symlink that flashinfer looks for. The first vLLM deploy is
slow while kernels compile; they're cached per GPU architecture under
`~/.cache/flashinfer`.

**Loader coverage.** `vllm` and `diffusers` get full GPU. `llama_server` is
CPU-only on a native install — the auto-provisioned binary carries no CUDA
backend, so use the `-cuda` image for GGUF offload, or point
`MSHIP_LLAMA_SERVER_BIN` at your own CUDA-enabled `llama-server`.
`stable_diffusion_cpp`, `whispercpp`, and `sherpa_onnx` are CPU-only here,
same as in the image.

## Local development

For building from source, running inside the dev container, or a manual
`uv sync` + `mship_deploy.py` workflow (including the full CLI/env var
reference and port list), see [Development Setup](development.md).

## Scaling beyond one node

To join multiple hosts into one Ray cluster with plain `docker run` (no
Kubernetes), see [Multi-node without Kubernetes](multi-node-docker.md). For a
Kubernetes/KubeRay deployment, see the
[Helm chart](https://github.com/modelship-ai/modelship/tree/main/helm/modelship).
