# Installation

## Requirements

- **Docker**, or any **Python 3.10+** for a native install — `mship` provisions the
  CPython 3.12.10 the engine needs itself
- **NVIDIA GPU** (optional) — 16 GB+ VRAM recommended for a full stack (LLM +
  TTS + STT + embeddings) via vLLM; 8 GB is sufficient for lighter setups.
  Not required when using the vLLM or llama.cpp backends on CPU
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)**
  — required only when running GPU models in Docker
- **HuggingFace token** for gated models
- **glibc** for a native install — `ray` publishes no `musllinux` wheels, so Alpine
  and other musl distros are refused. Use a glibc distro, or the Docker images,
  which are Debian-based.

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

## Native install

One command on every platform — the node's role is chosen at run time, not at
install time:

```bash
pipx install mship          # or: uv tool install mship / pip install mship
```

| Command | Node role |
|---|---|
| `mship deploy --cuda --config models.yaml` | NVIDIA GPU node (vLLM, Diffusers, llama.cpp GPU offload) |
| `mship deploy --cpu --config models.yaml` | CPU node (vLLM CPU, llama.cpp, whisper.cpp, sherpa-onnx, stable-diffusion.cpp) |
| `mship deploy --metal --config models.yaml` | Apple Silicon (Metal offload) |
| `mship deploy --thin --config models.yaml` | Coordinator/head only — serves nothing itself |

`mship deploy` with no variant is an error that lists these options; there is no
default and no auto-detection, so a node's role is always something you stated.
`MSHIP_VARIANT` is the environment-variable equivalent, for systemd units and CI.

### What the first run does

The `mship` package is a small installer that runs on any Python 3.10+. The first
time you use a variant it:

1. Refuses unsupported platforms (Windows, musl/Alpine) before downloading anything.
2. Checks the hardware matches — `--cuda` needs `nvidia-smi` to list a device — so a
   driverless box fails immediately instead of after several GB.
3. Provisions `~/.modelship/envs/<variant>/` on **CPython 3.12.10**, installing from a
   hash-pinned dependency list shipped inside the package.
4. Fetches the pinned `llama-server` build for the platform.
5. Runs the engine inside that environment.

Every node therefore lands on an identical interpreter and dependency set, which is
what lets a native node join a cluster of Docker nodes — Ray refuses to form a
cluster across mismatched Python versions. Later runs re-verify the environment in
milliseconds and start straight away.

A copy of the exact pinned list that built each environment is left at
`~/.modelship/envs/<variant>/pins.txt`. `mship info` reports what is provisioned.

### Platform prerequisites

These are not installed for you.

```bash
# macOS — compiles stable-diffusion-cpp-python on first install
xcode-select --install

# Linux, all variants
sudo apt-get install -y build-essential cmake

# Linux --cpu, additionally (lscpu feeds vLLM's CPU NUMA detection; its absence
# surfaces as an opaque `Engine core initialization failed`)
sudo apt-get install -y libnuma1 openssl util-linux

# Linux --cuda, additionally: flashinfer JIT-compiles kernels at model-load time
sudo apt-get install -y ninja-build gnupg
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
  | sudo gpg --dearmor -o /usr/share/keyrings/cuda-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
  | sudo tee /etc/apt/sources.list.d/cuda.list
sudo apt-get update && sudo apt-get install -y \
  cuda-nvcc-13-0 cuda-cuobjdump-13-0 libcurand-dev-13-0
```

`nvcc` does not need to be on `PATH` — those packages create the `/usr/local/cuda`
symlink flashinfer looks for. The first vLLM deploy is slow while kernels compile;
they are cached per GPU architecture under `~/.modelship/cache/flashinfer`.

**Loader coverage on `--cuda`.** `vllm`, `diffusers`, and `llama_server` all get full
GPU. `llama_server` gets a CUDA ggml backend beside the same binary every other
platform runs. ggml skips a backend it cannot load without saying so, so if a GGUF
deploy seems slow, check `"$MSHIP_LLAMA_SERVER_BIN" --list-devices` — it prints
`(none)` instead of a `CUDA0:` line. `stable_diffusion_cpp`, `whispercpp`, and
`sherpa_onnx` are CPU-only here, same as in the image.

### Files on disk

```
~/.modelship/
  cache/                    models and other downloads (MSHIP_CACHE_DIR)
  envs/<variant>/           one environment per variant
  builds/<variant>/         llama-server binaries
  bin/uv                    only if uv was not already installed
```

`MSHIP_CACHE_DIR` may point at shared storage — model weights are identical on every
node. `MSHIP_HOME` (default `~/.modelship`) must stay node-local: environments and
binaries are platform- and variant-specific. To reset a variant, delete
`~/.modelship/envs/<variant>/`; the next run rebuilds it and leaves your models alone.

## Local development

For building from source, running inside the dev container, or a manual
`uv sync` + `mship_deploy.py` workflow (including the full CLI/env var
reference and port list), see [Development Setup](development.md).

## Scaling beyond one node

To join multiple hosts into one Ray cluster with plain `docker run` (no
Kubernetes), see [Multi-node without Kubernetes](multi-node-docker.md). For a
Kubernetes/KubeRay deployment, see the
[Helm chart](https://github.com/modelship-ai/modelship/tree/main/helm/modelship).
