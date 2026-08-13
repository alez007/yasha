ARG CUDA_VERSION=13.0.2
ARG PYTHON_VERSION=3.12.10
ARG MSHIP_VARIANT=cuda
ARG UID=1000
ARG GID=1000

# Own-CI llama.cpp builds; rewritten by llama-cpp-build.yml's pin job.
ARG LLAMA_CPP_TAG=b10375
ARG LLAMA_CPP_SHA256_LINUX_X64=64625921d1257485a82cc7eee6de58075d5f81a1b588e3e2817cf9632ffc8090
ARG LLAMA_CPP_SHA256_LINUX_ARM64=122186a168c10c9510b6e43c670515206d3a4ca7f5c10ef9fa4708fbea77a9de
ARG LLAMA_CPP_SHA256_CUDA_X64=693d45d45b42902a2746f89e51e7caa62bffa22059673db0255c5b029755256a
ARG LLAMA_CPP_BUILDS_REPO=modelship-ai/llama-cpp-builds

# =============================================================================
# llama-server — assembles /opt/llama.cpp for the llama_server loader.
#
# The tarball holds llama-server plus the .so backends it dlopen()s
# (GGML_BACKEND_DL), which resolve each other via the binary's $ORIGIN runpath.
# libggml-cuda.so drops in over the same CPU tarball; it links
# libcudart/libcublas, resolved from the venv's torch-bundled copies through
# the LD_LIBRARY_PATH set below, plus the driver's libcuda.so.1 from the NVIDIA
# Container Toolkit. A CUDA backend that fails to load is skipped silently.
# =============================================================================
FROM ubuntu:24.04 AS llama-server-download

ARG LLAMA_CPP_TAG
ARG LLAMA_CPP_SHA256_LINUX_X64
ARG LLAMA_CPP_SHA256_LINUX_ARM64
ARG LLAMA_CPP_BUILDS_REPO
ARG TARGETARCH

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

RUN set -e; \
    case "$TARGETARCH" in \
        amd64) slug=linux-x64; sha=$LLAMA_CPP_SHA256_LINUX_X64 ;; \
        arm64) slug=linux-arm64; sha=$LLAMA_CPP_SHA256_LINUX_ARM64 ;; \
        *) echo "no llama.cpp build for TARGETARCH=$TARGETARCH" >&2; exit 1 ;; \
    esac; \
    name="llama-server-${LLAMA_CPP_TAG}-${slug}"; \
    curl -fsSL -o /tmp/llama.tar.gz \
        "https://github.com/${LLAMA_CPP_BUILDS_REPO}/releases/download/llamacpp-${LLAMA_CPP_TAG}/${name}.tar.gz"; \
    echo "${sha}  /tmp/llama.tar.gz" | sha256sum -c -; \
    mkdir -p /opt/llama.cpp; \
    tar xzf /tmp/llama.tar.gz -C /opt/llama.cpp --strip-components=1; \
    rm /tmp/llama.tar.gz

FROM llama-server-download AS llama-server-cuda

ARG LLAMA_CPP_TAG
ARG LLAMA_CPP_SHA256_CUDA_X64
ARG LLAMA_CPP_BUILDS_REPO

RUN set -e; \
    name="libggml-cuda-${LLAMA_CPP_TAG}-linux-x64-cuda13"; \
    curl -fsSL -o /tmp/cuda.tar.gz \
        "https://github.com/${LLAMA_CPP_BUILDS_REPO}/releases/download/llamacpp-${LLAMA_CPP_TAG}/${name}.tar.gz"; \
    echo "${LLAMA_CPP_SHA256_CUDA_X64}  /tmp/cuda.tar.gz" | sha256sum -c -; \
    tar xzf /tmp/cuda.tar.gz -C /opt/llama.cpp --strip-components=1; \
    rm /tmp/cuda.tar.gz; \
    test -f /opt/llama.cpp/libggml-cuda.so

FROM llama-server-download AS llama-server-cpu

# thin ships no llama-server binary — it never loads a model. The wrapper
# script + MSHIP_LLAMA_SERVER_BIN below still get written pointing at this
# empty dir; harmless, since thin never invokes it.
FROM ubuntu:24.04 AS llama-server-thin

RUN mkdir -p /opt/llama.cpp

# MSHIP_LLAMA_SERVER_BIN points at a wrapper in both the image and the native
# install (launcher._write_wrapper); sibling .so files resolve via $ORIGIN.
FROM llama-server-${MSHIP_VARIANT} AS llama-server

RUN printf '#!/bin/sh\nexport LD_LIBRARY_PATH="/opt/llama.cpp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\nexec /opt/llama.cpp/llama-server "$@"\n' \
        > /opt/llama.cpp/llama-server.sh && \
    chmod +x /opt/llama.cpp/llama-server.sh

# =============================================================================
# base — minimal runtime OS + uv + non-root user + env vars.
#
# CUDA strategy: torch cu130 bundles libcublas/libcudnn/libcurand/libnccl/
# libnvrtc inside the venv (under site-packages/nvidia/*/lib) and the NVIDIA
# Container Toolkit provides libcuda.so at run time via --gpus. However, vLLM's
# C extensions (_C.abi3.so, _moe_C.abi3.so, ...) are built with an RPATH that
# hard-references /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.13.
# Without that file, the vLLM registry subprocess that runs before torch has
# bootstrapped its dlopen paths crashes with malloc_consolidate/SIGABRT while
# the dynamic loader resolves symbols. We therefore install ONLY the tiny
# cuda-cudart runtime package (~800 KB) in the base image. libcublas/cudnn/
# curand/nvrtc are NOT installed — torch's bundled copies are resolved via its
# own rpath once Python imports torch, so adding system duplicates is wasted
# space.
# =============================================================================
FROM ubuntu:24.04 AS base

ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG MSHIP_VARIANT
ARG UID
ARG GID

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        espeak-ng \
        gcc \
        g++ \
        gnupg \
        gosu \
        libc6-dev \
        libgomp1 \
        libnuma1 \
        ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Register the NVIDIA CUDA apt repo and install cuda-cudart, cuda-nvcc, and
# cuda-cuobjdump (cuda variant only). gcc/g++ + libc6-dev and ninja-build stay
# because torch/triton and flashinfer JIT-compile kernels at model-load time
# and shell out to $CC/nvcc; without them, vllm crashes in _inductor (needs
# g++ for its CPU codegen backend, e.g. the vllm CPU loader's torch.compile
# path) or flashinfer on newer architectures (such as Blackwell).
RUN if [ "$MSHIP_VARIANT" = "cuda" ]; then \
    CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
        | gpg --dearmor -o /usr/share/keyrings/cuda-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
        > /etc/apt/sources.list.d/cuda.list && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        cuda-cudart-${CUDA_VERSION_DASH} \
        cuda-nvcc-${CUDA_VERSION_DASH} \
        cuda-cuobjdump-${CUDA_VERSION_DASH} \
        libcurand-dev-${CUDA_VERSION_DASH} && \
    apt-get purge -y --auto-remove gnupg && \
    rm -f /etc/apt/sources.list.d/cuda.list /usr/share/keyrings/cuda-keyring.gpg && \
    rm -rf /var/lib/apt/lists/*; \
    fi

RUN if ! getent group $GID >/dev/null; then groupadd -g $GID modelship; fi && \
    if ! getent passwd $UID >/dev/null; then useradd -m -u $UID -g $GID modelship; \
    else existing=$(getent passwd $UID | cut -d: -f1) && usermod -l modelship -d /home/modelship -m "$existing"; fi

ENV MSHIP_UID=$UID
ENV MSHIP_GID=$GID

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy

WORKDIR /modelship

ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV VIRTUAL_ENV=/.venv
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV MSHIP_CACHE_DIR=/.cache
ENV UV_CACHE_DIR=${MSHIP_CACHE_DIR}/uv
ENV MSHIP_USE_EXISTING_RAY_CLUSTER=false
ENV MSHIP_METRICS=true
ENV RAY_METRICS_EXPORT_PORT=8079
ENV MSHIP_LOG_LEVEL=INFO
ENV MSHIP_LOG_FORMAT=text
ENV UV_PYTHON_INSTALL_DIR=/usr/local/uv/python
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

# Pinned llama.cpp build for the llama_server loader (see the llama-server
# stages above). llama-server additionally needs libgomp1 (ggml CPU backends)
# and libssl3 (already pulled in via curl).
COPY --from=llama-server /opt/llama.cpp /opt/llama.cpp
ENV MSHIP_LLAMA_SERVER_BIN=/opt/llama.cpp/llama-server.sh

# onnxruntime-gpu (a direct dependency of the cuda/metal extras) dlopen()s
# libonnxruntime_providers_cuda.so which has plain DT_NEEDED entries for
# libcublasLt.so.13 / libcudnn.so.9 / etc. Torch cu130 bundles these under
# site-packages/nvidia/*/lib and resolves them via its own rpath once imported
# — but onnxruntime doesn't participate in that. Expose the torch-bundled
# CUDA libs on LD_LIBRARY_PATH so onnxruntime's CUDA provider can load.
# Python version is pinned via PYTHON_VERSION (see pyproject.toml); we hard-
# code 3.12 here because UV_PROJECT_ENVIRONMENT is fixed and the ENV cannot
# shell-evaluate.
ENV LD_LIBRARY_PATH="/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:/.venv/lib/python3.12/site-packages/nvidia/nvshmem/lib"

RUN mkdir -p /.cache /.venv /usr/local/uv/python && \
    chown -R $UID:$GID /modelship /.cache /.venv /usr/local/uv/python

# =============================================================================
# builder — adds build toolchain (nvcc, build-essential, dev headers, git) and
# re-registers the NVIDIA apt repo so we can pull nvcc / dev headers needed to
# compile wheels from source (flashinfer, llama-cpp-python, etc.). All of this
# stays in the builder stage and is NOT copied into prod.
#
# The venv is resolved with --extra $MSHIP_VARIANT, plus --extra vllm-cpu on cpu.
# =============================================================================
FROM base AS builder

ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG MSHIP_VARIANT
ARG UID
ARG GID

RUN if [ "$MSHIP_VARIANT" = "cuda" ]; then \
    apt-get update -y && \
    apt-get install -y --no-install-recommends gnupg && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
        | gpg --dearmor -o /usr/share/keyrings/cuda-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
        > /etc/apt/sources.list.d/cuda.list && \
    rm -rf /var/lib/apt/lists/*; \
    fi

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git && \
    if [ "$MSHIP_VARIANT" = "cuda" ]; then \
    CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') && \
    apt-get install -y --no-install-recommends \
        cuda-nvcc-${CUDA_VERSION_DASH} \
        cuda-cuobjdump-${CUDA_VERSION_DASH} \
        libcurand-dev-${CUDA_VERSION_DASH}; \
    fi && \
    rm -rf /var/lib/apt/lists/*

USER modelship

RUN --mount=type=cache,target=/.cache/uv,uid=$UID,gid=$GID \
    uv python install ${PYTHON_VERSION}
RUN --mount=type=cache,target=/.cache/uv,uid=$UID,gid=$GID \
    uv venv

ADD --chown=$UID:$GID ./pyproject.toml pyproject.toml
ADD --chown=$UID:$GID ./README.md README.md
ADD --chown=$UID:$GID ./uv.lock uv.lock
ADD --chown=$UID:$GID ./Makefile Makefile

RUN --mount=type=cache,target=/.cache/uv,uid=$UID,gid=$GID \
    if [ "$MSHIP_VARIANT" = "cpu" ]; then \
        uv sync --locked --no-install-project --extra $MSHIP_VARIANT --extra vllm-cpu; \
    else \
        uv sync --locked --no-install-project --extra $MSHIP_VARIANT; \
    fi

# =============================================================================
# dev — inherits builder (keeps toolchain) and adds dev extras so developers
# get editable installs for interactive REPL / pytest. Prod does not inherit
# this stage.
# =============================================================================
FROM builder AS dev

ARG MSHIP_VARIANT
ARG UID
ARG GID

USER modelship

RUN --mount=type=cache,target=/.cache/uv,uid=$UID,gid=$GID \
    if [ "$MSHIP_VARIANT" = "cpu" ]; then \
        uv sync --locked --no-install-project --extra dev --extra $MSHIP_VARIANT --extra vllm-cpu; \
    else \
        uv sync --locked --no-install-project --extra dev --extra $MSHIP_VARIANT; \
    fi

USER root

ENTRYPOINT ["/modelship/scripts/entrypoint.sh"]

# =============================================================================
# prod — minimal runtime. No build tools. Copies the resolved venv and
# Python interpreter from builder.
# =============================================================================
FROM base AS prod

ARG UID
ARG GID

COPY --from=builder --chown=$UID:$GID /usr/local/uv/python /usr/local/uv/python
COPY --from=builder --chown=$UID:$GID /.venv /.venv

ADD --chown=$UID:$GID ./pyproject.toml pyproject.toml
ADD --chown=$UID:$GID ./README.md README.md
ADD --chown=$UID:$GID ./uv.lock uv.lock
ADD --chown=$UID:$GID ./mship_deploy.py mship_deploy.py
ADD --chown=$UID:$GID ./modelship modelship
ADD --chown=$UID:$GID ./scripts scripts

USER root

ENTRYPOINT ["/modelship/scripts/entrypoint.sh", "--serve"]

# thin variant: pin capacity to 0 so it never advertises resources it can't
# serve (no torch/vllm). Layered on top since a shared stage can't
# conditionally set ENV.
FROM prod AS prod-thin

ENV MSHIP_NODE_NUM_CPUS=0
ENV MSHIP_NODE_NUM_GPUS=0
