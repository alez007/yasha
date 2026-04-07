ARG CUDA_VERSION=13.0.2
ARG PYTHON_VERSION=3.12.10

FROM ubuntu:24.04 AS base

ARG CUDA_VERSION
ARG PYTHON_VERSION

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
        | gpg --dearmor -o /usr/share/keyrings/cuda-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
        > /etc/apt/sources.list.d/cuda.list

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    espeak-ng \
    git

RUN CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') && CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f1) && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cuda-nvcc-${CUDA_VERSION_DASH} \
        cuda-cudart-${CUDA_VERSION_DASH} \
        cuda-nvrtc-${CUDA_VERSION_DASH} \
        cuda-cuobjdump-${CUDA_VERSION_DASH} \
        libcurand-dev-${CUDA_VERSION_DASH} \
        libcublas-${CUDA_VERSION_DASH} \
        cudnn9-cuda-${CUDA_MAJOR_VERSION}

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy

WORKDIR /yasha

ADD ./pyproject.toml pyproject.toml
ADD ./README.md README.md
ADD ./uv.lock uv.lock
ADD ./plugins plugins

ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV VIRTUAL_ENV=/.venv
ENV YASHA_CACHE_DIR=/yasha/.cache/models
ENV RAY_REDIS_PORT=6379
ENV RAY_CLUSTER_ADDRESS=0.0.0.0
ENV RAY_HEAD_CPU_NUM=2
ENV RAY_HEAD_GPU_NUM=1
ENV YASHA_USE_EXISTING_RAY_CLUSTER=false
ENV YASHA_METRICS=true
ENV RAY_METRICS_EXPORT_PORT=8079
RUN uv venv

ARG PYTHON_VERSION
RUN uv python install ${PYTHON_VERSION}

ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

# ---------------------------------------------------------------------------
# Development target
# ---------------------------------------------------------------------------
FROM base AS dev

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --extra dev

ADD ./scripts/start_ray.sh /yasha/scripts/start_ray.sh
RUN chmod +x /yasha/scripts/start_ray.sh

CMD ["/bin/bash"]

# ---------------------------------------------------------------------------
# Production target
# ---------------------------------------------------------------------------
FROM base AS prod

ADD ./start.py start.py
ADD ./yasha yasha
ADD ./config config
ADD ./scripts scripts

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

RUN chmod +x /yasha/scripts/start_ray.sh /yasha/scripts/start.sh

CMD ["uv", "run", "--active", "bash", "/yasha/scripts/start.sh"]
