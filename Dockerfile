ARG CUDA_VERSION=13.0.2
ARG PYTHON_VERSION=3.12.10

FROM ubuntu:24.04 AS base

ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG UID=1000
ARG GID=1000

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

# Create non-root user matching host UID/GID
# If a group with the target GID already exists (e.g. ubuntu:1000), reuse it;
# if a user with the target UID already exists, reuse and rename it.
RUN if ! getent group $GID >/dev/null; then groupadd -g $GID modelship; fi && \
    if ! getent passwd $UID >/dev/null; then useradd -m -u $UID -g $GID modelship; \
    else existing=$(getent passwd $UID | cut -d: -f1) && usermod -l modelship -d /home/modelship -m "$existing"; fi

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy

WORKDIR /modelship

ADD ./pyproject.toml pyproject.toml
ADD ./README.md README.md
ADD ./uv.lock uv.lock
ADD ./plugins plugins

ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV VIRTUAL_ENV=/.venv
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV MSHIP_CACHE_DIR=/modelship/.cache/models
ENV RAY_REDIS_PORT=6379
ENV RAY_CLUSTER_ADDRESS=0.0.0.0
ENV RAY_HEAD_CPU_NUM=2
ENV RAY_HEAD_GPU_NUM=1
ENV MSHIP_USE_EXISTING_RAY_CLUSTER=false
ENV MSHIP_METRICS=true
ENV RAY_METRICS_EXPORT_PORT=8079
ENV MSHIP_LOG_LEVEL=INFO
ENV MSHIP_LOG_FORMAT=text
ARG PYTHON_VERSION
ENV UV_PYTHON_INSTALL_DIR=/usr/local/uv/python
RUN uv python install ${PYTHON_VERSION}
RUN uv venv

ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

RUN chown -R $UID:$GID /modelship $UV_PROJECT_ENVIRONMENT

# ---------------------------------------------------------------------------
# Development target
# ---------------------------------------------------------------------------
FROM base AS dev

USER modelship

RUN --mount=type=cache,target=/home/modelship/.cache/uv,uid=$UID,gid=$GID \
    uv sync --locked --no-install-project --extra dev

ADD --chown=$UID:$GID ./scripts/start_ray.sh /modelship/scripts/start_ray.sh

CMD ["/bin/bash"]

# ---------------------------------------------------------------------------
# Production target
# ---------------------------------------------------------------------------
FROM base AS prod

ADD --chown=$UID:$GID ./start.py start.py
ADD --chown=$UID:$GID ./modelship modelship
ADD --chown=$UID:$GID ./scripts scripts

USER modelship

RUN --mount=type=cache,target=/home/modelship/.cache/uv,uid=$UID,gid=$GID \
    uv sync --locked --no-install-project

CMD ["uv", "run", "--active", "bash", "/modelship/scripts/start.sh"]
