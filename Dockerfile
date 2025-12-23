FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04 AS base

RUN CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') && CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f1) && \
    apt update -y && \
    apt install -y --no-install-recommends \
        build-essential \
        cuda-nvcc-${CUDA_VERSION_DASH} \
        cuda-cudart-${CUDA_VERSION_DASH} \
        cuda-nvrtc-${CUDA_VERSION_DASH} \
        cuda-cuobjdump-${CUDA_VERSION_DASH} \
        libcurand-dev-${CUDA_VERSION_DASH} \
        libcublas-${CUDA_VERSION_DASH}

# Verify GCC version
RUN echo $(gcc --version)

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy

WORKDIR /yasha

ADD ./pyproject.toml pyproject.toml
ADD ./README.md README.md
ADD ./uv.lock uv.lock


RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# just in case we decide to copy the project here
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --locked

ENV PATH="/yasha/.venv/bin:$PATH"

WORKDIR /yasha
COPY <<EOF start.sh
#!/bin/bash

uv sync --locked

ray start --head --dashboard-port=\${RAY_DASHBOARD_PORT} --port=\${RAY_REDIS_PORT} --dashboard-host=0.0.0.0 --num-cpus=8 --num-gpus=1 --disable-usage-stats
if ray status --address=0.0.0.0:\${RAY_REDIS_PORT}; then
    uv run --active start.py
else
    echo "ray cluster failed to start"
fi

EOF
RUN chmod +x start.sh

WORKDIR /yasha
CMD ["uv", "run", "bash", "start.sh"]





