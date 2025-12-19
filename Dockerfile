FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04 AS base

# install extra packages
RUN apt update && apt -y install -y --no-install-recommends build-essential retry

# Verify GCC version
RUN gcc --version

RUN CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') && CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f1) && \
    apt update -y && \
    apt install -y --no-install-recommends \
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


RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock,ro \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml,ro \
    uv sync --locked --no-install-project

ENV PATH="/yasha/.venv/bin:$PATH"

WORKDIR /
COPY <<EOF start.sh

ray start --head --dashboard-port=\${RAY_DASHBOARD_PORT} --port=\${RAY_REDIS_PORT} --dashboard-host=0.0.0.0 --num-cpus=8 --num-gpus=1 --disable-usage-stats
sleep 10
if ray status --address=0.0.0.0:\${RAY_REDIS_PORT}; then
    if serve start --http-host 0.0.0.0 --http-port 8000 --address=ray://0.0.0.0:\${RAY_HEAD_PORT}; then
        if timeout -k 30 20 serve status --address=http://0.0.0.0:\${RAY_DASHBOARD_PORT}; then
            echo "ray serve started"
        else
            echo "ray serve failed to start"
        fi
    fi
else
    echo "ray cluster failed to start"
fi

cd yasha && uv run start.py

# tail -f /dev/null
EOF
RUN chmod +x start.sh

CMD ["/bin/sh", "-c", "start.sh"]

# FROM base

# HEALTHCHECK --interval=5s --timeout=5s --retries=3 CMD ray status --address=0.0.0.0:${RAY_REDIS_PORT} || exit 1

# CMD serve start --http-host 0.0.0.0 --http-port 8000 --address=ray://0.0.0.0:${RAY_HEAD_PORT} && sleep 120





