#!/bin/bash
set -e

EXTRAS=""
if [ -n "${MSHIP_PLUGINS}" ]; then
    for plugin in $(echo "${MSHIP_PLUGINS}" | tr ',' ' '); do
        EXTRAS="$EXTRAS --extra $plugin"
    done
fi

# Detect if we should use GPU based on RAY_HEAD_GPU_NUM
if [ "${RAY_HEAD_GPU_NUM:-0}" -gt 0 ]; then
    EXTRAS="$EXTRAS --extra gpu"
else
    EXTRAS="$EXTRAS --extra cpu"
fi

uv sync --project /modelship --locked $EXTRAS

if [ "${MSHIP_USE_EXISTING_RAY_CLUSTER}" != "true" ]; then
    /modelship/scripts/start_ray.sh --num-cpus "${RAY_HEAD_CPU_NUM}" --num-gpus "${RAY_HEAD_GPU_NUM}"
fi

cd /modelship && uv run start.py
