#!/bin/bash
set -e

EXTRAS=""
if [ -n "${YASHA_PLUGINS}" ]; then
    for plugin in $(echo "${YASHA_PLUGINS}" | tr ',' ' '); do
        EXTRAS="$EXTRAS --extra $plugin"
    done
fi
uv sync --project /yasha --locked $EXTRAS

if [ "${YASHA_USE_EXISTING_RAY_CLUSTER}" != "true" ]; then
    /yasha/scripts/start_ray.sh --num-cpus "${RAY_HEAD_CPU_NUM}" --num-gpus "${RAY_HEAD_GPU_NUM}"
fi

cd /yasha && uv run start.py
