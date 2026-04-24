#!/bin/bash
set -e

if [ -n "${MSHIP_PLUGINS}" ]; then
    echo "warning: MSHIP_PLUGINS is deprecated and ignored; plugins referenced in models.yaml are loaded from ${MSHIP_PLUGIN_WHEEL_DIR:-<unset>} via Ray runtime_env." >&2
fi

if [ "${MSHIP_USE_EXISTING_RAY_CLUSTER}" != "true" ]; then
    /modelship/scripts/start_ray.sh --num-cpus "${RAY_HEAD_CPU_NUM}" --num-gpus "${RAY_HEAD_GPU_NUM}"
fi

cd /modelship && uv run --no-sync start.py
