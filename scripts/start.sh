#!/bin/bash
set -e

# When MSHIP_SKIP_SYNC=true (the default in prebuilt images where the venv is
# already fully resolved by the builder stage), skip the runtime `uv sync`.
# This keeps startup fast AND preserves every pre-baked plugin, so users can
# reference any plugin in models.yaml without having to set MSHIP_PLUGINS.
# Set MSHIP_SKIP_SYNC=false to force a re-sync (useful when running from a
# bind-mounted source tree during development or when adding a plugin extra
# on a non-prebuilt image).
if [ "${MSHIP_SKIP_SYNC:-true}" != "true" ]; then
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
fi

if [ "${MSHIP_USE_EXISTING_RAY_CLUSTER}" != "true" ]; then
    /modelship/scripts/start_ray.sh --num-cpus "${RAY_HEAD_CPU_NUM}" --num-gpus "${RAY_HEAD_GPU_NUM}"
fi

cd /modelship && uv run --no-sync start.py
