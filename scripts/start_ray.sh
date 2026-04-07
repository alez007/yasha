#!/bin/bash
set -e

usage() {
    echo "Usage: start_ray.sh --num-cpus <n> --num-gpus <n> [--enable-metrics <true|false>]"
    exit 1
}

ENABLE_METRICS="true"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-cpus) NUM_CPUS="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --enable-metrics) ENABLE_METRICS="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[ -z "${NUM_CPUS}" ] && usage
[ -z "${NUM_GPUS}" ] && usage

METRICS_FLAG=""
if [ "${ENABLE_METRICS}" = "true" ]; then
    METRICS_FLAG="--metrics-export-port=${RAY_METRICS_EXPORT_PORT:-8079}"
fi

ray start --head \
    --dashboard-host=0.0.0.0 \
    --num-cpus="${NUM_CPUS}" \
    --num-gpus="${NUM_GPUS}" \
    --disable-usage-stats \
    ${METRICS_FLAG}

if ! ray status; then
    echo "ray cluster failed to start"
    exit 1
fi
