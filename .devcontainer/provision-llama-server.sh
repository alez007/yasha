#!/usr/bin/env bash
# Fetches modelship's pinned llama-server build via bootstrap/mship_bootstrap/llama_cpp.py
# (stdlib-only, imported straight from source), since the `dev` Dockerfile target never
# runs `mship bootstrap`. Symlinks /.venv into the envs/<variant>/.venv layout that module
# expects, so it finds torch's bundled CUDA libs for libggml-cuda.so.
set -euo pipefail

# Must be unset: llama_cpp.provision() short-circuits on this env var, and
# remoteEnv points it at this script's own output.
unset MSHIP_LLAMA_SERVER_BIN

MSHIP_HOME="${MSHIP_HOME:-/opt/mship}"
mkdir -p "$MSHIP_HOME/envs/cuda"
ln -sfn /.venv "$MSHIP_HOME/envs/cuda/.venv"

# provision() also prints progress to stdout, so capture the path via a file
# instead of $(...) to avoid mixing the two.
OUT_FILE=$(mktemp)
trap 'rm -f "$OUT_FILE"' EXIT

python3 - "$OUT_FILE" <<'PY'
import sys

out_file = sys.argv[1]
sys.path.insert(0, "/modelship/bootstrap")
from mship_bootstrap import llama_cpp
from mship_bootstrap.variants import VARIANTS

wrapper = llama_cpp.provision(VARIANTS["cuda"])
if wrapper is None:
    sys.exit("error: llama-server provisioning failed (see warning above)")
with open(out_file, "w") as f:
    f.write(wrapper)
PY

WRAPPER=$(cat "$OUT_FILE")

# Stable path, so MSHIP_LLAMA_SERVER_BIN doesn't need updating on a tag bump.
STABLE_LINK="$MSHIP_HOME/bin/llama-server.sh"
mkdir -p "$(dirname "$STABLE_LINK")"
ln -sfn "$WRAPPER" "$STABLE_LINK"
echo "llama-server ready: $STABLE_LINK -> $WRAPPER"
