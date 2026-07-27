"""llama.cpp version pinned for the macOS Metal build.

Shared by .github/workflows/llama-cpp-metal.yml (what to build) and
modelship/provision/llama_server.py (what to download/verify). Also matches
the Linux image tag in Dockerfile's LLAMA_CPP_IMAGE_CUDA/CPU args — bump all
three together.
"""

LLAMA_CPP_TAG = "b9859"

LLAMA_CPP_METAL_ASSET_URL = (
    f"https://github.com/alez007/modelship/releases/download/llamacpp-{LLAMA_CPP_TAG}-metal/"
    f"llama-server-{LLAMA_CPP_TAG}-macos-arm64-metal.tar.gz"
)

# Filled in after .github/workflows/llama-cpp-metal.yml publishes the release
# for LLAMA_CPP_TAG. The resolver hard-fails on a mismatch, so this must never
# drift from the actually-published asset.
LLAMA_CPP_METAL_SHA256 = ""
