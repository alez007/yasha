"""launcher.py's pinned llama.cpp tag must not drift from the Dockerfile's pins,
nor from what llama-cpp-build.yml actually builds."""

from pathlib import Path

import yaml

from modelship import launcher

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUILD_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "llama-cpp-build.yml"


def _build_workflow() -> dict:
    return yaml.safe_load(_BUILD_WORKFLOW.read_text())


def test_dockerfile_pins_match_launcher():
    """Both fetch the same tarballs, so a drift here means the image and a native
    node run different binaries."""
    dockerfile = (_REPO_ROOT / "Dockerfile").read_text()
    assert f"\nARG LLAMA_CPP_TAG={launcher._LLAMA_CPP_TAG}\n" in dockerfile
    assert f"\nARG LLAMA_CPP_SHA256_LINUX_X64={launcher._SHA256_LINUX_X64}\n" in dockerfile
    assert f"\nARG LLAMA_CPP_SHA256_LINUX_ARM64={launcher._SHA256_LINUX_ARM64}\n" in dockerfile
    assert f"\nARG LLAMA_CPP_SHA256_CUDA_X64={launcher._SHA256_CUDA_X64}\n" in dockerfile


def test_asset_url_embeds_pinned_tag():
    for asset in launcher._LLAMA_CPP_ASSETS.values():
        assert launcher._LLAMA_CPP_TAG in asset.url
        assert asset.url.startswith(f"https://github.com/{launcher._LLAMA_CPP_BUILDS_REPO}/")


def test_cuda_addon_comes_from_the_same_build():
    """The backend is dlopened beside the CPU tarball's libggml-base.so; a
    different tag is an ABI mismatch."""
    assert launcher._LLAMA_CPP_TAG in launcher._CUDA_ADDON_URL
    assert launcher._CUDA_ADDON_URL.startswith(f"https://github.com/{launcher._LLAMA_CPP_BUILDS_REPO}/")


def test_build_workflow_matrix_names():
    built = {m["name"] for m in _build_workflow()["jobs"]["server"]["strategy"]["matrix"]["include"]}
    assert built == {"linux-x64", "linux-arm64", "macos-arm64-metal"}


def test_makefile_triggers_the_build_workflow():
    makefile = (_REPO_ROOT / "Makefile").read_text()
    assert f"gh workflow run {_BUILD_WORKFLOW.name}" in makefile


def test_pin_job_rewrites_both_pinned_files():
    pin = _build_workflow()["jobs"]["pin"]
    assert pin["needs"] == "publish"
    steps = " ".join(step.get("run", "") for step in pin["steps"])
    assert "modelship/launcher.py" in steps
    assert "Dockerfile" in steps


def test_cuda_backend_shares_the_linux_x64_runner():
    """A newer runner image would raise the backend's glibc floor above the
    linux-x64 binaries it is dlopened beside."""
    workflow = _build_workflow()
    x64 = next(m for m in workflow["jobs"]["server"]["strategy"]["matrix"]["include"] if m["name"] == "linux-x64")
    assert workflow["jobs"]["cuda-backend"]["runs-on"] == x64["os"]
