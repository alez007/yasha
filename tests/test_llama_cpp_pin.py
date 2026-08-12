"""launcher.py's pinned llama.cpp tag must not drift from the Dockerfile's image
tags, nor from what llama-cpp-build.yml actually builds."""

from pathlib import Path

import yaml

from modelship import launcher

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUILD_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "llama-cpp-build.yml"


def _build_workflow() -> dict:
    return yaml.safe_load(_BUILD_WORKFLOW.read_text())


def test_tag_matches_dockerfile_llama_cpp_images():
    dockerfile = (_REPO_ROOT / "Dockerfile").read_text()
    assert f"server-cuda13-{launcher._LLAMA_CPP_TAG}" in dockerfile
    assert f"server-{launcher._LLAMA_CPP_TAG}@sha256" in dockerfile


def test_asset_url_embeds_pinned_tag():
    metal_url = launcher._LLAMA_CPP_ASSETS[("Darwin", "arm64")].url
    assert launcher._LLAMA_CPP_TAG in metal_url
    assert metal_url.startswith("https://github.com/")


def test_build_workflow_covers_every_asset_platform():
    built = {m["name"] for m in _build_workflow()["jobs"]["server"]["strategy"]["matrix"]["include"]}
    assert built == {"linux-x64", "linux-arm64", "macos-arm64-metal"}


def test_cuda_backend_shares_the_linux_x64_runner():
    """A newer runner image would raise the backend's glibc floor above the
    linux-x64 binaries it is dlopened beside."""
    workflow = _build_workflow()
    x64 = next(m for m in workflow["jobs"]["server"]["strategy"]["matrix"]["include"] if m["name"] == "linux-x64")
    assert workflow["jobs"]["cuda-backend"]["runs-on"] == x64["os"]
