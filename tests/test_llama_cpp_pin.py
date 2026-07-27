"""launcher.py's pinned llama.cpp tag must not drift from the Dockerfile's image tags."""

from pathlib import Path

from modelship import launcher

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_tag_matches_dockerfile_llama_cpp_images():
    dockerfile = (_REPO_ROOT / "Dockerfile").read_text()
    assert f"server-cuda13-{launcher._LLAMA_CPP_TAG}" in dockerfile
    assert f"server-{launcher._LLAMA_CPP_TAG}@sha256" in dockerfile


def test_asset_url_embeds_pinned_tag():
    assert launcher._LLAMA_CPP_TAG in launcher._LLAMA_CPP_METAL_ASSET_URL
    assert launcher._LLAMA_CPP_METAL_ASSET_URL.startswith("https://github.com/")


def test_workflow_bumps_launcher_constants():
    workflow = (_REPO_ROOT / ".github" / "workflows" / "llama-cpp-metal.yml").read_text()
    assert "modelship/launcher.py" in workflow
    assert "_LLAMA_CPP_TAG" in workflow
