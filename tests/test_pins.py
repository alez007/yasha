"""modelship/_pins.py must not drift from the Dockerfile's llama.cpp image tags."""

from pathlib import Path

from modelship import _pins

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_tag_matches_dockerfile_llama_cpp_images():
    dockerfile = (_REPO_ROOT / "Dockerfile").read_text()
    assert f"server-cuda13-{_pins.LLAMA_CPP_TAG}" in dockerfile
    assert f"server-{_pins.LLAMA_CPP_TAG}@sha256" in dockerfile


def test_asset_url_embeds_pinned_tag():
    assert _pins.LLAMA_CPP_TAG in _pins.LLAMA_CPP_METAL_ASSET_URL
    assert _pins.LLAMA_CPP_METAL_ASSET_URL.startswith("https://github.com/")


def test_workflow_reads_pin_from_pins_module():
    workflow = (_REPO_ROOT / ".github" / "workflows" / "llama-cpp-metal.yml").read_text()
    assert "_pins.LLAMA_CPP_TAG" in workflow
