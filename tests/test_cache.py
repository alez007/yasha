import os
from unittest import mock

from modelship.actor_options import build_cache_env_vars
from modelship.utils import cache_dir, plugins_dir


def test_build_cache_env_vars_defaults():
    with mock.patch.dict(os.environ, {}, clear=True):
        env_vars = build_cache_env_vars()
        assert env_vars["HF_HOME"] == "/.cache/huggingface"
        assert env_vars["VLLM_CACHE_ROOT"] == "/.cache/vllm"
        assert env_vars["FLASHINFER_CACHE_DIR"] == "/.cache/flashinfer"


def test_build_cache_env_vars_custom_dir():
    custom_dir = "/tmp/custom_cache"
    with mock.patch.dict(os.environ, {"MSHIP_CACHE_DIR": custom_dir}, clear=True):
        env_vars = build_cache_env_vars()
        assert env_vars["HF_HOME"] == f"{custom_dir}/huggingface"
        assert env_vars["VLLM_CACHE_ROOT"] == f"{custom_dir}/vllm"
        assert env_vars["FLASHINFER_CACHE_DIR"] == f"{custom_dir}/flashinfer"


def test_utils_cache_dir_default():
    # We don't want to actually create directories in the test environment if we can avoid it,
    # but cache_dir calls os.makedirs.
    with mock.patch.dict(os.environ, {}, clear=True), mock.patch("os.makedirs"):
        assert cache_dir() == "/.cache"


def test_utils_plugins_dir():
    with mock.patch.dict(os.environ, {"MSHIP_CACHE_DIR": "/tmp/cache"}, clear=True), mock.patch("os.makedirs"):
        assert plugins_dir() == "/tmp/cache/plugins"
