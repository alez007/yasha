import hashlib
import os
import tarfile
from unittest.mock import patch

import pytest

from modelship.provision.llama_server import LlamaServerProvisionError, resolve_llama_server_bin


def _make_archive(tmp_path, contents: bytes = b"binary-contents") -> tuple[str, str]:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "llama-server").write_bytes(contents)
    (src_dir / "libggml.dylib").write_bytes(b"lib")
    archive = tmp_path / "archive.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(src_dir / "llama-server", arcname="llama-server")
        tar.add(src_dir / "libggml.dylib", arcname="libggml.dylib")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return str(archive), digest


class TestResolveLlamaServerBin:
    def test_explicit_env_var_short_circuits(self):
        with patch.dict(os.environ, {"MSHIP_LLAMA_SERVER_BIN": "/custom/llama-server.sh"}, clear=True):
            assert resolve_llama_server_bin() == "/custom/llama-server.sh"

    def test_non_darwin_raises(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("modelship.provision.llama_server.platform.system", return_value="Linux"),
            pytest.raises(LlamaServerProvisionError, match="MSHIP_LLAMA_SERVER_BIN"),
        ):
            resolve_llama_server_bin()

    def test_downloads_verifies_and_extracts(self, tmp_path):
        archive, digest = _make_archive(tmp_path)
        cache_root = str(tmp_path / "cache")

        def fake_download(url, dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(archive, "rb") as src, open(dest, "wb") as dst:
                dst.write(src.read())

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("modelship.provision.llama_server.platform.system", return_value="Darwin"),
            patch("modelship.provision.llama_server.resolve_cache_root", return_value=cache_root),
            patch("modelship.provision.llama_server.LLAMA_CPP_METAL_SHA256", digest),
            patch("modelship.provision.llama_server._download", side_effect=fake_download) as mock_download,
        ):
            wrapper = resolve_llama_server_bin()

        mock_download.assert_called_once()
        assert os.path.isfile(wrapper)
        assert "llama-server.sh" in wrapper
        extracted_bin = os.path.join(os.path.dirname(wrapper), "extracted", "llama-server")
        assert os.path.isfile(extracted_bin)

    def test_cached_archive_reused_without_redownload(self, tmp_path):
        archive, digest = _make_archive(tmp_path)
        cache_root = str(tmp_path / "cache")

        def fake_download(url, dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(archive, "rb") as src, open(dest, "wb") as dst:
                dst.write(src.read())

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("modelship.provision.llama_server.platform.system", return_value="Darwin"),
            patch("modelship.provision.llama_server.resolve_cache_root", return_value=cache_root),
            patch("modelship.provision.llama_server.LLAMA_CPP_METAL_SHA256", digest),
            patch("modelship.provision.llama_server._download", side_effect=fake_download) as mock_download,
        ):
            resolve_llama_server_bin()
            resolve_llama_server_bin()

        mock_download.assert_called_once()

    def test_digest_mismatch_raises(self, tmp_path):
        archive, _real_digest = _make_archive(tmp_path)
        cache_root = str(tmp_path / "cache")

        def fake_download(url, dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(archive, "rb") as src, open(dest, "wb") as dst:
                dst.write(src.read())

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("modelship.provision.llama_server.platform.system", return_value="Darwin"),
            patch("modelship.provision.llama_server.resolve_cache_root", return_value=cache_root),
            patch("modelship.provision.llama_server.LLAMA_CPP_METAL_SHA256", "0" * 64),
            patch("modelship.provision.llama_server._download", side_effect=fake_download),
            pytest.raises(LlamaServerProvisionError, match="sha256 verification"),
        ):
            resolve_llama_server_bin()
