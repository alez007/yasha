import hashlib
import io
import os
import tarfile
from unittest.mock import patch

import pytest

from mship_bootstrap import fetch


def _tarball(tmp_path, members: dict[str, bytes], *, prefix: str = "") -> tuple[str, str]:
    """Returns (path, sha256) of a tarball holding `members`."""
    path = os.path.join(tmp_path, "src.tar.gz")
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(prefix + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with open(path, "rb") as f:
        return path, hashlib.sha256(f.read()).hexdigest()


def _tarball_with(tmp_path, *infos: tarfile.TarInfo) -> tuple[str, str]:
    """Returns (path, sha256) of a tarball holding sizeless members verbatim."""
    path = os.path.join(tmp_path, "src.tar.gz")
    with tarfile.open(path, "w:gz") as tar:
        for info in infos:
            tar.addfile(info)
    with open(path, "rb") as f:
        return path, hashlib.sha256(f.read()).hexdigest()


def _member(name: str, tar_type: bytes, **attrs) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tar_type
    for key, value in attrs.items():
        setattr(info, key, value)
    return info


@pytest.fixture
def served(tmp_path):
    """Serves a local file through the urlopen call site, so no network is touched."""

    def _serve(source: str):
        def _urlopen(url, timeout=None):
            _serve.timeout = timeout
            return open(source, "rb")

        return patch("urllib.request.urlopen", side_effect=_urlopen)

    _serve.timeout = "not called"
    return _serve


class TestDownload:
    def test_urlopen_is_bounded(self, tmp_path, served):
        source, _ = _tarball(str(tmp_path), {"f": b"x"})
        with served(source):
            fetch.download("https://example.invalid/a.tar.gz", os.path.join(tmp_path, "out.tar.gz"))
        assert served.timeout == fetch._SOCKET_TIMEOUT_SECONDS

    def test_a_stalled_connection_raises_fetcherror(self, tmp_path):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with pytest.raises(fetch.FetchError):
                fetch.download("https://example.invalid/a.tar.gz", os.path.join(tmp_path, "out.tar.gz"))
        assert not os.listdir(tmp_path)

    def test_an_existing_file_is_not_refetched(self, tmp_path):
        target = os.path.join(tmp_path, "out.tar.gz")
        with open(target, "wb") as f:
            f.write(b"cached")
        with patch("urllib.request.urlopen", side_effect=AssertionError("should not fetch")):
            fetch.download("https://example.invalid/a.tar.gz", target)


class TestFetchAndExtract:
    def test_round_trip(self, tmp_path, served):
        source, digest = _tarball(str(tmp_path), {"llama-server": b"binary", "libggml.so": b"lib"})
        extract_dir = os.path.join(tmp_path, "extracted")
        with served(source):
            fetch.fetch_and_extract_archive(
                "https://example.invalid/a.tar.gz", digest, os.path.join(tmp_path, "a.tar.gz"), extract_dir
            )
        assert sorted(os.listdir(extract_dir)) == ["libggml.so", "llama-server"]

    def test_flatten_strips_the_archive_prefix(self, tmp_path, served):
        source, digest = _tarball(str(tmp_path), {"llama-server": b"binary"}, prefix="build/bin/")
        extract_dir = os.path.join(tmp_path, "extracted")
        with served(source):
            fetch.fetch_and_extract_archive(
                "https://example.invalid/a.tar.gz",
                digest,
                os.path.join(tmp_path, "a.tar.gz"),
                extract_dir,
                flatten=True,
            )
        assert os.listdir(extract_dir) == ["llama-server"]

    @pytest.mark.parametrize("name", ["../escaped", "/etc/escaped"])
    def test_a_member_outside_the_destination_is_refused(self, tmp_path, served, name):
        source, digest = _tarball(str(tmp_path), {name: b"x"})
        with served(source):
            with pytest.raises(fetch.FetchError, match="refusing to extract"):
                fetch.fetch_and_extract_archive(
                    "https://example.invalid/a.tar.gz",
                    digest,
                    os.path.join(tmp_path, "a.tar.gz"),
                    os.path.join(tmp_path, "extracted"),
                )

    @pytest.mark.parametrize("has_filter", [True, False])
    @pytest.mark.parametrize(
        "member",
        [
            pytest.param(_member("dev", tarfile.CHRTYPE, devmajor=1, devminor=3), id="device"),
            pytest.param(_member("pipe", tarfile.FIFOTYPE), id="fifo"),
        ],
    )
    def test_a_special_file_member_is_refused(self, tmp_path, served, member, has_filter):
        source, digest = _tarball_with(str(tmp_path), member)
        extract_dir = os.path.join(tmp_path, "extracted")
        with served(source), patch.object(fetch, "_HAS_EXTRACTION_FILTER", has_filter):
            with pytest.raises(fetch.FetchError, match="not a regular file"):
                fetch.fetch_and_extract_archive(
                    "https://example.invalid/a.tar.gz", digest, os.path.join(tmp_path, "a.tar.gz"), extract_dir
                )
        assert not os.path.exists(extract_dir)

    @pytest.mark.parametrize("has_filter", [True, False])
    @pytest.mark.parametrize("flatten", [True, False])
    @pytest.mark.parametrize(
        "member",
        [
            pytest.param(_member("bin/link", tarfile.SYMTYPE, linkname="../../etc/passwd"), id="symlink-relative"),
            pytest.param(_member("bin/link", tarfile.SYMTYPE, linkname="/etc/passwd"), id="symlink-absolute"),
            pytest.param(_member("bin/link", tarfile.LNKTYPE, linkname="../etc/passwd"), id="hardlink"),
        ],
    )
    def test_a_link_out_of_the_destination_is_refused(self, tmp_path, served, member, flatten, has_filter):
        """Flattening rewrites `name` but never `linkname`."""
        source, digest = _tarball_with(str(tmp_path), member)
        extract_dir = os.path.join(tmp_path, "extracted")
        with served(source), patch.object(fetch, "_HAS_EXTRACTION_FILTER", has_filter):
            with pytest.raises(fetch.FetchError, match="refusing to extract"):
                fetch.fetch_and_extract_archive(
                    "https://example.invalid/a.tar.gz",
                    digest,
                    os.path.join(tmp_path, "a.tar.gz"),
                    extract_dir,
                    flatten=flatten,
                )
        assert not os.path.exists(extract_dir)

    @pytest.mark.parametrize("has_filter", [True, False])
    def test_an_in_tree_symlink_is_extracted(self, tmp_path, served, has_filter):
        """Every llama-server asset ships SONAME symlinks the binary's rpath needs."""
        real = tarfile.TarInfo("bin/libggml.so.0")
        real.size = 3
        source = os.path.join(tmp_path, "src.tar.gz")
        with tarfile.open(source, "w:gz") as tar:
            tar.addfile(real, io.BytesIO(b"lib"))
            tar.addfile(_member("bin/libggml.so", tarfile.SYMTYPE, linkname="libggml.so.0"))
        with open(source, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()

        extract_dir = os.path.join(tmp_path, "extracted")
        with served(source), patch.object(fetch, "_HAS_EXTRACTION_FILTER", has_filter):
            fetch.fetch_and_extract_archive(
                "https://example.invalid/a.tar.gz", digest, os.path.join(tmp_path, "a.tar.gz"), extract_dir
            )
        link = os.path.join(extract_dir, "libggml.so")
        assert os.readlink(link) == "libggml.so.0"
        with open(link, "rb") as f:
            assert f.read() == b"lib"

    def test_without_the_filter_unsafe_modes_are_stripped(self, tmp_path, served):
        source, digest = _tarball_with(str(tmp_path), _member("llama-server", tarfile.REGTYPE, mode=0o4777))
        extract_dir = os.path.join(tmp_path, "extracted")
        with served(source), patch.object(fetch, "_HAS_EXTRACTION_FILTER", False):
            fetch.fetch_and_extract_archive(
                "https://example.invalid/a.tar.gz", digest, os.path.join(tmp_path, "a.tar.gz"), extract_dir
            )
        assert os.stat(os.path.join(extract_dir, "llama-server")).st_mode & 0o7777 == 0o755

    def test_a_bad_digest_drops_the_archive(self, tmp_path, served):
        """A retry must re-download rather than re-verify the same corrupt bytes."""
        source, _ = _tarball(str(tmp_path), {"f": b"x"})
        archive = os.path.join(tmp_path, "a.tar.gz")
        with served(source), pytest.raises(fetch.FetchError, match="sha256"):
            fetch.fetch_and_extract_archive(
                "https://example.invalid/a.tar.gz", "0" * 64, archive, os.path.join(tmp_path, "extracted")
            )
        assert not os.path.exists(archive)

    def test_the_archive_is_removed_unless_kept(self, tmp_path, served):
        source, digest = _tarball(str(tmp_path), {"f": b"x"})
        archive = os.path.join(tmp_path, "a.tar.gz")
        with served(source):
            fetch.fetch_and_extract_archive(
                "https://example.invalid/a.tar.gz", digest, archive, os.path.join(tmp_path, "extracted")
            )
        assert not os.path.exists(archive)
