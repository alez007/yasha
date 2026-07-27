import hashlib
import tarfile

import pytest

from modelship.utils import extract_tar, verify_sha256


class TestVerifySha256:
    def test_passes_when_hash_matches(self, tmp_path):
        path = tmp_path / "file.bin"
        path.write_bytes(b"hello")
        verify_sha256(str(path), hashlib.sha256(b"hello").hexdigest())  # no raise

    def test_raises_on_mismatch(self, tmp_path):
        path = tmp_path / "file.bin"
        path.write_bytes(b"hello")
        with pytest.raises(ValueError, match="sha256 verification"):
            verify_sha256(str(path), "0" * 64)


class TestExtractTar:
    def test_flattens_nested_members_to_basename(self, tmp_path):
        src_dir = tmp_path / "nested" / "dir"
        src_dir.mkdir(parents=True)
        (src_dir / "file.txt").write_bytes(b"contents")
        archive = tmp_path / "archive.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(src_dir / "file.txt", arcname="nested/dir/file.txt")

        extract_dir = tmp_path / "extracted"
        extract_tar(str(archive), str(extract_dir))

        assert (extract_dir / "file.txt").read_bytes() == b"contents"
        assert not (extract_dir / "nested").exists()
