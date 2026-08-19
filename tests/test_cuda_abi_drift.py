"""Every extension module's libcudart/libcublas/libcublasLt must match torch's CUDA major;
libcurand/libcusparse/libcusolver/libcufft are excluded since they don't track it in lockstep."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_NEEDED_RE = re.compile(r"\(NEEDED\)\s+Shared library: \[(?P<name>[^\]]+)\]")
_TRACKED_SONAME_RE = re.compile(r"^lib(?:cudart|cublas|cublasLt)\.so\.(?P<major>\d+)$")

# bitsandbytes bundles one prebuilt .so per CUDA major and selects the matching one
# at import time, so older-major libs are expected to sit unused on disk.
_EXCLUDED_DIST_DIRS = {"bitsandbytes"}


def _torch_cuda_major() -> str | None:
    cuda_version = torch.version.cuda
    if not cuda_version:
        return None
    return cuda_version.split(".")[0]


@pytest.mark.skipif(shutil.which("readelf") is None, reason="readelf not on PATH")
def test_cuda_linked_extensions_match_torch_cuda_major():
    torch_major = _torch_cuda_major()
    if torch_major is None:
        pytest.skip("torch build has no CUDA (cpu extra) — nothing to compare against")

    site_packages = Path(torch.__file__).resolve().parent.parent
    mismatches = []
    for so_file in site_packages.rglob("*.so"):
        if so_file.relative_to(site_packages).parts[0] in _EXCLUDED_DIST_DIRS:
            continue
        result = subprocess.run(
            ["readelf", "-d", str(so_file)],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        for match in _NEEDED_RE.finditer(result.stdout):
            soname_match = _TRACKED_SONAME_RE.match(match.group("name"))
            if soname_match and soname_match.group("major") != torch_major:
                mismatches.append(f"{so_file.relative_to(site_packages)} needs {match.group('name')}")

    assert not mismatches, (
        f"found extension modules linked against a different CUDA major than torch's "
        f"cu{torch_major} build:\n" + "\n".join(mismatches)
    )
