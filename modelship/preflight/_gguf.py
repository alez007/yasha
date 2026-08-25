"""Faster GGUF metadata reads for preflight.

`GGUFReader.__init__` materializes every field eagerly, including the tokenizer
vocab; skipping oversized arrays yields identical values for everything read here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gguf import GGUFReader
from gguf.constants import GGUFValueType

# Per-layer arrays run to the block count, vocab arrays to 100k+. Anything
# larger is skipped and reads back as None.
_MAX_KEPT_ARRAY = 4096


class SkimGGUFReader(GGUFReader):
    """`GGUFReader` that walks past oversized arrays instead of materializing them.

    `_build_fields` passes `skip_sum=True`, advancing by the size returned here.
    """

    def _get_field_parts(
        self, orig_offs: int, raw_type: int
    ) -> tuple[int, list[npt.NDArray[Any]], list[int], list[GGUFValueType]]:
        gtype = GGUFValueType(raw_type)
        if gtype != GGUFValueType.ARRAY:
            return super()._get_field_parts(orig_offs, raw_type)

        offs = orig_offs
        raw_itype = self._get(offs, np.uint32)
        offs += int(raw_itype.nbytes)
        alen = self._get(offs, np.uint64)
        offs += int(alen.nbytes)
        count = int(alen[0])
        if count <= _MAX_KEPT_ARRAY:
            return super()._get_field_parts(orig_offs, raw_type)

        itype = GGUFValueType(int(raw_itype[0]))
        nptype = self.gguf_scalar_to_np.get(itype)
        if nptype is not None:
            offs += count * int(np.empty([], dtype=nptype).itemsize)
        elif itype == GGUFValueType.STRING:
            # Variable stride, so every length must be read; a memoryview is
            # cheaper than slicing the memmap per element.
            buf = memoryview(self.data.data).cast("B")
            order = "big" if self.byte_order == "S" else "little"
            frm = int.from_bytes
            for _ in range(count):
                offs += 8 + frm(buf[offs : offs + 8], order)
        else:
            return super()._get_field_parts(orig_offs, raw_type)

        # No types means contents() returns None, not a misleading empty list.
        return offs - orig_offs, [raw_itype, alen], [], []
