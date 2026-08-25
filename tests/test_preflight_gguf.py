"""Parity tests for the skimming GGUF reader against stock GGUFReader."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gguf")

from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGUFEndian

from modelship.preflight._gguf import _MAX_KEPT_ARRAY, SkimGGUFReader

_VOCAB = _MAX_KEPT_ARRAY + 1000
_TENSORS = ["blk.0.attn_k.weight", "blk.1.ssm_conv1d.weight", "output.weight"]


def _write_gguf(path, endianess=GGUFEndian.LITTLE):
    writer = GGUFWriter(str(path), "testarch", endianess=endianess)
    writer.add_uint32("testarch.block_count", 12)
    writer.add_uint32("testarch.context_length", 262144)
    writer.add_string("testarch.chat_template", "{{ bos }}")
    writer.add_array("testarch.attention.head_count_kv", [8] * 12)
    writer.add_array("testarch.attention.sliding_window_pattern", [True, False] * 6)
    # Oversized: one string array (variable stride) and one scalar array.
    writer.add_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(_VOCAB)])
    writer.add_array("tokenizer.ggml.token_type", [1] * _VOCAB)
    for name in _TENSORS:
        writer.add_tensor(name, np.zeros((2, 2), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return str(path)


@pytest.fixture
def gguf_path(tmp_path):
    return _write_gguf(tmp_path / "model.gguf")


class TestSkimGGUFReader:
    def test_kept_fields_match_stock_reader(self, gguf_path):
        ref, fast = GGUFReader(gguf_path), SkimGGUFReader(gguf_path)
        assert set(ref.fields) == set(fast.fields)
        kept = [k for k, f in ref.fields.items() if len(f.data) <= _MAX_KEPT_ARRAY]
        assert {k: ref.fields[k].contents() for k in kept} == {k: fast.fields[k].contents() for k in kept}

    def test_data_offset_matches(self, gguf_path):
        # Accumulates every field size plus the tensor-info walk, so a one-byte
        # skip error shifts it.
        assert SkimGGUFReader(gguf_path).data_offset == GGUFReader(gguf_path).data_offset

    def test_tensor_names_match(self, gguf_path):
        assert [t.name for t in SkimGGUFReader(gguf_path).tensors] == _TENSORS

    @pytest.mark.parametrize("key", ["tokenizer.ggml.tokens", "tokenizer.ggml.token_type"])
    def test_oversized_arrays_read_back_as_none(self, gguf_path, key):
        assert SkimGGUFReader(gguf_path).fields[key].contents() is None

    def test_per_layer_arrays_survive(self, gguf_path):
        fields = SkimGGUFReader(gguf_path).fields
        assert fields["testarch.attention.head_count_kv"].contents() == [8] * 12
        assert fields["testarch.attention.sliding_window_pattern"].contents() == [True, False] * 6

    def test_byte_swapped_file_matches_stock_reader(self, tmp_path):
        path = _write_gguf(tmp_path / "be.gguf", endianess=GGUFEndian.BIG)
        ref, fast = GGUFReader(path), SkimGGUFReader(path)
        assert ref.byte_order == "S"
        assert ref.data_offset == fast.data_offset
        assert fast.fields["testarch.block_count"].contents() == 12
        assert fast.fields["testarch.attention.head_count_kv"].contents() == [8] * 12
