"""Stub for vllm speech serving.

TTS models require model-specific logic (codec decoding, prompt formatting,
sampling parameters) that cannot be generalised into the vllm loader. Use the
sherpa_onnx loader for TTS instead.
"""
