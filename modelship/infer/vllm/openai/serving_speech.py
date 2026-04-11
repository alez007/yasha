"""Stub for vllm speech serving.

TTS models require model-specific logic (codec decoding, prompt formatting,
sampling parameters) that cannot be generalised into the vllm loader.
Use loader='custom' with an appropriate plugin instead.
"""
