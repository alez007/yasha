"""Modality capability detection for a loaded llama.cpp ``Llama`` instance."""

from dataclasses import dataclass

from llama_cpp import Llama


@dataclass(frozen=True)
class LlamaCppCapabilities:
    """Modalities the underlying ``Llama`` instance can ingest."""

    supports_image: bool
    supports_audio: bool = False  # llama-cpp-python ships no native audio chat handlers today

    @classmethod
    def detect(cls, llama: Llama) -> "LlamaCppCapabilities":
        # The only reliable signal that a Llama instance can ingest images is whether
        # the user wired up a multimodal chat_handler at load time
        # (e.g. Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler).
        # Those handlers expose `load_image` to decode image_url parts.
        handler = getattr(llama, "chat_handler", None)
        supports_image = handler is not None and hasattr(handler, "load_image")
        return cls(supports_image=supports_image)
