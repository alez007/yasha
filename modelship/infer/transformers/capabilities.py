"""Modality capability detection for a HuggingFace ``Pipeline`` instance."""

from dataclasses import dataclass

from transformers import Pipeline

# HF pipeline tasks whose ``__call__`` accepts image inputs alongside chat messages.
_IMAGE_TASKS = frozenset({"image-text-to-text", "visual-question-answering"})


@dataclass(frozen=True)
class TransformersCapabilities:
    """Modalities the underlying HF pipeline can ingest."""

    supports_image: bool
    supports_audio: bool = False  # no audio-aware chat pipelines today

    @classmethod
    def detect(cls, pipeline: Pipeline) -> "TransformersCapabilities":
        # The HF pipeline's ``task`` attribute is the authoritative signal: a
        # text-generation pipeline cannot ingest image_url parts even if the
        # underlying weights are a VLM. Multimodal chat needs an explicit task
        # like "image-text-to-text".
        task = getattr(pipeline, "task", None)
        return cls(supports_image=task in _IMAGE_TASKS)
