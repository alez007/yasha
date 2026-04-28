from types import SimpleNamespace

from modelship.infer.transformers.capabilities import TransformersCapabilities


def test_text_generation_pipeline_is_text_only():
    pipe = SimpleNamespace(task="text-generation")
    caps = TransformersCapabilities.detect(pipe)  # type: ignore[arg-type]
    assert caps.supports_image is False
    assert caps.supports_audio is False


def test_image_text_to_text_pipeline_supports_image():
    pipe = SimpleNamespace(task="image-text-to-text")
    caps = TransformersCapabilities.detect(pipe)  # type: ignore[arg-type]
    assert caps.supports_image is True
    assert caps.supports_audio is False


def test_visual_question_answering_supports_image():
    pipe = SimpleNamespace(task="visual-question-answering")
    caps = TransformersCapabilities.detect(pipe)  # type: ignore[arg-type]
    assert caps.supports_image is True


def test_unknown_task_defaults_to_text_only():
    pipe = SimpleNamespace(task="some-future-task")
    caps = TransformersCapabilities.detect(pipe)  # type: ignore[arg-type]
    assert caps.supports_image is False


def test_pipeline_without_task_attribute_is_text_only():
    pipe = SimpleNamespace()
    caps = TransformersCapabilities.detect(pipe)  # type: ignore[arg-type]
    assert caps.supports_image is False
