"""A base model carries no chat template, so `get_chat_template` raises instead of
returning one. `init_serving_chat` must swallow that and leave the chat pipeline
unset, which is what the request seams already gate on."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from modelship.infer.infer_config import (
    ModelLoader,
    ModelshipModelConfig,
    ModelUsecase,
    RawRequestProxy,
)
from modelship.infer.vllm.vllm_infer import VllmInfer
from modelship.openai.protocol import ChatCompletionRequest, ErrorResponse

_NO_TEMPLATE = ValueError(
    "Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!"
)


def _make_infer(exc: Exception | None) -> VllmInfer:
    """Bypasses __init__ — `init_serving_chat` only reads these three attributes
    before it reaches the tokenizer."""
    infer = object.__new__(VllmInfer)
    infer.model_config = ModelshipModelConfig(
        name="base-model", model="some/base", usecase=ModelUsecase.generate, loader=ModelLoader.vllm
    )
    infer.supported_tasks = ["generate"]

    def get_chat_template():
        if exc is not None:
            raise exc
        return "{{ messages }}"

    # shutdown is a no-op so the actor's __del__ does not log a stub teardown failure.
    infer.engine = SimpleNamespace(
        get_tokenizer=lambda: SimpleNamespace(get_chat_template=get_chat_template), shutdown=lambda: None
    )
    return infer


class TestMissingChatTemplate:
    @pytest.mark.asyncio
    async def test_init_serving_chat_leaves_the_pipeline_unset(self):
        infer = _make_infer(_NO_TEMPLATE)
        await infer.init_serving_chat()
        assert not hasattr(infer, "openai_serving_render")

    @pytest.mark.asyncio
    async def test_a_non_value_error_still_propagates(self):
        infer = _make_infer(RuntimeError("tokenizer is gone"))
        with pytest.raises(RuntimeError):
            await infer.init_serving_chat()

    @pytest.mark.asyncio
    async def test_chat_requests_are_rejected_rather_than_crashing(self):
        infer = _make_infer(_NO_TEMPLATE)
        await infer.init_serving_chat()
        result = await infer._prepare_chat(
            ChatCompletionRequest(model="base-model", messages=[{"role": "user", "content": "hi"}]),
            RawRequestProxy(None, {}),
        )
        assert isinstance(result, ErrorResponse)
        assert result._http_status == 404
