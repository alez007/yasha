"""Tests for ModelshipAPI model discovery and routing."""

from unittest.mock import MagicMock, patch

import pytest

from modelship.openai.api import ModelshipAPI

# Access the underlying class, bypassing the @serve.deployment wrapper.
_ModelshipAPI = ModelshipAPI.func_or_class


@pytest.fixture
def api():
    """Create a ModelshipAPI instance with mocked Ray Serve context."""
    with patch("modelship.openai.api.serve.get_replica_context") as mock_ctx:
        mock_ctx.return_value.app_name = "test-gateway"
        return _ModelshipAPI()


class TestAddModels:
    @pytest.mark.asyncio
    async def test_add_single_model(self, api):
        mock_handle = MagicMock()
        with patch("modelship.openai.api.serve.get_app_handle", return_value=mock_handle):
            await api.add_models({"qwen-a3f9k": "qwen"})

        assert "qwen" in api.models
        assert len(api.models["qwen"]) == 1
        assert api.model_list[0].id == "qwen"

    @pytest.mark.asyncio
    async def test_add_multiple_deployments_same_model(self, api):
        mock_handle_1 = MagicMock()
        mock_handle_2 = MagicMock()
        with patch("modelship.openai.api.serve.get_app_handle", side_effect=[mock_handle_1, mock_handle_2]):
            await api.add_models({"qwen-a3f9k": "qwen", "qwen-b7x2p": "qwen"})

        assert len(api.models["qwen"]) == 2
        assert len(api.model_list) == 1

    @pytest.mark.asyncio
    async def test_add_different_models(self, api):
        mock_handle = MagicMock()
        with patch("modelship.openai.api.serve.get_app_handle", return_value=mock_handle):
            await api.add_models({"qwen-a3f9k": "qwen", "kokoro-c1m4n": "kokoro"})

        assert "qwen" in api.models
        assert "kokoro" in api.models
        assert len(api.model_list) == 2

    @pytest.mark.asyncio
    async def test_incremental_adds_new_handle_to_existing_model(self, api):
        handle_1 = MagicMock()
        handle_2 = MagicMock()
        with patch("modelship.openai.api.serve.get_app_handle", return_value=handle_1):
            await api.add_models({"qwen-a3f9k": "qwen"})
        with patch("modelship.openai.api.serve.get_app_handle", return_value=handle_2):
            await api.add_models({"qwen-b7x2p": "qwen"})

        assert len(api.models["qwen"]) == 2
        assert api.models["qwen"][0] is handle_1
        assert api.models["qwen"][1] is handle_2
        # Only one model card despite two deployments
        assert len(api.model_list) == 1

    @pytest.mark.asyncio
    async def test_handle_failure_skips(self, api):
        with patch("modelship.openai.api.serve.get_app_handle", side_effect=Exception("not found")):
            await api.add_models({"qwen-a3f9k": "qwen"})

        assert "qwen" not in api.models
        assert len(api.model_list) == 0

    @pytest.mark.asyncio
    async def test_records_per_model_load_times_and_ready_timestamp(self, api):
        await api.set_expected_models(["qwen", "kokoro"])
        assert api._expected_set_at is not None
        assert api._all_ready_at is None

        mock_handle = MagicMock()
        with patch("modelship.openai.api.serve.get_app_handle", return_value=mock_handle):
            await api.add_models({"qwen-a3f9k": "qwen"})
            assert "qwen" in api._model_load_times
            assert api._model_load_times["qwen"] >= 0
            assert api._all_ready_at is None

            await api.add_models({"kokoro-c1m4n": "kokoro"})
            assert "kokoro" in api._model_load_times
            assert api._all_ready_at is not None

    @pytest.mark.asyncio
    async def test_status_body_ready_flag(self, api):
        await api.set_expected_models(["qwen"])
        body = api._status_body()
        assert body["ready"] is False
        assert body["models_pending"] == ["qwen"]
        assert body["time_to_ready_s"] is None

        mock_handle = MagicMock()
        with patch("modelship.openai.api.serve.get_app_handle", return_value=mock_handle):
            await api.add_models({"qwen-a3f9k": "qwen"})

        body = api._status_body()
        assert body["ready"] is True
        assert body["models_pending"] == []
        assert body["time_to_ready_s"] is not None
        assert "qwen" in body["model_load_times_s"]


class TestGetHandle:
    @pytest.mark.asyncio
    async def test_round_robin(self, api):
        handle_a = MagicMock()
        handle_b = MagicMock()
        with patch("modelship.openai.api.serve.get_app_handle", side_effect=[handle_a, handle_b]):
            await api.add_models({"qwen-a3f9k": "qwen", "qwen-b7x2p": "qwen"})

        assert api._get_handle("qwen") is handle_a
        assert api._get_handle("qwen") is handle_b
        assert api._get_handle("qwen") is handle_a

    def test_unknown_model_raises(self, api):
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            api._get_handle("nonexistent")

    def test_none_model_raises(self, api):
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            api._get_handle(None)
