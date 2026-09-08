"""BaseInfer.backend_died: the replica's report-then-exit path for a backend that
died after startup."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import ray.serve.context
from ray import serve

from modelship.infer import base_infer
from modelship.infer.base_infer import BaseInfer
from modelship.infer.infer_config import ModelshipModelConfig


class _Infer(BaseInfer[dict]):
    """backend_died is concrete on the base; these three are not."""

    def shutdown(self) -> None: ...

    async def start(self) -> None: ...

    async def warmup(self) -> None: ...


def _infer(**overrides) -> _Infer:
    raw = {"name": "qwen", "model": "org/qwen", "usecase": "generate", "loader": "vllm", **overrides}
    obj = object.__new__(_Infer)
    obj.model_config = ModelshipModelConfig.model_validate(raw)
    return obj


class _ExitError(Exception):
    """Stands in for os._exit so a test can assert the call without dying."""


@pytest.fixture
def harness(monkeypatch):
    coordinator = MagicMock()
    monkeypatch.setattr(base_infer.os, "_exit", MagicMock(side_effect=_ExitError))
    monkeypatch.setattr(base_infer.os, "environ", {"MSHIP_GATEWAY_NAME": "gw"})
    monkeypatch.setattr(base_infer.serve, "get_replica_context", lambda: SimpleNamespace(app_name="qwen-aaaa"))
    monkeypatch.setattr(base_infer.ray, "get", MagicMock())
    with patch("modelship.infer.deploy_coordinator.get_or_create_coordinator", return_value=coordinator):
        yield coordinator


def _report(coordinator):
    return coordinator.report_replica_death.remote.call_args


class TestReplicaContextCrossesThreads:
    """llama_server reports from its process-watcher thread, so the app name has to
    be readable there. Ray stores the replica context in a module global, not the
    ContextVar it uses for request state — this fails if that ever changes."""

    def test_the_app_name_is_readable_from_another_thread(self, monkeypatch):
        monkeypatch.setattr(
            ray.serve.context, "_INTERNAL_REPLICA_CONTEXT", SimpleNamespace(app_name="qwen-aaaa"), raising=False
        )
        seen: list[str] = []
        thread = threading.Thread(target=lambda: seen.append(serve.get_replica_context().app_name))
        thread.start()
        thread.join()
        assert seen == ["qwen-aaaa"]


class TestBackendDied:
    def test_reports_then_exits(self, harness):
        with pytest.raises(_ExitError):
            _infer().backend_died("engine core died")
        assert _report(harness).args == ("gw", "qwen-aaaa", 1, "engine core died")
        base_infer.os._exit.assert_called_once_with(1)

    def test_exits_even_when_the_report_fails(self, harness, monkeypatch):
        monkeypatch.setattr(base_infer.ray, "get", MagicMock(side_effect=TimeoutError("coordinator wedged")))
        with pytest.raises(_ExitError):
            _infer().backend_died("engine core died")
        base_infer.os._exit.assert_called_once_with(1)

    def test_the_report_is_bounded_by_a_timeout(self, harness):
        with pytest.raises(_ExitError):
            _infer().backend_died("engine core died")
        assert base_infer.ray.get.call_args.kwargs["timeout"] == base_infer._DEATH_REPORT_TIMEOUT_S

    def test_a_fixed_replica_count_is_the_ceiling(self, harness):
        with pytest.raises(_ExitError):
            _infer(num_replicas=3).backend_died("engine core died")
        assert _report(harness).args[2] == 3

    def test_autoscaling_reports_its_max(self, harness):
        with pytest.raises(_ExitError):
            _infer(autoscaling_config={"min_replicas": 1, "max_replicas": 6}).backend_died("engine core died")
        assert _report(harness).args[2] == 6
