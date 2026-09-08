"""`run_deploy_loop` must give up on a model that keeps failing to deploy, while
still waiting indefinitely for one that is only short of capacity."""

from unittest.mock import MagicMock

import pytest
from ray.serve.schema import LoggingConfig

from modelship.deploy import strategy
from modelship.infer.infer_config import ModelshipModelConfig


def _model(name: str) -> ModelshipModelConfig:
    return ModelshipModelConfig.model_validate(
        {"name": name, "model": f"org/{name}", "usecase": "generate", "loader": "vllm", "num_gpus": 1}
    )


def _ctx() -> strategy.DeployContext:
    return strategy.DeployContext(
        coordinator=MagicMock(),
        replica_coordinator=MagicMock(),
        probe=MagicMock(),
        operator_id="op-1",
        gateway_name="g",
        serve_logging_config=LoggingConfig(),
        deployed_this_run={},
    )


@pytest.fixture
def loop(monkeypatch):
    """Drives run_deploy_loop off a per-model script of statuses, recording the
    pass sleeps and any app deletion. The last entry of a script repeats."""
    sleeps: list[float] = []
    deleted: list[str] = []
    monkeypatch.setattr(strategy.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(strategy.serve, "delete", lambda name: deleted.append(name))

    def run(scripts: dict[str, list[tuple[str, str | None]]]):
        calls: dict[str, int] = {}

        def fake_attempt(config, ctx):
            script = scripts[config.name]
            i = calls.get(config.name, 0)
            calls[config.name] = i + 1
            return script[min(i, len(script) - 1)]

        monkeypatch.setattr(strategy, "try_reserve_and_deploy", fake_attempt)
        models = [_model(name) for name in scripts]
        passes, failed = strategy.run_deploy_loop(models, _ctx())
        return {
            "passes": passes,
            "failed": {c.name: detail for c, detail in failed},
            "attempts": calls,
            "sleeps": sleeps,
            "deleted": deleted,
        }

    return run


TRANSIENT = ("transient", "RuntimeError: engine died")
SKIPPED = ("skipped", None)
DEPLOYED = ("deployed", None)


class TestTransientCap:
    def test_a_model_that_always_fails_is_given_up_on(self, loop):
        r = loop({"a": [TRANSIENT]})
        assert r["attempts"]["a"] == strategy._MAX_TRANSIENT_FAILURES
        assert r["failed"] == {"a": "RuntimeError: engine died"}

    def test_giving_up_deletes_the_failed_app(self, loop):
        r = loop({"a": [TRANSIENT]})
        assert r["deleted"] == [_model("a").deployment_name("g")]

    def test_a_recovering_model_is_not_given_up_on(self, loop):
        r = loop({"a": [TRANSIENT, TRANSIENT, DEPLOYED]})
        assert r["failed"] == {}
        assert r["attempts"]["a"] == 3

    def test_a_fatal_report_still_wins_immediately(self, loop):
        r = loop({"a": [("fatal", "bad config")]})
        assert r["attempts"]["a"] == 1
        assert r["failed"] == {"a": "bad config"}

    def test_one_failing_model_does_not_strand_a_healthy_one(self, loop):
        r = loop({"a": [TRANSIENT], "b": [DEPLOYED]})
        assert r["failed"] == {"a": "RuntimeError: engine died"}
        assert r["attempts"]["b"] == 1


class TestCapacityWaitIsUncapped:
    def test_skips_never_consume_an_attempt(self, loop):
        r = loop({"a": [SKIPPED] * 20 + [DEPLOYED]})
        assert r["failed"] == {}
        assert r["attempts"]["a"] == 21

    def test_a_skip_between_failures_does_not_reset_the_count(self, loop):
        r = loop({"a": [TRANSIENT, SKIPPED, TRANSIENT, SKIPPED, TRANSIENT]})
        assert r["failed"] == {"a": "RuntimeError: engine died"}


class TestBackoff:
    def test_the_pass_sleep_doubles_per_failure(self, loop):
        r = loop({"a": [TRANSIENT]})
        # No sleep after the pass that gives up — nothing is left to retry.
        assert r["sleeps"] == [2 * strategy._DEPLOY_RETRY_SLEEP_S, 4 * strategy._DEPLOY_RETRY_SLEEP_S]

    def test_a_pure_capacity_wait_keeps_the_base_cadence(self, loop):
        r = loop({"a": [SKIPPED, SKIPPED, DEPLOYED]})
        assert r["sleeps"] == [strategy._DEPLOY_RETRY_SLEEP_S] * 2
