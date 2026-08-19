"""End-to-end multi-replica gateway routing consistency: a deployed/removed
model must converge on every gateway replica, not just the one a direct push
would have hit."""

import time

import pytest

from openai import OpenAI


def _model_in_all_samples(client: OpenAI, model: str, samples: int = 20) -> bool:
    """True iff `model` appears on every sampled /v1/models call — a stale
    replica would omit it on some."""
    return all(model in {m.id for m in client.models.list().data} for _ in range(samples))


def _model_in_no_samples(client: OpenAI, model: str, samples: int = 20) -> bool:
    return all(model not in {m.id for m in client.models.list().data} for _ in range(samples))


def _poll(predicate, deadline_s: float) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if predicate():
            return True
        time.sleep(1)
    return False


@pytest.mark.integration
@pytest.mark.llama_server
@pytest.mark.gateway_ha
class TestGatewayReplicaConsistency:
    """With 2 gateway replicas, a deployed model must become routable on both,
    and a removed one must stop routing on both."""

    def test_add_and_remove_propagate_to_all_replicas(self, client, model_deployer):
        # Warm both replicas (spread requests so each starts its watch loop).
        for _ in range(10):
            client.models.list()

        model_deployer.deploy("chat-llama-server-plain")
        assert _poll(lambda: _model_in_all_samples(client, "chat-llama-server-plain"), deadline_s=60), (
            "deployed model did not become routable on all gateway replicas"
        )
        completion = client.chat.completions.create(
            model="chat-llama-server-plain", messages=[{"role": "user", "content": "hi"}], max_tokens=5
        )
        assert completion.choices[0].message.content is not None

        # Reconcile to a different model — chat-llama-server-plain is removed everywhere.
        model_deployer.deploy("chat-llama-server")
        assert _poll(lambda: _model_in_no_samples(client, "chat-llama-server-plain"), deadline_s=60), (
            "removed model still routable on some gateway replica"
        )

        # Requests to the removed model now 404 on every replica — none route into
        # the torn-down deployment (which would surface as a 5xx, not a 404).
        import openai

        for _ in range(20):
            with pytest.raises(openai.NotFoundError):
                client.chat.completions.create(
                    model="chat-llama-server-plain", messages=[{"role": "user", "content": "hi"}], max_tokens=5
                )
