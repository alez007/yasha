"""Two loaders sharing one physical GPU via fractional num_gpus.

`frac-share-vllm` (num_gpus=0.6) and `frac-share-llama-server` (num_gpus=0.3)
are deployed together — Ray's own fractional scheduling packs both fractional
requests onto the same physical GPU. Proves preflight's share-basis sizing
(fraction * total VRAM, not free VRAM) produces a config each engine can
actually boot and serve from concurrently, not just a config that validates.

`MODEL_CONFIGS`, `_Deployer`, and the `mship_cluster`/`model_deployer`/`client`
fixtures live in conftest.py — shared with test_integration.py so both files
run against the same live cluster within one pytest session.
"""

import concurrent.futures

import pytest


@pytest.mark.integration
@pytest.mark.fractional_gpu_sharing
class TestFractionalGpuSharing:
    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy("frac-share-vllm", "frac-share-llama-server")

    def test_vllm_tenant_serves_requests(self, client):
        completion = client.chat.completions.create(
            model="frac-share-vllm",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=32,
            temperature=0,
        )
        content = completion.choices[0].message.content
        assert content
        assert "paris" in content.lower()

    def test_llama_server_tenant_serves_requests(self, client):
        completion = client.chat.completions.create(
            model="frac-share-llama-server",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=32,
            temperature=0,
        )
        content = completion.choices[0].message.content
        assert content
        assert "paris" in content.lower()

    def test_concurrent_requests_across_both_tenants_succeed(self, client):
        # The real proof of co-tenancy: both engines already have their KV
        # cache/context allocated against their own declared share. Driving
        # both at once and getting real completions back from each confirms
        # neither starved the other at inference time, not just at boot.
        prompts = [
            {
                "model": "frac-share-vllm",
                "messages": [{"role": "user", "content": "Count from 1 to 20, one number per line."}],
                "max_tokens": 128,
            },
            {
                "model": "frac-share-llama-server",
                "messages": [{"role": "user", "content": "Count from 1 to 20, one number per line."}],
                "max_tokens": 128,
            },
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = [pool.submit(client.chat.completions.create, **p) for p in prompts]
            completions = [f.result() for f in futures]

        for completion in completions:
            assert completion.choices[0].message.content
            assert completion.choices[0].finish_reason in ("stop", "length")
