"""End-to-end deployment crash recovery: kill a backend engine subprocess
directly and verify Ray Serve respawns a working replica."""

import time

import pytest

_PING_PROMPT = [{"role": "user", "content": "hi"}]


def _find_and_kill_vllm_engine_core(deadline_s: float = 30) -> int:
    """SIGKILL the vLLM engine-core subprocess (titled `VLLM::EngineCore` via
    setproctitle) and return its PID."""
    import psutil

    end = time.time() + deadline_s
    while time.time() < end:
        for proc in psutil.process_iter():
            try:
                cmdline = " ".join(proc.cmdline())
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            if "VLLM::EngineCore" in cmdline:
                pid = proc.pid
                proc.kill()
                return pid
        time.sleep(1)
    pytest.fail("Could not find a vLLM EngineCore subprocess to kill within the deadline")


def _poll(predicate, deadline_s: float) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if predicate():
            return True
        time.sleep(1)
    return False


@pytest.mark.integration
@pytest.mark.vllm
class TestVllmCrashRecovery:
    """Kills the vLLM engine-core subprocess directly and verifies the replica recovers."""

    MODEL = "chat-capable"

    @pytest.fixture(autouse=True, scope="class")
    def _deploy(self, model_deployer):
        model_deployer.deploy(self.MODEL)

    def test_recovers_after_engine_core_crash(self, client):
        # Sanity: the deployment is actually serving before we kill anything.
        client.chat.completions.create(model=self.MODEL, messages=_PING_PROMPT, max_tokens=4)

        _find_and_kill_vllm_engine_core()

        def _request_succeeds() -> bool:
            try:
                client.chat.completions.create(model=self.MODEL, messages=_PING_PROMPT, max_tokens=4, timeout=15)
                return True
            except Exception:
                return False

        recovered = _poll(_request_succeeds, deadline_s=90)
        assert recovered, (
            "expected the deployment to recover (replica respawn) after its engine core crashed; requests kept failing"
        )
