"""Same-box integration test for the --address/--token cluster join path
(mship_deploy.py's connect_ray join branch, modelship/deploy/serve_utils.py).

Its own file/process, separate from test_integration.py's shared
`mship_cluster` fixture: runs its own throwaway mship_deploy.py head (own-head,
not a bare `ray start --head`, so it carries real mship_<loader> capability
resources) on distinct ports + RAY_TMPDIR, and only ever signals its own
processes — never `ray stop`.
"""

import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

# Off Ray's own 6379 (test_integration.py's shared cluster) and modelship's real
# defaults (6380/8265/8000) — must never collide with a real cluster on this box.
_THROWAWAY_HEAD_PORT = 6480
_THROWAWAY_DASHBOARD_PORT = 6481
_THROWAWAY_GATEWAY_PORT = 6482


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM the whole process group a `start_new_session=True` Popen created,
    escalating to SIGKILL on timeout — never `ray stop`, which sweeps every Ray
    process on the machine."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def _poll(predicate, deadline_s: float) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if predicate():
            return True
        time.sleep(1)
    return False


def _empty_config_path(dir_path) -> str:
    """Write an empty models.yaml under dir_path and return its path, for an
    explicit --config that never falls back to the repo's real config/models.yaml."""
    path = str(Path(dir_path) / "empty-models.yaml")
    Path(path).write_text("models: []\n")
    return path


def _ray_status(env: dict) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            ["ray", "status", f"--address=127.0.0.1:{_THROWAWAY_HEAD_PORT}"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


@pytest.fixture
def throwaway_head(tmp_path):
    """A fully independent modelship own-head cluster, token auth on. RAY_TMPDIR
    is a short /tmp-rooted dir (AF_UNIX socket path length limit), not tmp_path.
    Yields (port, token, env)."""
    head_home = tmp_path / "head_home"
    head_home.mkdir()
    head_ray_tmp = tempfile.mkdtemp(prefix="mship-join-test-head-")
    # RAY_AUTH_MODE=token stays set (not popped) so `ray status` polling below can
    # auto-read the token from HOME/.ray/auth_token once ray.init() writes it.
    env = {
        **os.environ,
        "HOME": str(head_home),
        "RAY_TMPDIR": head_ray_tmp,
        "PYTHONUNBUFFERED": "1",
        "RAY_AUTH_MODE": "token",
    }
    env.pop("RAY_AUTH_TOKEN", None)

    log_path = tmp_path / "head.log"
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [
                "uv",
                "run",
                "mship_deploy.py",
                "--config",
                _empty_config_path(head_home),
                "--ray-port",
                str(_THROWAWAY_HEAD_PORT),
                "--dashboard-port",
                str(_THROWAWAY_DASHBOARD_PORT),
                "--openai-api-port",
                str(_THROWAWAY_GATEWAY_PORT),
                "--ray-auth",
                "token",
                "--no-metrics",
                "--prune-ray-sessions",
                "false",
            ],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        try:
            # ray.init()'s own-cluster path auto-generates ~/.ray/auth_token itself
            # (unlike `ray start`'s CLI path, which requires one to already exist).
            deadline = time.time() + 60
            while _ray_status(env) is None:
                if proc.poll() is not None:
                    pytest.fail(f"Throwaway head process exited before starting up. Log:\n{log_path.read_text()}")
                if time.time() > deadline:
                    _terminate_process_group(proc)
                    pytest.fail(f"Timed out waiting for the throwaway head to start. Log:\n{log_path.read_text()}")
                time.sleep(0.5)

            token_path = head_home / ".ray" / "auth_token"
            assert _poll(lambda: token_path.exists(), deadline_s=10), (
                f"ray.init() didn't auto-generate an auth token. Log:\n{log_path.read_text()}"
            )
            token = token_path.read_text().strip()

            try:
                yield _THROWAWAY_HEAD_PORT, token, env
            finally:
                _terminate_process_group(proc)
        finally:
            shutil.rmtree(head_ray_tmp, ignore_errors=True)


def _throwaway_head_node_count(env: dict) -> int:
    result = _ray_status(env)
    return result.stdout.count("node_") if result else 0


def _join_node_procs_alive(join_ray_tmp: str) -> bool:
    """True iff any of the joiner's own Ray node subprocesses are still
    running, matched by its unique RAY_TMPDIR."""
    result = subprocess.run(
        ["pgrep", "-f", join_ray_tmp],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


@pytest.mark.integration
@pytest.mark.cluster_join
class TestClusterJoin:
    def _joiner_env(self, tmp_path, suffix: str) -> tuple[dict, str, str]:
        """Returns (env, join_ray_tmp, config_path) for a joiner subprocess."""
        join_home = tmp_path / suffix
        join_home.mkdir(exist_ok=True)
        join_ray_tmp = tempfile.mkdtemp(prefix=f"mship-join-test-{suffix}-")
        # PYTHONUNBUFFERED: stdout goes to a log file, not a TTY, so Python block-buffers by default.
        env = {**os.environ, "HOME": str(join_home), "RAY_TMPDIR": join_ray_tmp, "PYTHONUNBUFFERED": "1"}
        env.pop("RAY_AUTH_MODE", None)
        env.pop("RAY_AUTH_TOKEN", None)
        return env, join_ray_tmp, _empty_config_path(join_home)

    def _run_joiner(self, tmp_path, head_port, token, suffix="join_home") -> subprocess.CompletedProcess:
        env, join_ray_tmp, config_path = self._joiner_env(tmp_path, suffix)
        args = [
            "uv",
            "run",
            "mship_deploy.py",
            "--config",
            config_path,
            "--address",
            f"127.0.0.1:{head_port}",
            "--node-num-cpus",
            "0",
            "--node-num-gpus",
            "0",
            "--no-metrics",  # avoid a real 8079 collision on this shared test box
            "--prune-ray-sessions",
            "false",
        ]
        if token is not None:
            args += ["--token", token]
        try:
            return subprocess.run(args, env=env, capture_output=True, text=True, timeout=90)
        finally:
            shutil.rmtree(join_ray_tmp, ignore_errors=True)

    def test_join_without_token_rejected(self, tmp_path, throwaway_head):
        head_port, _token, _env = throwaway_head
        result = self._run_joiner(tmp_path, head_port, token=None)
        assert result.returncode != 0, f"expected non-zero exit, got 0. stdout/err:\n{result.stdout}{result.stderr}"

    def test_join_with_wrong_token_rejected(self, tmp_path, throwaway_head):
        head_port, _token, _env = throwaway_head
        result = self._run_joiner(tmp_path, head_port, token="not-the-right-token")
        assert result.returncode != 0, f"expected non-zero exit, got 0. stdout/err:\n{result.stdout}{result.stderr}"

    def test_join_with_correct_token_adds_node_then_leaves_cleanly(self, tmp_path, throwaway_head):
        head_port, token, head_env = throwaway_head
        env, join_ray_tmp, config_path = self._joiner_env(tmp_path, "join_home")

        log_path = tmp_path / "joiner.log"
        try:
            with open(log_path, "w") as log_file:
                proc = subprocess.Popen(
                    [
                        "uv",
                        "run",
                        "mship_deploy.py",
                        "--config",
                        config_path,
                        "--address",
                        f"127.0.0.1:{head_port}",
                        "--token",
                        token,
                        "--gateway-name",
                        "join-test-gateway",
                        "--node-num-cpus",
                        "0",
                        "--node-num-gpus",
                        "0",
                        "--no-metrics",
                        "--prune-ray-sessions",
                        "false",
                    ],
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                try:
                    assert _poll(lambda: _throwaway_head_node_count(head_env) == 2, deadline_s=60), (
                        f"joiner did not appear as a second node within timeout. Log:\n{log_path.read_text()}"
                    )
                    assert proc.poll() is None, (
                        f"joiner exited early (code {proc.poll()}) instead of staying resident. "
                        f"Log:\n{log_path.read_text()}"
                    )

                    # SIGTERM triggers _cleanup -> leave_ray_cluster (this node only).
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=60)
                    assert _poll(lambda: not _join_node_procs_alive(join_ray_tmp), deadline_s=60), (
                        f"joiner's Ray node subprocesses are still running after leave. Log:\n{log_path.read_text()}"
                    )
                    # The head itself was never signaled and must still be reachable.
                    assert _throwaway_head_node_count(head_env) >= 1
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=10)
        finally:
            shutil.rmtree(join_ray_tmp, ignore_errors=True)
