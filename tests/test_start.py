"""Tests for start.py CLI argument parsing and helpers."""

from start import _rand_suffix, parse_args


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.config is None
        assert args.redeploy is False
        assert args.gateway_name is None
        assert args.ray_cluster_address is None
        assert args.ray_redis_port is None
        assert args.use_existing_ray_cluster is None

    def test_redeploy_flag(self):
        args = parse_args(["--redeploy"])
        assert args.redeploy is True

    def test_config_path(self):
        args = parse_args(["--config", "/some/path/models.yaml"])
        assert args.config == "/some/path/models.yaml"

    def test_gateway_name(self):
        args = parse_args(["--gateway-name", "my-gateway"])
        assert args.gateway_name == "my-gateway"

    def test_all_flags_combined(self):
        args = parse_args(
            [
                "--config",
                "llm.yaml",
                "--gateway-name",
                "llm-api",
                "--redeploy",
                "--ray-cluster-address",
                "10.0.0.1",
                "--ray-redis-port",
                "6379",
                "--use-existing-ray-cluster",
            ]
        )
        assert args.config == "llm.yaml"
        assert args.gateway_name == "llm-api"
        assert args.redeploy is True
        assert args.ray_cluster_address == "10.0.0.1"
        assert args.ray_redis_port == "6379"
        assert args.use_existing_ray_cluster is True


class TestRandSuffix:
    def test_default_length(self):
        suffix = _rand_suffix()
        assert len(suffix) == 5

    def test_custom_length(self):
        suffix = _rand_suffix(10)
        assert len(suffix) == 10

    def test_chars_are_alphanumeric_lowercase(self):
        for _ in range(50):
            suffix = _rand_suffix()
            assert all(c.islower() or c.isdigit() for c in suffix)
