"""Tests for modelship.openai.mcp.egress: outbound MCP server URL policy."""

import pytest

from modelship.openai.mcp.egress import validate_server_url
from modelship.openai.utils.responses import ResponsesApiError


class TestPermissiveDefaults:
    def test_localhost_http_allowed(self):
        validate_server_url("http://localhost:3000/mcp")

    def test_private_ip_allowed(self):
        validate_server_url("http://192.168.1.5:8000/mcp")

    def test_public_https_allowed(self):
        validate_server_url("https://example.com/mcp")


class TestSchemeValidation:
    def test_non_http_scheme_rejected(self):
        with pytest.raises(ResponsesApiError, match="http or https"):
            validate_server_url("ftp://example.com/mcp")

    def test_require_https_env_rejects_plain_http(self, monkeypatch):
        monkeypatch.setenv("MSHIP_MCP_REQUIRE_HTTPS", "true")
        with pytest.raises(ResponsesApiError, match="https"):
            validate_server_url("http://example.com/mcp")

    def test_require_https_env_allows_https(self, monkeypatch):
        monkeypatch.setenv("MSHIP_MCP_REQUIRE_HTTPS", "true")
        validate_server_url("https://example.com/mcp")


class TestHostnameRequired:
    def test_missing_hostname_rejected(self):
        with pytest.raises(ResponsesApiError, match="hostname"):
            validate_server_url("http:///mcp")


class TestMetadataBlock:
    def test_literal_metadata_ip_blocked(self):
        with pytest.raises(ResponsesApiError, match="metadata"):
            validate_server_url("http://169.254.169.254/mcp")

    def test_other_link_local_ip_blocked(self):
        with pytest.raises(ResponsesApiError, match="metadata"):
            validate_server_url("http://169.254.1.1/mcp")

    def test_google_metadata_hostname_blocked(self):
        with pytest.raises(ResponsesApiError, match="metadata"):
            validate_server_url("http://metadata.google.internal/mcp")

    def test_metadata_block_applies_even_with_allowed_hosts(self, monkeypatch):
        monkeypatch.setenv("MSHIP_MCP_ALLOWED_HOSTS", "169.254.169.254")
        with pytest.raises(ResponsesApiError, match="metadata"):
            validate_server_url("http://169.254.169.254/mcp")


class TestAllowedHosts:
    def test_allowed_host_passes(self, monkeypatch):
        monkeypatch.setenv("MSHIP_MCP_ALLOWED_HOSTS", "example.com,other.com")
        validate_server_url("https://example.com/mcp")

    def test_non_allowed_host_rejected(self, monkeypatch):
        monkeypatch.setenv("MSHIP_MCP_ALLOWED_HOSTS", "example.com")
        with pytest.raises(ResponsesApiError, match="MSHIP_MCP_ALLOWED_HOSTS"):
            validate_server_url("https://evil.com/mcp")

    def test_allowlist_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("MSHIP_MCP_ALLOWED_HOSTS", "Example.COM")
        validate_server_url("https://example.com/mcp")
