"""Tests for handlers.py — validation, platform classification, secret scanning."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handlers import (
    validate_hostname,
    validate_ssh_user,
    validate_account_name,
    is_auto_platform,
    is_manual_platform,
    platform_type_label,
    _safe_ssh_args,
)


class TestValidation:
    """Test input validation functions."""

    def test_validate_hostname_valid(self):
        assert validate_hostname("192.168.1.1")
        assert validate_hostname("example.com")
        assert validate_hostname("my-server.example.com")
        assert validate_hostname("10.0.0.1")

    def test_validate_hostname_invalid(self):
        assert not validate_hostname("")
        assert not validate_hostname("host with spaces")
        assert not validate_hostname("-invalid-start")
        assert not validate_hostname("a" * 254)  # too long

    def test_validate_ssh_user_valid(self):
        assert validate_ssh_user("ubuntu")
        assert validate_ssh_user("opc")
        assert validate_ssh_user("user-name")
        assert validate_ssh_user("user_name")

    def test_validate_ssh_user_invalid(self):
        assert not validate_ssh_user("")
        assert not validate_ssh_user("user name")
        assert not validate_ssh_user("-invalid")
        assert not validate_ssh_user("a" * 65)  # too long

    def test_validate_account_name_valid(self):
        ok, msg = validate_account_name("my-colab-1")
        assert ok
        ok, msg = validate_account_name("kaggle_prod")
        assert ok

    def test_validate_account_name_invalid(self):
        ok, msg = validate_account_name("")
        assert not ok
        ok, msg = validate_account_name("a" * 65)
        assert not ok
        ok, msg = validate_account_name("has spaces")
        assert not ok
        ok, msg = validate_account_name("-starts-dash")
        assert not ok


class TestPlatformClassification:
    """Test platform type classification."""

    def test_auto_platforms(self):
        assert is_auto_platform("kaggle")
        assert is_auto_platform("oracle_cloud")
        assert is_auto_platform("gcp")
        assert not is_auto_platform("google_colab")

    def test_manual_platforms(self):
        assert is_manual_platform("google_colab")
        assert is_manual_platform("paperspace")
        assert is_manual_platform("huggingface")
        assert not is_manual_platform("kaggle")

    def test_platform_type_label(self):
        assert platform_type_label("kaggle") == "AUTO"
        assert platform_type_label("google_colab") == "MANUAL"
        assert "MANUAL" in platform_type_label("huggingface")


class TestSafeSSHArgs:
    """Test SSH argument validation."""

    def test_valid_ssh_args(self):
        result = _safe_ssh_args("192.168.1.1", "ubuntu", "/dev/null")
        assert result is not None
        host, user, key = result
        assert host == "192.168.1.1"
        assert user == "ubuntu"

    def test_invalid_host(self):
        result = _safe_ssh_args("not valid host", "ubuntu", "/dev/null")
        assert result is None

    def test_invalid_user(self):
        result = _safe_ssh_args("192.168.1.1", "not valid user", "/dev/null")
        assert result is None


class TestColabSecretScanning:
    """Test the Colab handler's secret scanning."""

    def test_scan_detects_api_key(self):
        from handlers import GoogleColabHandler
        handler = GoogleColabHandler()
        warnings = handler._scan_for_secrets('api_key = "sk-1234567890abcdef1234567890"')
        assert len(warnings) > 0

    def test_scan_detects_aws_key(self):
        from handlers import GoogleColabHandler
        handler = GoogleColabHandler()
        warnings = handler._scan_for_secrets('AWS_KEY = "AKIAIOSFODNN7EXAMPLE"')
        assert len(warnings) > 0

    def test_scan_detects_private_key(self):
        from handlers import GoogleColabHandler
        handler = GoogleColabHandler()
        warnings = handler._scan_for_secrets("-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----")
        assert len(warnings) > 0

    def test_scan_clean_script(self):
        from handlers import GoogleColabHandler
        handler = GoogleColabHandler()
        warnings = handler._scan_for_secrets("import torch\nmodel = torch.nn.Linear(10, 5)")
        assert len(warnings) == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
