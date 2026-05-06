"""Tests for vault.py — credential encryption."""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vault import (
    encrypt_credentials,
    decrypt_credentials,
    delete_credentials,
    fernet_encrypt,
    fernet_decrypt,
    get_storage_mode,
    KEYRING_SERVICE,
)


class TestFernetEncryption:
    """Test Fernet symmetric encryption."""

    def test_encrypt_decrypt_roundtrip(self):
        """Encrypt then decrypt should return original value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            encrypted = fernet_encrypt("hello_secret", tmpdir)
            assert encrypted.startswith("enc:"), f"Expected enc: prefix, got {encrypted[:10]}"
            decrypted = fernet_decrypt(encrypted, tmpdir)
            assert decrypted == "hello_secret", f"Expected 'hello_secret', got '{decrypted}'"

    def test_decrypt_non_encrypted_passthrough(self):
        """Decrypting a non-encrypted string should pass through."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = fernet_decrypt("plaintext_value", tmpdir)
            assert result == "plaintext_value"

    def test_different_keys_fail(self):
        """Decrypting with a different master key should return empty string."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            encrypted = fernet_encrypt("secret_data", tmpdir1)
        with tempfile.TemporaryDirectory() as tmpdir2:
            # Different tmpdir = different master key
            result = fernet_decrypt(encrypted, tmpdir2)
            assert result == "", f"Expected empty string, got '{result}'"


class TestEncryptCredentials:
    """Test encrypt_credentials and decrypt_credentials."""

    def test_fernet_roundtrip(self):
        """Full encrypt/decrypt roundtrip with Fernet backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            creds = {"api_key": "sk-1234567890abcdef", "username": "testuser"}
            # Mock keyring to be unavailable
            with patch("vault._has_keyring", return_value=False):
                encrypted = encrypt_credentials("test_platform", "test_acct", creds, tmpdir)
                assert encrypted.get("_storage") == "fernet"
                assert "api_key" in encrypted
                assert encrypted["api_key"].startswith("enc:")

                decrypted = decrypt_credentials("test_platform", "test_acct", encrypted, tmpdir)
                assert decrypted["api_key"] == "sk-1234567890abcdef"
                assert decrypted["username"] == "testuser"

    def test_plaintext_fallback_disabled(self):
        """When no encryption is available, should raise RuntimeError."""
        with patch("vault._has_keyring", return_value=False):
            with patch("vault._has_cryptography", return_value=False):
                try:
                    encrypt_credentials("test_platform", "test_acct", {"key": "value"})
                    assert False, "Expected RuntimeError"
                except RuntimeError as e:
                    assert "no encryption available" in str(e).lower()

    def test_empty_credentials(self):
        """Empty credentials dict should return empty dict."""
        result = encrypt_credentials("platform", "acct", {})
        assert result == {}


class TestStorageMode:
    """Test get_storage_mode()."""

    def test_keyring_available(self):
        with patch("vault._has_keyring", return_value=True):
            mode = get_storage_mode()
            assert "Keychain" in mode or "keyring" in mode.lower()

    def test_no_encryption(self):
        with patch("vault._has_keyring", return_value=False):
            with patch("vault._has_cryptography", return_value=False):
                mode = get_storage_mode()
                assert "NO ENCRYPTION" in mode or "install" in mode.lower()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
