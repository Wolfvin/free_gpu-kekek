"""Tests for the vault module — encryption and secret redaction."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vault import (
    redact_text, scan_for_secrets, get_storage_mode,
    encrypt_credentials, decrypt_credentials, delete_credentials,
)


class TestSecretRedaction(unittest.TestCase):
    """Tests for secret redaction in text output."""

    def test_redact_google_api_key(self):
        text = "Using key AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q for auth"
        redacted = redact_text(text)
        self.assertNotIn("AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_redact_huggingface_token(self):
        text = "HF_TOKEN=hf_abc123def456ghi789jkl012mno345"
        redacted = redact_text(text)
        self.assertNotIn("hf_abc123def456ghi789jkl012mno345", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_redact_aws_key(self):
        text = "aws_access_key = AKIAIOSFODNN7EXAMPLE"
        redacted = redact_text(text)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", redacted)

    def test_redact_private_key(self):
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
        redacted = redact_text(text)
        self.assertNotIn("MIIEpAIBAAKCAQEA", redacted)
        self.assertIn("REDACTED", redacted)

    def test_redact_password(self):
        text = 'password = "mysecretpassword123"'
        redacted = redact_text(text)
        self.assertNotIn("mysecretpassword123", redacted)

    def test_redact_token(self):
        text = 'token = "abc123def456ghi789jkl012"'
        redacted = redact_text(text)
        self.assertNotIn("abc123def456ghi789jkl012", redacted)

    def test_no_redaction_needed(self):
        text = "Training epoch 5, loss=0.0234"
        redacted = redact_text(text)
        self.assertEqual(text, redacted)


class TestSecretScanning(unittest.TestCase):
    """Tests for the scan_for_secrets function."""

    def test_detect_api_key(self):
        content = 'api_key = "sk-1234567890abcdefghijklmnopqrstuvwxyz"'
        warnings = scan_for_secrets(content)
        self.assertTrue(len(warnings) > 0)

    def test_detect_private_key(self):
        content = "-----BEGIN PRIVATE KEY-----\nsomekeydata\n-----END PRIVATE KEY-----"
        warnings = scan_for_secrets(content)
        self.assertTrue(len(warnings) > 0)

    def test_clean_content(self):
        content = "import torch\nmodel = torch.nn.Linear(10, 10)\nprint('hello')"
        warnings = scan_for_secrets(content)
        self.assertEqual(len(warnings), 0)


class TestVaultStorageMode(unittest.TestCase):
    """Tests for vault storage mode detection."""

    def test_storage_mode_returns_string(self):
        mode = get_storage_mode()
        self.assertIsInstance(mode, str)
        self.assertTrue(len(mode) > 0)


class TestPlaintextDisabled(unittest.TestCase):
    """Verify that plaintext credential storage is disabled."""

    def test_decrypt_rejects_plaintext(self):
        """Plaintext credentials (no _storage marker, no enc: prefix) should be rejected."""
        stored = {"kaggle_key": "plaintext_value_here"}
        result = decrypt_credentials("kaggle", "test", stored)
        # Should return empty dict (rejected)
        self.assertEqual(result, {})

    def test_decrypt_handles_empty(self):
        result = decrypt_credentials("kaggle", "test", {})
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
