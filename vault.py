"""Credential encryption and secret redaction for FamilyGPU Orchestrator.

Uses OS keychain (keyring) when available, falls back to
Fernet symmetric encryption with a master key stored in ~/.familygpu/.master_key.

Security model:
  - Preferred: OS keychain via `keyring` (credentials never touch disk as plaintext)
  - Fallback: Fernet encryption (cryptography library) with ~/.familygpu/.master_key
  - Plaintext is DISABLED — the system will refuse to save or load plaintext credentials
  - All log output is passed through redaction to prevent credential leaks

Usage:
  vault.encrypt_credentials(platform_key, account_name, creds)
  vault.decrypt_credentials(platform_key, account_name, stored_creds)
  vault.redact_text(text_with_possible_secrets)
"""

import os
import re
import json
import shutil
import logging
import base64
from pathlib import Path
from typing import Optional

logger = logging.getLogger("fgt.vault")

# ── Encryption Constants ───────────────────────────────────────────

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".familygpu")
KEY_FILE = ".master_key"
KEYRING_SERVICE = "familygpu-orchestrator"
ENC_PREFIX = "enc:"

# ── Secret Redaction Patterns ──────────────────────────────────────
# Patterns for detecting and redacting secrets in text output.
# Used for logs, notebook content, and training scripts.

SECRET_PATTERNS = [
    # Google API keys
    (re.compile(r"AIza[0-9A-Za-z\-_]{35}"), "AIza[REDACTED]"),
    # HuggingFace tokens
    (re.compile(r"hf_[0-9A-Za-z]{20,}"), "hf_[REDACTED]"),
    # AWS Access Key IDs
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA[REDACTED]"),
    # AWS Secret Access Keys (base64-ish, 40 chars after key/secret)
    (re.compile(r"(?:aws_secret_access_key|AWS_SECRET_ACCESS_KEY)\s*[=:]\s*[A-Za-z0-9/+=]{40}"),
     "AWS_SECRET=[REDACTED]"),
    # Private keys (DOTALL flag to match across newlines)
    (re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----.*?-----END (?:RSA |EC |DSA )?PRIVATE KEY-----", re.DOTALL),
     "-----BEGIN PRIVATE KEY [REDACTED]-----"),
    # Generic password/secret assignments
    (re.compile(r"(password|passwd|secret)\s*[=:]\s*['\"][^'\"]{8,}['\"]", re.IGNORECASE),
     r"\1=[REDACTED]"),
    # Generic token assignments
    (re.compile(r"(token|bearer|auth_key|api_key)\s*[=:]\s*['\"][a-zA-Z0-9_\-]{20,}['\"]", re.IGNORECASE),
     r"\1=[REDACTED]"),
    # Kaggle keys (32-char hex)
    (re.compile(r"[0-9a-f]{32}"), "[KAGGLE_KEY_REDACTED]"),
    # Database URLs with credentials
    (re.compile(r"(postgres|mysql|mongodb|redis)://[^:]+:[^@]+@"), r"\1://[REDACTED]@"),
    # SSH private key paths in config
    (re.compile(r"(ssh_key|private_key_path)\s*[=:]\s*['\"]?[/~][^'\"]+['\"]?", re.IGNORECASE),
     r"\1=[REDACTED]"),
]


def redact_text(text: str) -> str:
    """Redact known secret patterns from text.

    Apply this to any text before logging, embedding in notebooks,
    or sending to external systems.
    """
    if not text:
        return text
    result = text
    for pattern, replacement in SECRET_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def scan_for_secrets(content: str) -> list[str]:
    """Scan content for potential secrets.

    Returns a list of warning messages (empty = no secrets found).
    """
    warnings = []
    for pattern, replacement in SECRET_PATTERNS:
        if pattern.search(content):
            # Extract a label from the replacement
            label = replacement.replace("[REDACTED]", "").strip("=_ ")
            warnings.append(f"Potential {label} detected" if label else "Potential secret detected")
    return list(set(warnings))  # Deduplicate


# ── Keyring Check ─────────────────────────────────────────────────

def _has_keyring() -> bool:
    """Check if keyring library is available and functional.

    Uses a round-trip set/get/delete test instead of checking
    internal backend name strings, which are fragile and may
    change across keyring versions.
    """
    try:
        import keyring
        test_svc = "familygpu-test"
        test_usr = "_probe_"
        keyring.set_password(test_svc, test_usr, "1")
        got = keyring.get_password(test_svc, test_usr)
        keyring.delete_password(test_svc, test_usr)
        return got == "1"
    except Exception:
        return False


def _has_cryptography() -> bool:
    """Check if cryptography library is available."""
    try:
        from cryptography.fernet import Fernet
        return True
    except ImportError:
        return False


# ── Keyring Storage ────────────────────────────────────────────────

def keyring_set(platform_key: str, account_name: str, creds: dict) -> bool:
    """Store credentials in OS keychain."""
    if not _has_keyring():
        return False
    try:
        import keyring
        key = f"{platform_key}:{account_name}"
        payload = json.dumps(creds)
        keyring.set_password(KEYRING_SERVICE, key, payload)
        return True
    except Exception as e:
        logger.debug(f"keyring set failed: {e}")
        return False


def keyring_get(platform_key: str, account_name: str) -> Optional[dict]:
    """Retrieve credentials from OS keychain."""
    if not _has_keyring():
        return None
    try:
        import keyring
        key = f"{platform_key}:{account_name}"
        payload = keyring.get_password(KEYRING_SERVICE, key)
        if payload:
            return json.loads(payload)
    except Exception as e:
        logger.debug(f"keyring get failed: {e}")
    return None


def keyring_delete(platform_key: str, account_name: str) -> bool:
    """Delete credentials from OS keychain."""
    if not _has_keyring():
        return False
    try:
        import keyring
        key = f"{platform_key}:{account_name}"
        keyring.delete_password(KEYRING_SERVICE, key)
        return True
    except Exception:
        return False


# ── Fernet Encryption ──────────────────────────────────────────────

def _get_key_path() -> Path:
    """Get the master key path in ~/.familygpu/.

    Creates the config directory with restricted permissions if needed.
    Migrates an existing .master_key from the project root.
    """
    config_dir = Path(os.path.expanduser("~")) / ".familygpu"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(str(config_dir), 0o700)

    key_path = config_dir / KEY_FILE

    # Migration: move .master_key from project root to ~/.familygpu/
    old_key = Path(__file__).parent / ".master_key"
    if old_key.exists() and not key_path.exists():
        shutil.copy2(str(old_key), str(key_path))
        os.chmod(str(key_path), 0o600)
        logger.info(f"Migrated master key: {old_key} -> {key_path}")

    return key_path


def _get_or_create_key(config_dir: str = None) -> bytes:
    """Get or create the master encryption key."""
    # config_dir is kept for API compatibility but now defaults to ~/.familygpu/
    if config_dir is None or config_dir == ".":
        key_path = _get_key_path()
    else:
        key_path = Path(config_dir) / KEY_FILE
    if key_path.exists():
        return base64.urlsafe_b64decode(key_path.read_text().strip())
    else:
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(key.decode())
        os.chmod(str(key_path), 0o600)
        logger.info(f"Generated new master key: {key_path}")
        return base64.urlsafe_b64decode(key)


def fernet_encrypt(plaintext: str, config_dir: str = None) -> str:
    """Encrypt a string using Fernet."""
    from cryptography.fernet import Fernet
    key = _get_or_create_key(config_dir)
    f = Fernet(base64.urlsafe_b64encode(key))
    return ENC_PREFIX + f.encrypt(plaintext.encode()).decode()


def fernet_decrypt(token: str, config_dir: str = None) -> str:
    """Decrypt a Fernet-encrypted string."""
    from cryptography.fernet import Fernet, InvalidToken
    if not token.startswith(ENC_PREFIX):
        return token
    key = _get_or_create_key(config_dir)
    f = Fernet(base64.urlsafe_b64encode(key))
    try:
        return f.decrypt(token[len(ENC_PREFIX):].encode()).decode()
    except InvalidToken:
        logger.error("Failed to decrypt credential — master key may have changed")
        return ""


# ── Public API ─────────────────────────────────────────────────────

def encrypt_credentials(platform_key: str, account_name: str, creds: dict,
                        config_dir: str = None) -> dict:
    """Encrypt credentials for storage.

    Strategy:
    1. Try OS keyring (best — creds never on disk)
    2. Fallback: Fernet-encrypt values in config
    3. No encryption available: raises RuntimeError (refuses to save)

    IMPORTANT: Plaintext storage is DISABLED.
    The system will NEVER save credentials as plaintext.
    """
    if not creds:
        return {}

    # Try keyring first
    if keyring_set(platform_key, account_name, creds):
        logger.debug(f"Credentials stored in OS keychain for {platform_key}/{account_name}")
        return {"_storage": "keyring"}

    # Try Fernet encryption
    if _has_cryptography():
        encrypted = {}
        for k, v in creds.items():
            if v:
                encrypted[k] = fernet_encrypt(v, config_dir)
            else:
                encrypted[k] = v
        encrypted["_storage"] = "fernet"
        logger.debug(f"Credentials Fernet-encrypted for {platform_key}/{account_name}")
        return encrypted

    # NO PLAINTEXT FALLBACK — raise error
    raise RuntimeError(
        f"Cannot save credentials for {platform_key}/{account_name}: "
        f"no encryption available. Install 'keyring' or 'cryptography' "
        f"to enable secure credential storage. "
        f"Plaintext storage is DISABLED for security."
    )


def decrypt_credentials(platform_key: str, account_name: str, stored: dict,
                        config_dir: str = None) -> dict:
    """Decrypt credentials from storage.

    Returns dict of {key: plaintext_value}.

    IMPORTANT: If stored credentials are in plaintext format (no _storage marker,
    no enc: prefix), they will be REJECTED with a warning.
    This prevents accidental loading of plaintext credentials.
    """
    if not stored:
        return {}

    storage = stored.get("_storage", "")

    # Keyring
    if storage == "keyring":
        creds = keyring_get(platform_key, account_name)
        if creds:
            return creds
        logger.warning(f"Keyring empty for {platform_key}/{account_name}, trying stored values")
        stored = {k: v for k, v in stored.items() if k != "_storage"}
        if not stored:
            return {}

    # Fernet
    if storage == "fernet" or any(
        isinstance(v, str) and v.startswith(ENC_PREFIX) for v in stored.values()
    ):
        decrypted = {}
        for k, v in stored.items():
            if k == "_storage":
                continue
            if isinstance(v, str) and v.startswith(ENC_PREFIX):
                decrypted[k] = fernet_decrypt(v, config_dir)
            else:
                decrypted[k] = v
        return decrypted

    # Plaintext — REJECT
    logger.error(
        f"SECURITY: Plaintext credentials detected for {platform_key}/{account_name}. "
        f"Plaintext storage is disabled. Please re-enter credentials to encrypt them."
    )
    return {}


def delete_credentials(platform_key: str, account_name: str) -> None:
    """Delete credentials from keyring when account is removed."""
    keyring_delete(platform_key, account_name)


def get_storage_mode() -> str:
    """Get current credential storage mode for display."""
    if _has_keyring():
        return "OS Keychain (keyring)"
    if _has_cryptography():
        return "Fernet Encryption (~/.familygpu/.master_key)"
    return "NO ENCRYPTION AVAILABLE (install keyring or cryptography to save credentials)"


def get_credential_status(credential_ref: str) -> str:
    """Get display status for a credential reference.

    Returns: 'configured', 'missing', or 'invalid'
    """
    if not credential_ref or credential_ref == "none":
        return "missing"
    if credential_ref.startswith("keyring:"):
        # Verify keyring has the data
        parts = credential_ref.split(":", 2)
        if len(parts) >= 3:
            creds = keyring_get(parts[1], parts[2])
            return "configured" if creds else "invalid"
    if credential_ref.startswith("fernet:"):
        return "configured"
    return "missing"
