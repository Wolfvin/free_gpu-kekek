"""Credential encryption for free-gpu-trainer.

Uses OS keychain (keyring) when available, falls back to
Fernet symmetric encryption with a master key stored in .master_key.

Security model:
  - Preferred: OS keychain via `keyring` (credentials never touch disk as plaintext)
  - Fallback: Fernet encryption (cryptography library) with .master_key file
  - If neither is available: raises RuntimeError (refuses to save)

Usage in config.yaml:
  credentials:
    kaggle_username: "plain:myuser"
    kaggle_key: "enc:gAAAAABl..."
"""

import os
import json
import logging
import base64
from pathlib import Path
from typing import Optional

logger = logging.getLogger("fgt.vault")

# ── Encryption Constants ───────────────────────────────────────────

KEY_FILE = ".master_key"
KEYRING_SERVICE = "free-gpu-trainer"
ENC_PREFIX = "enc:"
PLAIN_PREFIX = "plain:"


def _has_keyring() -> bool:
    """Check if keyring library is available and functional.

    Uses a round-trip set/get/delete test instead of checking
    internal backend name strings, which are fragile and may
    change across keyring versions.
    """
    try:
        import keyring
        test_svc = "free-gpu-trainer-test"
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
        # Store all creds as a single JSON blob
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

def _get_or_create_key(config_dir: str = ".") -> bytes:
    """Get or create the master encryption key."""
    key_path = Path(config_dir) / KEY_FILE
    if key_path.exists():
        return base64.urlsafe_b64decode(key_path.read_text().strip())
    else:
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        key_path.write_text(key.decode())
        # Restrict permissions (owner only)
        os.chmod(key_path, 0o600)
        logger.info(f"Generated new master key: {key_path}")
        return base64.urlsafe_b64decode(key)


def fernet_encrypt(plaintext: str, config_dir: str = ".") -> str:
    """Encrypt a string using Fernet."""
    from cryptography.fernet import Fernet
    key = _get_or_create_key(config_dir)
    f = Fernet(base64.urlsafe_b64encode(key))
    return ENC_PREFIX + f.encrypt(plaintext.encode()).decode()


def fernet_decrypt(token: str, config_dir: str = ".") -> str:
    """Decrypt a Fernet-encrypted string."""
    from cryptography.fernet import Fernet, InvalidToken
    if not token.startswith(ENC_PREFIX):
        return token  # Not encrypted
    key = _get_or_create_key(config_dir)
    f = Fernet(base64.urlsafe_b64encode(key))
    try:
        return f.decrypt(token[len(ENC_PREFIX):].encode()).decode()
    except InvalidToken:
        logger.error("Failed to decrypt credential — master key may have changed")
        return ""


# ── Public API ─────────────────────────────────────────────────────

def encrypt_credentials(platform_key: str, account_name: str, creds: dict, config_dir: str = ".") -> dict:
    """Encrypt credentials for storage.

    Strategy:
    1. Try OS keyring (best — creds never on disk)
    2. Fallback: Fernet-encrypt values in config.yaml
    3. No encryption available: raises RuntimeError (refuses to save)

    Returns dict suitable for config.yaml (encrypted values if using Fernet,
    or empty dict if keyring handled it).
    """
    if not creds:
        return {}

    # Try keyring first
    if keyring_set(platform_key, account_name, creds):
        logger.debug(f"Credentials stored in OS keychain for {platform_key}/{account_name}")
        # Return special marker so we know to use keyring on load
        return {"_storage": "keyring"}

    # Try Fernet encryption
    if _has_cryptography():
        encrypted = {}
        for k, v in creds.items():
            if v:  # Don't encrypt empty strings
                encrypted[k] = fernet_encrypt(v, config_dir)
            else:
                encrypted[k] = v
        encrypted["_storage"] = "fernet"
        logger.debug(f"Credentials Fernet-encrypted for {platform_key}/{account_name}")
        return encrypted

    # No encryption available — REFUSE to save plaintext
    raise RuntimeError(
        f"Cannot save credentials for {platform_key}/{account_name}: "
        f"no encryption available. Install 'keyring' or 'cryptography' "
        f"to enable secure credential storage."
    )


def decrypt_credentials(platform_key: str, account_name: str, stored: dict, config_dir: str = ".") -> dict:
    """Decrypt credentials from config.yaml or keyring.

    Returns dict of {key: plaintext_value}.
    """
    if not stored:
        return {}

    storage = stored.get("_storage", "plain")

    # Keyring
    if storage == "keyring":
        creds = keyring_get(platform_key, account_name)
        if creds:
            return creds
        # Fallthrough — keyring may have been cleared
        logger.warning(f"Keyring empty for {platform_key}/{account_name}, trying stored values")
        # Remove _storage and try to use whatever's left
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

    # Plaintext
    return {k: v for k, v in stored.items() if k != "_storage"}


def delete_credentials(platform_key: str, account_name: str) -> None:
    """Delete credentials from keyring when account is removed."""
    keyring_delete(platform_key, account_name)


def get_storage_mode() -> str:
    """Get current credential storage mode for display."""
    if _has_keyring():
        return "OS Keychain (keyring)"
    if _has_cryptography():
        return "Fernet Encryption (.master_key)"
    return "❌ NO ENCRYPTION (install keyring or cryptography to save credentials)"
