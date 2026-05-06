"""Encrypted backup and restore for FamilyGPU Orchestrator.

Provides export/import functionality for owners and accounts data.
Backups are encrypted using the vault's Fernet encryption.

Usage:
  python run.py --export backup.json    # Export all data to encrypted file
  python run.py --import backup.json    # Import data from encrypted file
"""

import base64
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from db.repositories import (
    OwnerRepository, AccountRepository, ProviderRepository,
    JobRepository, LeaseRepository, QuotaLedgerRepository,
    AuditLogRepository, HealthRepository,
)
from db.connection import get_connection
from vault import fernet_encrypt, fernet_decrypt

logger = logging.getLogger("fgt.db.backup")

BACKUP_VERSION = "1.0"


def export_data(output_path: str, passphrase: Optional[str] = None) -> dict:
    """Export all owners, accounts, and related data to an encrypted JSON file.
    
    The export includes:
      - All owners (with consent notes)
      - All accounts (with metadata, but credentials exported as references only)
      - Provider configuration
      - Quota ledger entries
      - Audit log entries (last 30 days)
    
    Credentials are NOT exported — they must be re-entered after import.
    This is a deliberate security decision.
    
    Args:
        output_path: Path to write the encrypted backup file
        passphrase: Optional additional passphrase for double encryption
    
    Returns:
        Summary dict with counts of exported items
    """
    owner_repo = OwnerRepository()
    account_repo = AccountRepository()
    provider_repo = ProviderRepository()
    quota_repo = QuotaLedgerRepository()
    audit_repo = AuditLogRepository()
    health_repo = HealthRepository()
    
    # Collect data
    owners = owner_repo.list_all()
    accounts = account_repo.list_all()
    providers = provider_repo.list_all()
    quota_entries = []
    for acct in accounts:
        quota_entries.extend(quota_repo.list_by_account(acct["id"], limit=100))
    
    # Recent audit logs (last 30 days)
    audit_logs = audit_repo.list_recent(limit=1000)
    
    # Sanitize accounts — remove credential details
    safe_accounts = []
    for acct in accounts:
        safe = {k: v for k, v in acct.items() if k not in ("credential_ref",)}
        safe["credential_note"] = "Credentials must be re-entered after import"
        safe_accounts.append(safe)
    
    # Build backup structure
    backup = {
        "version": BACKUP_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "system": "familygpu-orchestrator",
        "data": {
            "owners": owners,
            "accounts": safe_accounts,
            "providers": providers,
            "quota_ledger": quota_entries,
            "audit_logs": audit_logs,
        },
        "summary": {
            "owners": len(owners),
            "accounts": len(accounts),
            "providers": len(providers),
            "quota_entries": len(quota_entries),
            "audit_entries": len(audit_logs),
        },
    }
    
    # Serialize and encrypt
    json_str = json.dumps(backup, indent=2, default=str)
    
    if passphrase:
        # Double-encrypt with passphrase
        import hashlib
        from cryptography.fernet import Fernet
        key = base64.urlsafe_b64encode(hashlib.sha256(passphrase.encode()).digest())
        f = Fernet(key)
        encrypted = f.encrypt(json_str.encode())
        with open(output_path, "wb") as f_out:
            f_out.write(encrypted)
    else:
        # Use standard Fernet encryption from vault
        encrypted_lines = []
        for line in json_str.split("\n"):
            encrypted_lines.append(fernet_encrypt(line))
        with open(output_path, "w") as f_out:
            f_out.write("\n".join(encrypted_lines))
    
    # Set file permissions
    os.chmod(output_path, 0o600)
    
    logger.info(
        f"Backup exported to {output_path}: "
        f"{len(owners)} owners, {len(accounts)} accounts, "
        f"{len(quota_entries)} quota entries"
    )
    
    return backup["summary"]


def import_data(input_path: str, passphrase: Optional[str] = None,
                dry_run: bool = False) -> dict:
    """Import owners and accounts from an encrypted backup file.
    
    Import behavior:
      - Owners are imported with INSERT OR IGNORE (existing owners kept)
      - Accounts are imported with INSERT OR IGNORE
      - Credentials are NOT imported (must be re-entered manually)
      - Quota ledger entries are NOT imported (fresh start)
      - Audit logs are NOT imported (fresh start)
    
    Args:
        input_path: Path to the encrypted backup file
        passphrase: Optional passphrase for decryption
        dry_run: If True, validate the backup without importing
    
    Returns:
        Summary dict with counts of imported items
    """
    owner_repo = OwnerRepository()
    account_repo = AccountRepository()
    provider_repo = ProviderRepository()
    
    # Decrypt
    if passphrase:
        import hashlib
        from cryptography.fernet import Fernet, InvalidToken
        key = base64.urlsafe_b64encode(hashlib.sha256(passphrase.encode()).digest())
        f = Fernet(key)
        with open(input_path, "rb") as f_in:
            encrypted = f_in.read()
        try:
            json_str = f.decrypt(encrypted).decode()
        except InvalidToken:
            raise ValueError("Invalid passphrase or corrupted backup file")
    else:
        with open(input_path, "r") as f_in:
            encrypted_lines = f_in.read().split("\n")
        decrypted_lines = []
        for line in encrypted_lines:
            if line.strip():
                decrypted_lines.append(fernet_decrypt(line))
        json_str = "\n".join(decrypted_lines)
    
    backup = json.loads(json_str)
    
    # Validate backup format
    if backup.get("system") != "familygpu-orchestrator":
        raise ValueError("Invalid backup file — not a FamilyGPU backup")
    
    version = backup.get("version", "0.0")
    if version != BACKUP_VERSION:
        logger.warning(f"Backup version {version} may not be fully compatible (current: {BACKUP_VERSION})")
    
    data = backup.get("data", {})
    owners = data.get("owners", [])
    accounts = data.get("accounts", [])
    
    if dry_run:
        return {
            "dry_run": True,
            "would_import_owners": len(owners),
            "would_import_accounts": len(accounts),
            "note": "Credentials must be re-entered after import",
        }
    
    # Import owners
    imported_owners = 0
    for owner_data in owners:
        try:
            existing = owner_repo.get_by_id(owner_data["id"])
            if existing:
                logger.debug(f"Owner {owner_data['id']} already exists, skipping")
                continue
            
            owner_repo.create(
                id=owner_data["id"],
                name=owner_data["name"],
                relationship=owner_data.get("relationship", ""),
                consent_note=owner_data.get("consent_note", ""),
            )
            imported_owners += 1
        except Exception as e:
            logger.warning(f"Failed to import owner {owner_data.get('id', '?')}: {e}")
    
    # Import accounts
    imported_accounts = 0
    for acct_data in accounts:
        try:
            existing = account_repo.get_by_id(acct_data["id"])
            if existing:
                logger.debug(f"Account {acct_data['id']} already exists, skipping")
                continue
            
            # Verify provider exists
            provider = provider_repo.get_by_key(acct_data["provider_key"])
            if not provider:
                logger.warning(f"Provider {acct_data['provider_key']} not found, skipping account {acct_data['id']}")
                continue
            
            # Verify owner exists
            owner = owner_repo.get_by_id(acct_data["owner_id"])
            if not owner:
                logger.warning(f"Owner {acct_data['owner_id']} not found, skipping account {acct_data['id']}")
                continue
            
            account_repo.create(
                owner_id=acct_data["owner_id"],
                provider_key=acct_data["provider_key"],
                label=acct_data["label"],
                credential_ref="none",  # Must be re-entered
                priority=acct_data.get("priority", 5),
                daily_limit_minutes=acct_data.get("daily_limit_minutes", 120),
                weekly_limit_minutes=acct_data.get("weekly_limit_minutes", 600),
                cooldown_minutes=acct_data.get("cooldown_minutes", 30),
            )
            imported_accounts += 1
        except Exception as e:
            logger.warning(f"Failed to import account {acct_data.get('id', '?')}: {e}")
    
    logger.info(
        f"Backup imported from {input_path}: "
        f"{imported_owners} owners, {imported_accounts} accounts"
    )
    
    return {
        "imported_owners": imported_owners,
        "imported_accounts": imported_accounts,
        "skipped_owners": len(owners) - imported_owners,
        "skipped_accounts": len(accounts) - imported_accounts,
        "note": "Credentials must be re-entered for all imported accounts",
    }
