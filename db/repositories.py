"""Repository classes for FamilyGPU Orchestrator.

Each repository provides CRUD operations for its corresponding
SQLite table. All use parameterized queries to prevent SQL injection.
Repositories use the shared connection from db.connection.
"""

import json
import uuid
import sqlite3
import logging
from datetime import datetime, timezone, date, timedelta
from typing import Optional

from db.connection import get_connection

logger = logging.getLogger("fgt.db.repo")


def _now() -> str:
    """Current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


# ── Owner Repository ────────────────────────────────────────────

class OwnerRepository:
    """CRUD for the owners table."""

    def create(self, id: str, name: str, relationship: str = "",
               consent_note: str = "") -> dict:
        """Create a new owner. Raises if ID already exists."""
        now = _now()
        conn = get_connection()
        conn.execute("""
            INSERT INTO owners (id, name, relationship, consent_note, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (id, name, relationship, consent_note, now, now))
        conn.commit()
        return self.get_by_id(id)

    def get_by_id(self, id: str) -> Optional[dict]:
        conn = get_connection()
        row = conn.execute("SELECT * FROM owners WHERE id = ?", (id,)).fetchone()
        return dict(row) if row else None

    def list_all(self) -> list[dict]:
        conn = get_connection()
        rows = conn.execute("SELECT * FROM owners ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def update(self, id: str, **kwargs) -> Optional[dict]:
        """Update owner fields. Pass keyword args like name=..., relationship=..."""
        if not kwargs:
            return self.get_by_id(id)
        kwargs["updated_at"] = _now()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [id]
        conn = get_connection()
        conn.execute(f"UPDATE owners SET {sets} WHERE id = ?", vals)
        conn.commit()
        return self.get_by_id(id)

    def delete(self, id: str) -> bool:
        conn = get_connection()
        cursor = conn.execute("DELETE FROM owners WHERE id = ?", (id,))
        conn.commit()
        return cursor.rowcount > 0


# ── Provider Repository ─────────────────────────────────────────

class ProviderRepository:
    """Read-only repository for providers (seeded by migrations)."""

    def get_by_key(self, key: str) -> Optional[dict]:
        conn = get_connection()
        row = conn.execute("SELECT * FROM providers WHERE key = ?", (key,)).fetchone()
        return dict(row) if row else None

    def list_all(self, enabled_only: bool = False) -> list[dict]:
        conn = get_connection()
        if enabled_only:
            rows = conn.execute(
                "SELECT * FROM providers WHERE enabled = 1 ORDER BY display_name"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM providers ORDER BY display_name"
            ).fetchall()
        return [dict(r) for r in rows]

    def list_by_class(self, provider_class: str) -> list[dict]:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM providers WHERE provider_class = ? AND enabled = 1 ORDER BY display_name",
            (provider_class,)
        ).fetchall()
        return [dict(r) for r in rows]

    def set_enabled(self, key: str, enabled: bool):
        conn = get_connection()
        conn.execute(
            "UPDATE providers SET enabled = ?, updated_at = ? WHERE key = ?",
            (1 if enabled else 0, _now(), key)
        )
        conn.commit()


# ── Account Repository ──────────────────────────────────────────

class AccountRepository:
    """CRUD for the accounts table."""

    def create(self, owner_id: str, provider_key: str, label: str,
               credential_ref: str, priority: int = 5,
               daily_limit_minutes: int = 120,
               weekly_limit_minutes: int = 600,
               cooldown_minutes: int = 30) -> Optional[dict]:
        """Create a new account linked to an owner and provider."""
        id = f"acct_{_uuid()[:8]}"
        now = _now()
        conn = get_connection()
        try:
            conn.execute("""
                INSERT INTO accounts
                    (id, owner_id, provider_key, label, credential_ref,
                     status, priority, daily_limit_minutes, weekly_limit_minutes,
                     cooldown_minutes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?)
            """, (id, owner_id, provider_key, label, credential_ref,
                  priority, daily_limit_minutes, weekly_limit_minutes,
                  cooldown_minutes, now, now))
            conn.commit()
            return self.get_by_id(id)
        except sqlite3.IntegrityError as e:
            logger.error(f"Account creation failed: {e}")
            return None

    def get_by_id(self, id: str) -> Optional[dict]:
        conn = get_connection()
        row = conn.execute("SELECT * FROM accounts WHERE id = ?", (id,)).fetchone()
        return dict(row) if row else None

    def list_all(self, status: Optional[str] = None) -> list[dict]:
        conn = get_connection()
        if status:
            rows = conn.execute(
                "SELECT * FROM accounts WHERE status = ? ORDER BY provider_key, label",
                (status,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM accounts ORDER BY provider_key, label"
            ).fetchall()
        return [dict(r) for r in rows]

    def list_by_owner(self, owner_id: str) -> list[dict]:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM accounts WHERE owner_id = ? ORDER BY provider_key, label",
            (owner_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_by_provider(self, provider_key: str) -> list[dict]:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM accounts WHERE provider_key = ? ORDER BY label",
            (provider_key,)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_available(self) -> list[dict]:
        """List accounts that are active, not in cooldown, not busy."""
        conn = get_connection()
        now = _now()
        rows = conn.execute("""
            SELECT * FROM accounts
            WHERE status = 'active'
              AND (cooldown_until IS NULL OR cooldown_until <= ?)
            ORDER BY priority DESC, provider_key, label
        """, (now,)).fetchall()
        return [dict(r) for r in rows]

    def update(self, id: str, **kwargs) -> Optional[dict]:
        """Update account fields. Pass keyword args like status=..., priority=..."""
        if not kwargs:
            return self.get_by_id(id)
        kwargs["updated_at"] = _now()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [id]
        conn = get_connection()
        conn.execute(f"UPDATE accounts SET {sets} WHERE id = ?", vals)
        conn.commit()
        return self.get_by_id(id)

    def set_status(self, id: str, status: str):
        """Set account status (active, disabled, cooldown, etc.)."""
        self.update(id, status=status)

    def set_cooldown(self, id: str, cooldown_minutes: int):
        """Put an account into cooldown."""
        until = (datetime.now(timezone.utc) + timedelta(minutes=cooldown_minutes)).isoformat()
        self.update(id, status="cooldown", cooldown_until=until)

    def clear_cooldown(self, id: str):
        """Clear cooldown if the cooldown period has passed."""
        acct = self.get_by_id(id)
        if not acct:
            return
        if acct.get("cooldown_until") and acct["cooldown_until"] <= _now():
            self.update(id, status="active", cooldown_until=None)

    def record_use(self, id: str, error: Optional[str] = None):
        """Record that an account was just used."""
        updates = {"last_used_at": _now()}
        if error:
            updates["last_error"] = error
            updates["last_error_at"] = _now()
        self.update(id, **updates)

    def disable(self, id: str):
        """Disable an account."""
        self.set_status(id, "disabled")

    def enable(self, id: str):
        """Re-enable a disabled account."""
        self.set_status(id, "active")

    def delete(self, id: str) -> bool:
        conn = get_connection()
        cursor = conn.execute("DELETE FROM accounts WHERE id = ?", (id,))
        conn.commit()
        return cursor.rowcount > 0


# ── Job Repository ──────────────────────────────────────────────

class JobRepository:
    """CRUD for the jobs table."""

    def create(self, name: str, gpu_profile: str, max_runtime_minutes: int,
               checkpoint_uri: str, entrypoint: str = "train.py",
               priority: str = "normal", args: Optional[dict] = None,
               allow_providers: Optional[list[str]] = None,
               deny_providers: Optional[list[str]] = None,
               created_by: str = "user") -> Optional[dict]:
        id = f"job_{_uuid()[:8]}"
        now = _now()
        conn = get_connection()
        try:
            conn.execute("""
                INSERT INTO jobs
                    (id, name, status, gpu_profile, priority,
                     max_runtime_minutes, checkpoint_uri, entrypoint,
                     args_json, allow_providers_json, deny_providers_json,
                     created_by, created_at, updated_at)
                VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                id, name, gpu_profile, priority,
                max_runtime_minutes, checkpoint_uri, entrypoint,
                json.dumps(args) if args else None,
                json.dumps(allow_providers) if allow_providers else None,
                json.dumps(deny_providers) if deny_providers else None,
                created_by, now, now,
            ))
            conn.commit()
            return self.get_by_id(id)
        except sqlite3.IntegrityError as e:
            logger.error(f"Job creation failed: {e}")
            return None

    def get_by_id(self, id: str) -> Optional[dict]:
        conn = get_connection()
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (id,)).fetchone()
        if row:
            d = dict(row)
            d["args"] = json.loads(d["args_json"]) if d.get("args_json") else None
            d["allow_providers"] = json.loads(d["allow_providers_json"]) if d.get("allow_providers_json") else None
            d["deny_providers"] = json.loads(d["deny_providers_json"]) if d.get("deny_providers_json") else None
            return d
        return None

    def list_all(self, status: Optional[str] = None) -> list[dict]:
        conn = get_connection()
        if status:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC",
                (status,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def list_queued(self) -> list[dict]:
        return self.list_all(status="queued")

    def list_active(self) -> list[dict]:
        """Jobs that are running or starting."""
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status IN ('running', 'starting') ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def update(self, id: str, **kwargs) -> Optional[dict]:
        if not kwargs:
            return self.get_by_id(id)
        kwargs["updated_at"] = _now()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [id]
        conn = get_connection()
        conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?", vals)
        conn.commit()
        return self.get_by_id(id)

    def mark_started(self, id: str):
        """Mark a job as running."""
        self.update(id, status="running", started_at=_now())

    def mark_completed(self, id: str):
        """Mark a job as completed."""
        self.update(id, status="completed", completed_at=_now())

    def mark_failed(self, id: str, reason: str):
        """Mark a job as failed."""
        self.update(id, status="failed", failure_reason=reason, completed_at=_now())

    def cancel(self, id: str):
        """Cancel a job."""
        self.update(id, status="cancelled", completed_at=_now())


# ── Lease Repository ────────────────────────────────────────────

class LeaseRepository:
    """CRUD for the leases table."""

    def create(self, job_id: str, account_id: str, provider_key: str,
               expires_at: Optional[str] = None) -> Optional[dict]:
        id = f"lease_{_uuid()[:8]}"
        now = _now()
        conn = get_connection()
        try:
            conn.execute("""
                INSERT INTO leases
                    (id, job_id, account_id, provider_key, status,
                     started_at, heartbeat_at, expires_at,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
            """, (id, job_id, account_id, provider_key, now, now, expires_at, now, now))
            conn.commit()
            return self.get_by_id(id)
        except sqlite3.IntegrityError as e:
            logger.error(f"Lease creation failed: {e}")
            return None

    def get_by_id(self, id: str) -> Optional[dict]:
        conn = get_connection()
        row = conn.execute("SELECT * FROM leases WHERE id = ?", (id,)).fetchone()
        return dict(row) if row else None

    def list_all(self, status: Optional[str] = None) -> list[dict]:
        conn = get_connection()
        if status:
            rows = conn.execute(
                "SELECT * FROM leases WHERE status = ? ORDER BY created_at DESC",
                (status,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM leases ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def list_active(self) -> list[dict]:
        """Leases that are currently running."""
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM leases WHERE status IN ('pending', 'starting', 'running') ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def list_by_account(self, account_id: str, active_only: bool = False) -> list[dict]:
        conn = get_connection()
        if active_only:
            rows = conn.execute(
                "SELECT * FROM leases WHERE account_id = ? AND status IN ('pending', 'starting', 'running')",
                (account_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM leases WHERE account_id = ? ORDER BY created_at DESC",
                (account_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def is_account_busy(self, account_id: str) -> bool:
        """Check if an account has any active lease."""
        leases = self.list_by_account(account_id, active_only=True)
        return len(leases) > 0

    def get_active_for_job(self, job_id: str) -> Optional[dict]:
        """Get the active lease for a job, if any."""
        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM leases WHERE job_id = ? AND status IN ('pending', 'starting', 'running')",
            (job_id,)
        ).fetchone()
        return dict(row) if row else None

    def update(self, id: str, **kwargs) -> Optional[dict]:
        if not kwargs:
            return self.get_by_id(id)
        kwargs["updated_at"] = _now()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [id]
        conn = get_connection()
        conn.execute(f"UPDATE leases SET {sets} WHERE id = ?", vals)
        conn.commit()
        return self.get_by_id(id)

    def heartbeat(self, id: str):
        """Update the heartbeat timestamp for a lease."""
        self.update(id, heartbeat_at=_now())

    def mark_running(self, id: str, remote_job_ref: Optional[str] = None):
        """Mark a lease as running."""
        updates = {"status": "running", "started_at": _now()}
        if remote_job_ref:
            updates["remote_job_ref"] = remote_job_ref
        self.update(id, **updates)

    def mark_completed(self, id: str, runtime_minutes: int = 0):
        """Mark a lease as completed."""
        self.update(id, status="completed", ended_at=_now(), runtime_minutes=runtime_minutes)

    def mark_failed(self, id: str, reason: str):
        """Mark a lease as failed."""
        self.update(id, status="failed", ended_at=_now(), failure_reason=reason)

    def mark_expired(self, id: str):
        """Mark a lease as expired."""
        self.update(id, status="expired", ended_at=_now())

    def mark_checkpointing(self, id: str):
        """Mark a lease as checkpointing (before expiry)."""
        self.update(id, status="checkpointing")

    def cancel(self, id: str):
        """Cancel a lease."""
        self.update(id, status="cancelled", ended_at=_now())


# ── Quota Ledger Repository ─────────────────────────────────────

class QuotaLedgerRepository:
    """CRUD for the quota_ledger table."""

    def record_usage(self, account_id: str, provider_key: str,
                     used_minutes: int, job_id: Optional[str] = None,
                     lease_id: Optional[str] = None,
                     started_at: Optional[str] = None,
                     ended_at: Optional[str] = None) -> dict:
        """Record usage for an account."""
        id = f"ql_{_uuid()[:8]}"
        now = _now()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        conn = get_connection()
        conn.execute("""
            INSERT INTO quota_ledger
                (id, account_id, job_id, lease_id, provider_key,
                 used_minutes, usage_date, started_at, ended_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (id, account_id, job_id, lease_id, provider_key,
              used_minutes, today, started_at or now, ended_at or now, now))
        conn.commit()
        return {"id": id, "account_id": account_id, "used_minutes": used_minutes, "usage_date": today}

    def get_daily_usage(self, account_id: str, usage_date: Optional[str] = None) -> int:
        """Get total minutes used by an account on a specific date."""
        if not usage_date:
            usage_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        conn = get_connection()
        row = conn.execute("""
            SELECT COALESCE(SUM(used_minutes), 0) as total
            FROM quota_ledger
            WHERE account_id = ? AND usage_date = ?
        """, (account_id, usage_date)).fetchone()
        return row["total"] if row else 0

    def get_weekly_usage(self, account_id: str) -> int:
        """Get total minutes used by an account in the current week."""
        today = datetime.now(timezone.utc).date()
        # ISO week: Monday is day 1
        monday = today - timedelta(days=today.weekday())
        week_start = monday.isoformat()
        conn = get_connection()
        row = conn.execute("""
            SELECT COALESCE(SUM(used_minutes), 0) as total
            FROM quota_ledger
            WHERE account_id = ? AND usage_date >= ?
        """, (account_id, week_start)).fetchone()
        return row["total"] if row else 0

    def remaining_daily(self, account_id: str, daily_limit_minutes: int) -> int:
        """Remaining daily quota in minutes."""
        used = self.get_daily_usage(account_id)
        return max(0, daily_limit_minutes - used)

    def remaining_weekly(self, account_id: str, weekly_limit_minutes: int) -> int:
        """Remaining weekly quota in minutes."""
        used = self.get_weekly_usage(account_id)
        return max(0, weekly_limit_minutes - used)

    def list_by_account(self, account_id: str, limit: int = 50) -> list[dict]:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM quota_ledger WHERE account_id = ? ORDER BY usage_date DESC, created_at DESC LIMIT ?",
            (account_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_by_date(self, usage_date: str) -> list[dict]:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM quota_ledger WHERE usage_date = ? ORDER BY account_id",
            (usage_date,)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_by_provider(self, provider_key: str, limit: int = 50) -> list[dict]:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM quota_ledger WHERE provider_key = ? ORDER BY usage_date DESC LIMIT ?",
            (provider_key, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_usage_summary(self, group_by: str = "provider") -> list[dict]:
        """Get usage summary grouped by provider or owner.

        group_by: 'provider' or 'owner'
        """
        conn = get_connection()
        if group_by == "provider":
            rows = conn.execute("""
                SELECT provider_key,
                       COUNT(*) as entry_count,
                       SUM(used_minutes) as total_minutes
                FROM quota_ledger
                GROUP BY provider_key
                ORDER BY total_minutes DESC
            """).fetchall()
        elif group_by == "owner":
            rows = conn.execute("""
                SELECT a.owner_id,
                       COUNT(*) as entry_count,
                       SUM(ql.used_minutes) as total_minutes
                FROM quota_ledger ql
                JOIN accounts a ON ql.account_id = a.id
                GROUP BY a.owner_id
                ORDER BY total_minutes DESC
            """).fetchall()
        else:
            rows = []
        return [dict(r) for r in rows]


# ── Audit Log Repository ───────────────────────────────────────

class AuditLogRepository:
    """Append-only log for tracking important actions."""

    def log(self, action: str, entity_type: str, entity_id: str = "",
            actor: str = "system", message: str = "",
            metadata: Optional[dict] = None):
        """Record an audit log entry."""
        id = f"audit_{_uuid()[:8]}"
        now = _now()
        conn = get_connection()
        conn.execute("""
            INSERT INTO audit_logs
                (id, actor, action, entity_type, entity_id, message,
                 metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            id, actor, action, entity_type, entity_id, message,
            json.dumps(metadata) if metadata else None, now,
        ))
        conn.commit()

    def list_recent(self, limit: int = 100) -> list[dict]:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d["metadata_json"]) if d.get("metadata_json") else None
            result.append(d)
        return result

    def list_by_entity(self, entity_type: str, entity_id: str) -> list[dict]:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM audit_logs WHERE entity_type = ? AND entity_id = ? ORDER BY created_at DESC",
            (entity_type, entity_id)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_by_action(self, action: str, limit: int = 50) -> list[dict]:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM audit_logs WHERE action = ? ORDER BY created_at DESC LIMIT ?",
            (action, limit)
        ).fetchall()
        return [dict(r) for r in rows]


# ── Health Repository ───────────────────────────────────────────

class HealthRepository:
    """Track provider health check results."""

    def record(self, account_id: str, provider_key: str,
               status: str, message: str = "") -> dict:
        id = f"health_{_uuid()[:8]}"
        now = _now()
        conn = get_connection()
        conn.execute("""
            INSERT INTO provider_health
                (id, account_id, provider_key, status, message, checked_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (id, account_id, provider_key, status, message, now))
        conn.commit()
        # Also update account's last_health_status
        conn.execute(
            "UPDATE accounts SET last_health_status = ?, updated_at = ? WHERE id = ?",
            (status, now, account_id)
        )
        conn.commit()
        return {"id": id, "status": status}

    def get_latest(self, account_id: str) -> Optional[dict]:
        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM provider_health WHERE account_id = ? ORDER BY checked_at DESC LIMIT 1",
            (account_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_by_provider(self, provider_key: str, limit: int = 20) -> list[dict]:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM provider_health WHERE provider_key = ? ORDER BY checked_at DESC LIMIT ?",
            (provider_key, limit)
        ).fetchall()
        return [dict(r) for r in rows]


# ── Secret Metadata Repository ──────────────────────────────────

class SecretMetadataRepository:
    """Track credential metadata (not the secrets themselves)."""

    def record(self, secret_type: str, storage_backend: str,
               provider_key: Optional[str] = None,
               account_id: Optional[str] = None) -> dict:
        id = f"secmeta_{_uuid()[:8]}"
        now = _now()
        conn = get_connection()
        conn.execute("""
            INSERT INTO secrets_metadata
                (id, secret_type, provider_key, account_id,
                 storage_backend, created_at, rotated_at, last_used_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
        """, (id, secret_type, provider_key, account_id, storage_backend, now))
        conn.commit()
        return {"id": id, "storage_backend": storage_backend}

    def record_rotation(self, id: str):
        conn = get_connection()
        conn.execute(
            "UPDATE secrets_metadata SET rotated_at = ? WHERE id = ?",
            (_now(), id)
        )
        conn.commit()

    def record_usage(self, id: str):
        conn = get_connection()
        conn.execute(
            "UPDATE secrets_metadata SET last_used_at = ? WHERE id = ?",
            (_now(), id)
        )
        conn.commit()

    def list_by_account(self, account_id: str) -> list[dict]:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM secrets_metadata WHERE account_id = ? ORDER BY created_at DESC",
            (account_id,)
        ).fetchall()
        return [dict(r) for r in rows]
