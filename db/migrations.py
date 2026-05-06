"""Schema migrations for FamilyGPU Orchestrator.

All migrations are idempotent — safe to run multiple times.
Uses a _migrations table to track which migrations have been applied.
"""

import sqlite3
import logging
from datetime import datetime, timezone

logger = logging.getLogger("fgt.db.migrations")

# ── Migration Tracking ──────────────────────────────────────────

def _ensure_migration_table(conn: sqlite3.Connection):
    """Create the migration tracking table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            applied_at TEXT NOT NULL
        )
    """)
    conn.commit()


def _is_applied(conn: sqlite3.Connection, name: str) -> bool:
    """Check if a migration has already been applied."""
    row = conn.execute(
        "SELECT 1 FROM _migrations WHERE name = ?", (name,)
    ).fetchone()
    return row is not None


def _mark_applied(conn: sqlite3.Connection, name: str):
    """Mark a migration as applied."""
    conn.execute(
        "INSERT INTO _migrations (name, applied_at) VALUES (?, ?)",
        (name, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()


def _apply_migration(conn: sqlite3.Connection, name: str, sql: str):
    """Apply a single migration if not already applied."""
    if _is_applied(conn, name):
        logger.debug(f"Migration already applied: {name}")
        return

    logger.info(f"Applying migration: {name}")
    conn.executescript(sql)
    _mark_applied(conn, name)
    logger.info(f"Migration applied: {name}")


# ── Migration Definitions ───────────────────────────────────────

MIGRATIONS = [
    ("001_owners", """
        CREATE TABLE IF NOT EXISTS owners (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            relationship TEXT,
            consent_note TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """),

    ("002_providers", """
        CREATE TABLE IF NOT EXISTS providers (
            key TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            provider_class TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            supports_gpu INTEGER NOT NULL DEFAULT 1,
            supports_checkpoint INTEGER NOT NULL DEFAULT 1,
            default_session_limit_minutes INTEGER,
            default_cooldown_minutes INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """),

    ("003_accounts", """
        CREATE TABLE IF NOT EXISTS accounts (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            provider_key TEXT NOT NULL,
            label TEXT NOT NULL,
            credential_ref TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            priority INTEGER NOT NULL DEFAULT 5,
            daily_limit_minutes INTEGER NOT NULL DEFAULT 120,
            weekly_limit_minutes INTEGER NOT NULL DEFAULT 600,
            cooldown_minutes INTEGER NOT NULL DEFAULT 30,
            cooldown_until TEXT,
            last_used_at TEXT,
            last_health_status TEXT,
            last_error TEXT,
            last_error_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,

            FOREIGN KEY (owner_id) REFERENCES owners(id),
            FOREIGN KEY (provider_key) REFERENCES providers(key)
        );

        CREATE INDEX IF NOT EXISTS idx_accounts_provider ON accounts(provider_key);
        CREATE INDEX IF NOT EXISTS idx_accounts_owner ON accounts(owner_id);
        CREATE INDEX IF NOT EXISTS idx_accounts_status ON accounts(status);
    """),

    ("004_jobs", """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            gpu_profile TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'normal',
            max_runtime_minutes INTEGER NOT NULL,
            checkpoint_uri TEXT NOT NULL,
            entrypoint TEXT NOT NULL,
            args_json TEXT,
            allow_providers_json TEXT,
            deny_providers_json TEXT,
            created_by TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            failure_reason TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
    """),

    ("005_leases", """
        CREATE TABLE IF NOT EXISTS leases (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            account_id TEXT NOT NULL,
            provider_key TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            started_at TEXT,
            heartbeat_at TEXT,
            expires_at TEXT,
            ended_at TEXT,
            runtime_minutes INTEGER DEFAULT 0,
            remote_job_ref TEXT,
            log_uri TEXT,
            failure_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,

            FOREIGN KEY (job_id) REFERENCES jobs(id),
            FOREIGN KEY (account_id) REFERENCES accounts(id)
        );

        CREATE INDEX IF NOT EXISTS idx_leases_status ON leases(status);
        CREATE INDEX IF NOT EXISTS idx_leases_account_status ON leases(account_id, status);
    """),

    ("006_quota_ledger", """
        CREATE TABLE IF NOT EXISTS quota_ledger (
            id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            job_id TEXT,
            lease_id TEXT,
            provider_key TEXT NOT NULL,
            used_minutes INTEGER NOT NULL,
            usage_date TEXT NOT NULL,
            started_at TEXT,
            ended_at TEXT,
            created_at TEXT NOT NULL,

            FOREIGN KEY (account_id) REFERENCES accounts(id),
            FOREIGN KEY (job_id) REFERENCES jobs(id),
            FOREIGN KEY (lease_id) REFERENCES leases(id)
        );

        CREATE INDEX IF NOT EXISTS idx_quota_account_date ON quota_ledger(account_id, usage_date);
    """),

    ("007_provider_health", """
        CREATE TABLE IF NOT EXISTS provider_health (
            id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            provider_key TEXT NOT NULL,
            status TEXT NOT NULL,
            message TEXT,
            checked_at TEXT NOT NULL,

            FOREIGN KEY (account_id) REFERENCES accounts(id)
        );
    """),

    ("008_audit_logs", """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id TEXT PRIMARY KEY,
            actor TEXT,
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT,
            message TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_logs(created_at);
    """),

    ("009_secrets_metadata", """
        CREATE TABLE IF NOT EXISTS secrets_metadata (
            id TEXT PRIMARY KEY,
            secret_type TEXT NOT NULL,
            provider_key TEXT,
            account_id TEXT,
            storage_backend TEXT NOT NULL,
            created_at TEXT NOT NULL,
            rotated_at TEXT,
            last_used_at TEXT
        );
    """),

    ("010_provider_automation_level", """
        -- Add automation_level column to providers table
        ALTER TABLE providers ADD COLUMN automation_level TEXT NOT NULL DEFAULT 'manual';

        -- Update existing providers based on class and capabilities
        -- Class A: full_auto (API/SSH automated)
        UPDATE providers SET automation_level = 'full_auto'
            WHERE key IN ('oracle_cloud', 'gcp');

        -- Class A: partial_auto (some API but needs confirm)
        UPDATE providers SET automation_level = 'partial_auto'
            WHERE key IN ('paperspace', 'lightning_ai', 'codesphere');

        -- Class B: partial_auto (has API)
        UPDATE providers SET automation_level = 'partial_auto'
            WHERE key = 'kaggle';

        -- Class B: manual (notebook-based, user must start)
        UPDATE providers SET automation_level = 'manual'
            WHERE key IN ('google_colab', 'sagemaker', 'deepnote');

        -- Class C: manual
        UPDATE providers SET automation_level = 'manual'
            WHERE key IN ('huggingface', 'intel_devcloud', 'nvidia_vgpu');
    """),
]


def run_migrations(conn: sqlite3.Connection):
    """Run all pending migrations in order.

    Each migration is tracked in _migrations table.
    Migrations are idempotent — safe to call multiple times.
    """
    _ensure_migration_table(conn)
    for name, sql in MIGRATIONS:
        _apply_migration(conn, name, sql)
    logger.info("All migrations complete")


# ── Provider Seeding ────────────────────────────────────────────

# 12 providers from the original platform registry
PROVIDER_SEEDS = [
    {
        "key": "google_colab",
        "display_name": "Google Colab",
        "provider_class": "B",
        "automation_level": "manual",
        "default_session_limit_minutes": 720,
        "default_cooldown_minutes": 5,
    },
    {
        "key": "kaggle",
        "display_name": "Kaggle Notebooks",
        "provider_class": "B",
        "automation_level": "partial_auto",
        "default_session_limit_minutes": 540,
        "default_cooldown_minutes": 5,
    },
    {
        "key": "huggingface",
        "display_name": "HuggingFace Spaces",
        "provider_class": "C",
        "automation_level": "manual",
        "default_session_limit_minutes": 240,
        "default_cooldown_minutes": 10,
    },
    {
        "key": "paperspace",
        "display_name": "Paperspace Gradient",
        "provider_class": "A",
        "automation_level": "partial_auto",
        "default_session_limit_minutes": 360,
        "default_cooldown_minutes": 10,
    },
    {
        "key": "sagemaker",
        "display_name": "Amazon SageMaker Studio Lab",
        "provider_class": "B",
        "automation_level": "manual",
        "default_session_limit_minutes": 240,
        "default_cooldown_minutes": 10,
    },
    {
        "key": "lightning_ai",
        "display_name": "Lightning AI",
        "provider_class": "A",
        "automation_level": "partial_auto",
        "default_session_limit_minutes": 240,
        "default_cooldown_minutes": 10,
    },
    {
        "key": "codesphere",
        "display_name": "Codesphere",
        "provider_class": "A",
        "automation_level": "partial_auto",
        "default_session_limit_minutes": 240,
        "default_cooldown_minutes": 15,
    },
    {
        "key": "oracle_cloud",
        "display_name": "Oracle Cloud Free Tier",
        "provider_class": "A",
        "automation_level": "full_auto",
        "default_session_limit_minutes": 1440,
        "default_cooldown_minutes": 0,
    },
    {
        "key": "gcp",
        "display_name": "Google Cloud Platform",
        "provider_class": "A",
        "automation_level": "full_auto",
        "default_session_limit_minutes": 480,
        "default_cooldown_minutes": 0,
    },
    {
        "key": "intel_devcloud",
        "display_name": "Intel Developer Cloud",
        "provider_class": "C",
        "automation_level": "manual",
        "default_session_limit_minutes": 240,
        "default_cooldown_minutes": 15,
    },
    {
        "key": "deepnote",
        "display_name": "Deepnote",
        "provider_class": "B",
        "automation_level": "manual",
        "default_session_limit_minutes": 240,
        "default_cooldown_minutes": 15,
    },
    {
        "key": "nvidia_vgpu",
        "display_name": "NVIDIA vGPU Trial",
        "provider_class": "C",
        "automation_level": "manual",
        "default_session_limit_minutes": 480,
        "default_cooldown_minutes": 0,
    },
]


def seed_providers(conn: sqlite3.Connection):
    """Seed the 12 providers into the providers table.

    Uses INSERT OR IGNORE so re-seeding is safe.
    """
    now = datetime.now(timezone.utc).isoformat()
    for p in PROVIDER_SEEDS:
        conn.execute("""
            INSERT OR IGNORE INTO providers
                (key, display_name, provider_class, automation_level, enabled,
                 supports_gpu, supports_checkpoint, default_session_limit_minutes,
                 default_cooldown_minutes, created_at, updated_at)
            VALUES (?, ?, ?, ?, 1, 1, 1, ?, ?, ?, ?)
        """, (
            p["key"],
            p["display_name"],
            p["provider_class"],
            p["automation_level"],
            p["default_session_limit_minutes"],
            p["default_cooldown_minutes"],
            now,
            now,
        ))
    conn.commit()
    logger.info(f"Seeded {len(PROVIDER_SEEDS)} providers")
