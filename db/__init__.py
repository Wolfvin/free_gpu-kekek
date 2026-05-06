"""Database layer for FamilyGPU Orchestrator.

Provides SQLite connection management, schema migrations,
and repository classes for all entities.
"""

from db.connection import get_connection, init_db, DB_PATH
from db.migrations import run_migrations, seed_providers
from db.repositories import (
    OwnerRepository,
    ProviderRepository,
    AccountRepository,
    JobRepository,
    LeaseRepository,
    QuotaLedgerRepository,
    AuditLogRepository,
    HealthRepository,
    SecretMetadataRepository,
)

__all__ = [
    "get_connection",
    "init_db",
    "DB_PATH",
    "run_migrations",
    "seed_providers",
    "OwnerRepository",
    "ProviderRepository",
    "AccountRepository",
    "JobRepository",
    "LeaseRepository",
    "QuotaLedgerRepository",
    "AuditLogRepository",
    "HealthRepository",
    "SecretMetadataRepository",
]
