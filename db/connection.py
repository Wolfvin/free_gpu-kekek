"""SQLite connection management for FamilyGPU Orchestrator.

Uses a single persistent connection with WAL mode for concurrency safety.
The database file is created automatically on first use.
State files are stored in ~/.familygpu/ for security (not in the project root).
"""

import os
import shutil
import sqlite3
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("fgt.db")

# Default database path — stored in ~/.familygpu/ for security
_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".familygpu")
DEFAULT_DB_PATH = os.environ.get(
    "DATABASE_URL",
    os.path.join(_CONFIG_DIR, "familygpu.db")
)

# If DATABASE_URL starts with "file:", strip the prefix
if DEFAULT_DB_PATH.startswith("file:"):
    DEFAULT_DB_PATH = DEFAULT_DB_PATH[5:]


def _ensure_config_dir() -> None:
    """Create ~/.familygpu/ with restricted permissions if it doesn't exist.

    Also migrates an existing familygpu.db from the project root to the new
    location, so users don't lose their data on upgrade.
    """
    config_dir = Path(_CONFIG_DIR)
    if not config_dir.exists():
        config_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(str(config_dir), 0o700)
        logger.info(f"Created config directory: {config_dir}")

    # Migration: move database from project root to ~/.familygpu/
    old_db = Path(__file__).parent.parent / "familygpu.db"
    new_db = config_dir / "familygpu.db"
    if old_db.exists() and not new_db.exists():
        shutil.move(str(old_db), str(new_db))
        logger.info(f"Migrated database: {old_db} -> {new_db}")

DB_PATH = DEFAULT_DB_PATH

# Module-level connection cache
_connection: Optional[sqlite3.Connection] = None


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Get or create a persistent SQLite connection.

    Uses WAL mode for better concurrency and sets foreign keys ON.
    Connection is reused across calls (singleton pattern).
    """
    # Ensure config dir exists before opening database
    _ensure_config_dir()
    global _connection
    if _connection is not None:
        try:
            _connection.execute("SELECT 1")
            return _connection
        except sqlite3.Error:
            _connection = None

    path = db_path or DB_PATH
    logger.info(f"Opening database: {path}")

    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")

    _connection = conn
    return conn


def init_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Initialize the database: create connection, run migrations, seed data.

    This should be called once at app startup.
    """
    from db.migrations import run_migrations, seed_providers

    conn = get_connection(db_path)
    run_migrations(conn)
    seed_providers(conn)
    return conn


def close_connection():
    """Close the persistent database connection."""
    global _connection
    if _connection is not None:
        try:
            _connection.close()
        except Exception:
            pass
        _connection = None
