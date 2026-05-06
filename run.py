#!/usr/bin/env python3
"""FamilyGPU Orchestrator — Main entry point.

A quota-aware multi-account GPU scheduler for authorized family accounts.
Supports 12 GPU/compute providers with AI agent integration.

Usage:
  python run.py              — Launch TUI
  python run.py --api        — Launch HTTP API server
  python run.py --status     — Show system status
  python run.py --json       — Show status as JSON
"""

import sys
import os
import argparse
import json
import logging

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(level: str = "INFO"):
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="FamilyGPU Orchestrator — Quota-Aware Multi-Account GPU Scheduler"
    )
    parser.add_argument("--api", action="store_true", help="Start HTTP API server")
    parser.add_argument("--api-host", default="127.0.0.1", help="API bind address")
    parser.add_argument("--api-port", type=int, default=8420, help="API port")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--db", default=None, help="Path to SQLite database")
    parser.add_argument("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING)")

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.api:
        from api.server import start_api_server
        start_api_server(host=args.api_host, port=args.api_port, db_path=args.db)
        return

    if args.status:
        _show_status(args.json, args.db)
        return

    # Default: launch TUI
    from tui import FamilyGPUTUI
    app = FamilyGPUTUI(db_path=args.db)
    app.run()


def _show_status(as_json: bool = False, db_path: str = None):
    """Show system status."""
    from db.connection import init_db
    from db.repositories import AccountRepository, JobRepository, LeaseRepository, QuotaLedgerRepository
    from api import GPUSchedulerAPI

    api = GPUSchedulerAPI(db_path=db_path)
    capacity = api.get_available_capacity()
    jobs = api.list_jobs()
    active_leases = api.lease_repo.list_active()

    if as_json:
        print(json.dumps({
            "capacity": capacity,
            "jobs": jobs,
            "active_leases": len(active_leases),
        }, indent=2, default=str))
    else:
        print("╔══════════════════════════════════════════════╗")
        print("║     FamilyGPU Orchestrator — Status          ║")
        print("╚══════════════════════════════════════════════╝")
        print()
        print(f"  Available accounts: {capacity['total_accounts']}")
        print()
        print("  By Provider:")
        for pk, info in capacity.get("by_provider", {}).items():
            auto_label = "AUTO" if info["auto"] else "MANUAL"
            print(f"    [{auto_label}] {info['name']}: {info['count']} account(s)")
        print()
        print("  By GPU Profile:")
        for profile, count in capacity.get("by_profile", {}).items():
            print(f"    {profile}: {count} account(s)")
        print()
        print(f"  Total jobs: {len(jobs)}")
        print(f"  Active leases: {len(active_leases)}")


if __name__ == "__main__":
    main()
