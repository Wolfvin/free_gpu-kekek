#!/usr/bin/env python3
"""FamilyGPU Orchestrator — Main entry point.

A quota-aware multi-account GPU scheduler for authorized family accounts.
Supports 12 GPU/compute providers with AI agent integration.

Usage:
  python run.py              — Launch TUI
  python run.py --auto       — Launch TUI with auto-scheduling daemon
  python run.py --api        — Launch HTTP API server
  python run.py --auto --api — Launch API with auto-scheduling daemon
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
    parser.add_argument("--auto", action="store_true",
                        help="Enable auto-scheduling daemon (continuous lease monitoring, auto-failover, auto-start queued jobs)")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--db", default=None, help="Path to SQLite database")
    parser.add_argument("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING)")

    # Auto-loop configuration
    parser.add_argument("--lease-check-interval", type=float, default=30.0,
                        help="Seconds between lease expiry checks (default: 30)")
    parser.add_argument("--queue-check-interval", type=float, default=15.0,
                        help="Seconds between queued job checks (default: 15)")
    parser.add_argument("--health-check-interval", type=float, default=300.0,
                        help="Seconds between account health checks (default: 300)")
    parser.add_argument("--checkpoint-before-expiry", type=int, default=10,
                        help="Minutes before lease expiry to trigger checkpoint (default: 10)")
    parser.add_argument("--no-failover", action="store_true",
                        help="Disable automatic failover on lease expiry")
    parser.add_argument("--no-auto-start", action="store_true",
                        help="Disable auto-starting queued jobs")
    parser.add_argument("--no-health-check", action="store_true",
                        help="Disable automatic health checks")

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.status:
        _show_status(args.json, args.db, args.auto)
        return

    if args.api:
        _start_api(args)
        return

    # Default: launch TUI
    _start_tui(args)


def _create_autoloop(args) -> "AutoLoop":
    """Create an AutoLoop instance from CLI arguments."""
    from scheduler.autoloop import AutoLoop, AutoLoopConfig

    config = AutoLoopConfig(
        lease_check_interval=args.lease_check_interval,
        queue_check_interval=args.queue_check_interval,
        health_check_interval=args.health_check_interval,
        checkpoint_before_expiry_minutes=args.checkpoint_before_expiry,
        auto_failover=not args.no_failover,
        auto_start_queued=not args.no_auto_start,
        auto_health_check=not args.no_health_check,
    )

    return AutoLoop(config=config)


def _start_tui(args):
    """Launch the TUI application, optionally with auto loop."""
    from db.connection import init_db
    from tui import FamilyGPUTUI

    # Initialize database
    init_db(args.db)

    # Create and start auto loop if requested
    autoloop = None
    if args.auto:
        autoloop = _create_autoloop(args)
        autoloop.start()

    app = FamilyGPUTUI(db_path=args.db, autoloop=autoloop)
    try:
        app.run()
    finally:
        # Ensure auto loop is stopped when TUI exits
        if autoloop and autoloop.is_running:
            autoloop.stop()


def _start_api(args):
    """Start the HTTP API server, optionally with auto loop."""
    from api.server import start_api_server

    autoloop = None
    if args.auto:
        autoloop = _create_autoloop(args)
        autoloop.start()

    try:
        start_api_server(
            host=args.api_host,
            port=args.api_port,
            db_path=args.db,
            autoloop=autoloop,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if autoloop and autoloop.is_running:
            autoloop.stop()


def _show_status(as_json: bool = False, db_path: str = None, show_auto: bool = False):
    """Show system status."""
    from db.connection import init_db
    from db.repositories import AccountRepository, JobRepository, LeaseRepository, QuotaLedgerRepository
    from api import GPUSchedulerAPI

    api = GPUSchedulerAPI(db_path=db_path)
    capacity = api.get_available_capacity()
    jobs = api.list_jobs()
    active_leases = api.lease_repo.list_active()

    status_data = {
        "capacity": capacity,
        "jobs": jobs,
        "active_leases": len(active_leases),
    }

    if as_json:
        print(json.dumps(status_data, indent=2, default=str))
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

        # Show queued jobs count
        queued = [j for j in jobs if j.get("status") == "queued"]
        if queued:
            print(f"  Queued jobs: {len(queued)}")

        print()
        print("  ── Auto Mode ────────────────────────────")
        print(f"  Run with --auto to enable continuous scheduling:")
        print(f"    python run.py --auto")
        print(f"    python run.py --auto --api")
        print()
        print("  Auto mode features:")
        print("    ✓ Auto-failover when leases expire")
        print("    ✓ Auto-start queued jobs when capacity available")
        print("    ✓ Preemptive checkpoint before lease expiry")
        print("    ✓ Periodic health checks on accounts")
        print("    ✓ Auto-disable accounts with repeated errors")


if __name__ == "__main__":
    main()
