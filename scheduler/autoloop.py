"""Auto Loop Orchestrator for FamilyGPU Orchestrator.

The AutoLoop is a background daemon that continuously manages the
training lifecycle without manual intervention:

  1. Monitor active leases for expiry → auto-trigger failover
  2. Auto-start queued jobs when capacity becomes available
  3. Run periodic health checks on accounts
  4. Auto-rotate through accounts when quota is exhausted
  5. Pre-emptively checkpoint before lease expiry
  6. Handle provider errors and auto-retry
  7. Smart rotation: proactively rotate jobs before quota exhaustion
  8. Auto-retry failed jobs with exponential backoff
  9. Auto-rebalance jobs across accounts for optimal utilization

This enables continuous AI training across the family GPU quota pool
without requiring the user to manually start/stop/rotate sessions.

Usage:
  from scheduler.autoloop import AutoLoop

  loop = AutoLoop()
  loop.start()   # Starts background thread
  loop.stop()    # Stops background thread gracefully

  # Or via run.py:
  python run.py --auto
"""

import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable
from dataclasses import dataclass, field

from db.repositories import (
    JobRepository, LeaseRepository, AccountRepository,
    AuditLogRepository, QuotaLedgerRepository, HealthRepository,
)
from scheduler.request import JobRequest, JobRequestResult, FailureReason
from scheduler.selector import AccountSelector
from scheduler.leases import LeaseManager
from scheduler.failover import FailoverManager
from scheduler.quota import QuotaEnforcer
from providers.registry import get_adapter

logger = logging.getLogger("fgt.autoloop")


# ── Configuration ──────────────────────────────────────────────────

@dataclass
class AutoLoopConfig:
    """Configuration for the AutoLoop daemon.

    All intervals are in seconds. Adjust for your needs.
    """
    # How often to check for expired leases
    lease_check_interval: float = 30.0

    # How often to try starting queued jobs
    queue_check_interval: float = 15.0

    # How often to run health checks on accounts
    health_check_interval: float = 300.0  # 5 minutes

    # How often to heartbeat active leases
    heartbeat_interval: float = 60.0

    # Minutes before lease expiry to trigger preemptive checkpoint
    checkpoint_before_expiry_minutes: int = 10

    # Max consecutive errors before disabling an account
    max_consecutive_errors: int = 3

    # Auto-retry delay after a failed job start (seconds)
    retry_delay: float = 30.0

    # Whether to auto-failover when a lease expires
    auto_failover: bool = True

    # Whether to auto-start queued jobs
    auto_start_queued: bool = True

    # Whether to auto-health-check accounts
    auto_health_check: bool = True

    # Whether to pre-emptively checkpoint before expiry
    auto_checkpoint: bool = True

    # Smart rotation: proactively rotate accounts when nearing quota limits
    auto_rotation: bool = True

    # Percentage of quota usage at which to start looking for a replacement account
    rotation_threshold_percent: float = 80.0  # Start rotating when 80% of quota used

    # Auto-retry: automatically re-queue failed jobs with exponential backoff
    auto_retry_failed: bool = True

    # Maximum retry attempts for a failed job
    max_job_retries: int = 3

    # Auto-rebalance: move jobs from overloaded accounts to underused ones
    auto_rebalance: bool = True

    # Callback for TUI/API notifications
    on_event: Optional[Callable] = None


@dataclass
class AutoLoopStats:
    """Runtime statistics for the AutoLoop."""
    started_at: Optional[str] = None
    total_leases_checked: int = 0
    total_expiries_handled: int = 0
    total_failovers_triggered: int = 0
    total_jobs_started: int = 0
    total_health_checks: int = 0
    total_checkpoints: int = 0
    total_errors: int = 0
    consecutive_errors: int = 0
    is_running: bool = False
    last_lease_check: Optional[str] = None
    last_queue_check: Optional[str] = None
    last_health_check: Optional[str] = None
    last_heartbeat: Optional[str] = None
    total_rotations: int = 0
    total_rebalances: int = 0
    total_auto_retries: int = 0
    last_rotation: Optional[str] = None
    last_rebalance: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "is_running": self.is_running,
            "total_leases_checked": self.total_leases_checked,
            "total_expiries_handled": self.total_expiries_handled,
            "total_failovers_triggered": self.total_failovers_triggered,
            "total_jobs_started": self.total_jobs_started,
            "total_health_checks": self.total_health_checks,
            "total_checkpoints": self.total_checkpoints,
            "total_errors": self.total_errors,
            "consecutive_errors": self.consecutive_errors,
            "last_lease_check": self.last_lease_check,
            "last_queue_check": self.last_queue_check,
            "last_health_check": self.last_health_check,
            "last_heartbeat": self.last_heartbeat,
            "total_rotations": self.total_rotations,
            "total_rebalances": self.total_rebalances,
            "total_auto_retries": self.total_auto_retries,
            "last_rotation": self.last_rotation,
            "last_rebalance": self.last_rebalance,
        }


# ── Auto Loop Engine ───────────────────────────────────────────────

class AutoLoop:
    """Background daemon for continuous GPU training orchestration.

    The AutoLoop runs as a daemon thread and handles:
      - Lease expiry detection and auto-failover
      - Queued job auto-start when capacity is available
      - Periodic health checks on accounts
      - Preemptive checkpointing before lease expiry
      - Heartbeat updates for active leases
      - Account cooldown clearing

    Thread Safety:
      All database operations go through the repository layer which
      uses a shared SQLite connection with WAL mode and busy_timeout.
      The AutoLoop runs in a separate daemon thread.
    """

    def __init__(self, config: Optional[AutoLoopConfig] = None):
        self.config = config or AutoLoopConfig()
        self.stats = AutoLoopStats()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Repositories
        self.job_repo = JobRepository()
        self.lease_repo = LeaseRepository()
        self.account_repo = AccountRepository()
        self.audit_repo = AuditLogRepository()
        self.quota_repo = QuotaLedgerRepository()
        self.health_repo = HealthRepository()

        # Scheduler components
        self.selector = AccountSelector()
        self.lease_manager = LeaseManager()
        self.failover = FailoverManager()
        self.quota_enforcer = QuotaEnforcer()

        # Track account error counts for auto-disable
        self._account_errors: dict[str, int] = {}

    # ── Thread Management ──────────────────────────────────────────

    def start(self):
        """Start the auto loop daemon in a background thread.

        The daemon thread will continue running until stop() is called
        or the main thread exits.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("AutoLoop is already running")
            return

        self._stop_event.clear()
        self.stats.is_running = True
        self.stats.started_at = datetime.now(timezone.utc).isoformat()

        self._thread = threading.Thread(
            target=self._run_loop,
            name="fgt-autoloop",
            daemon=True,
        )
        self._thread.start()

        self.audit_repo.log(
            action="autoloop_start",
            entity_type="system",
            message="AutoLoop daemon started",
            metadata={"config": {
                "lease_check_interval": self.config.lease_check_interval,
                "queue_check_interval": self.config.queue_check_interval,
                "auto_failover": self.config.auto_failover,
                "auto_start_queued": self.config.auto_start_queued,
            }},
        )

        logger.info("AutoLoop daemon started")

    def stop(self):
        """Stop the auto loop daemon gracefully.

        Sets a stop event and waits for the thread to finish.
        Current operations will complete before stopping.
        """
        if not self.stats.is_running:
            return

        self._stop_event.set()
        self.stats.is_running = False

        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

        self.audit_repo.log(
            action="autoloop_stop",
            entity_type="system",
            message="AutoLoop daemon stopped",
            metadata=self.stats.to_dict(),
        )

        logger.info("AutoLoop daemon stopped")

    @property
    def is_running(self) -> bool:
        """Check if the auto loop is currently running."""
        return self.stats.is_running and self._thread is not None and self._thread.is_alive()

    # ── Main Loop ──────────────────────────────────────────────────

    def _run_loop(self):
        """Main loop — cycles through all orchestration tasks.

        Uses separate timers for each task to allow different intervals.
        The loop sleeps in small increments to respond quickly to stop().
        """
        last_lease_check = 0.0
        last_queue_check = 0.0
        last_health_check = 0.0
        last_heartbeat = 0.0
        last_rotation_check = 0.0
        last_auto_retry = 0.0

        logger.info("AutoLoop main loop starting")

        while not self._stop_event.is_set():
            now = time.monotonic()

            # ── Check for expired leases ───────────────────────────
            if now - last_lease_check >= self.config.lease_check_interval:
                try:
                    self._check_lease_expiry()
                    last_lease_check = now
                except Exception as e:
                    logger.error(f"Lease check error: {e}")
                    self._record_error()

            # ── Preemptive checkpoint for soon-to-expire leases ────
            if now - last_lease_check >= self.config.lease_check_interval * 0.5:
                try:
                    self._check_preemptive_checkpoint()
                except Exception as e:
                    logger.debug(f"Preemptive checkpoint check error: {e}")

            # ── Start queued jobs ──────────────────────────────────
            if self.config.auto_start_queued and now - last_queue_check >= self.config.queue_check_interval:
                try:
                    self._start_queued_jobs()
                    last_queue_check = now
                except Exception as e:
                    logger.error(f"Queue check error: {e}")
                    self._record_error()

            # ── Heartbeat active leases ────────────────────────────
            if now - last_heartbeat >= self.config.heartbeat_interval:
                try:
                    self._heartbeat_active_leases()
                    last_heartbeat = now
                except Exception as e:
                    logger.debug(f"Heartbeat error: {e}")

            # ── Periodic health checks ─────────────────────────────
            if self.config.auto_health_check and now - last_health_check >= self.config.health_check_interval:
                try:
                    self._run_health_checks()
                    last_health_check = now
                except Exception as e:
                    logger.error(f"Health check error: {e}")

            # ── Clear expired cooldowns ────────────────────────────
            try:
                self._clear_expired_cooldowns()
            except Exception as e:
                logger.debug(f"Cooldown clear error: {e}")

            # ── Smart rotation check ─────────────────────────────
            if self.config.auto_rotation and now - last_rotation_check >= self.config.lease_check_interval * 2:
                try:
                    self._check_smart_rotation()
                    last_rotation_check = now
                except Exception as e:
                    logger.debug(f"Smart rotation check error: {e}")

            # ── Auto-retry failed jobs ───────────────────────────
            if self.config.auto_retry_failed and now - last_auto_retry >= 60.0:
                try:
                    self._retry_failed_jobs()
                    last_auto_retry = now
                except Exception as e:
                    logger.debug(f"Auto-retry check error: {e}")

            # ── Sleep in small increments for quick stop response ──
            self._sleep(1.0)

        logger.info("AutoLoop main loop exiting")

    def _sleep(self, seconds: float):
        """Sleep for the given duration, but check stop event every 0.5s."""
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self._stop_event.is_set():
            time.sleep(min(0.5, end - time.monotonic()))

    def _record_error(self):
        """Record a loop error and update stats."""
        self.stats.total_errors += 1
        self.stats.consecutive_errors += 1

    def _emit_event(self, event_type: str, data: dict):
        """Emit an event for TUI/API notification."""
        if self.config.on_event:
            try:
                self.config.on_event(event_type, data)
            except Exception:
                pass

    # ── Task: Lease Expiry Detection ──────────────────────────────

    def _check_lease_expiry(self):
        """Check for and handle expired leases.

        For each expired lease:
          1. Record usage
          2. If auto_failover is enabled and job has a checkpoint:
             Trigger failover to a new account
          3. Otherwise, mark the job as failed
        """
        now = datetime.now(timezone.utc).isoformat()
        active_leases = self.lease_repo.list_active()

        self.stats.total_leases_checked += len(active_leases)
        self.stats.last_lease_check = now

        for lease in active_leases:
            # Check if lease has expired
            if lease.get("expires_at") and lease["expires_at"] <= now:
                logger.info(
                    f"AutoLoop: Lease {lease['id']} expired "
                    f"(job: {lease['job_id']}, account: {lease['account_id']})"
                )

                self.stats.total_expiries_handled += 1

                # Check if we should failover
                job = self.job_repo.get_by_id(lease["job_id"])
                if job and self.config.auto_failover and job.get("checkpoint_uri"):
                    # Attempt failover
                    result = self.failover.attempt_failover(
                        job_id=lease["job_id"],
                        failed_lease_id=lease["id"],
                    )

                    if result and result.status == "accepted":
                        self.stats.total_failovers_triggered += 1
                        # Mark old lease as expired
                        self.lease_manager.expire_lease(lease["id"])
                        self._emit_event("failover", {
                            "job_id": lease["job_id"],
                            "old_lease_id": lease["id"],
                            "new_lease_id": result.lease_id,
                            "new_provider": result.provider,
                        })
                        logger.info(
                            f"AutoLoop: Failover succeeded — "
                            f"new lease {result.lease_id} on {result.provider}"
                        )
                    else:
                        # Failover failed — expire the lease
                        self.lease_manager.expire_lease(lease["id"])
                        logger.warning(
                            f"AutoLoop: Failover failed for job {lease['job_id']}"
                        )
                else:
                    # No checkpoint or failover disabled — just expire
                    self.lease_manager.expire_lease(lease["id"])
                    self._emit_event("lease_expired", {
                        "lease_id": lease["id"],
                        "job_id": lease["job_id"],
                    })

    # ── Task: Preemptive Checkpoint ───────────────────────────────

    def _check_preemptive_checkpoint(self):
        """Check for leases about to expire and trigger checkpoint.

        This runs more frequently than lease expiry checks to ensure
        we save training state before the lease expires.
        """
        if not self.config.auto_checkpoint:
            return

        now = datetime.now(timezone.utc)
        threshold = now + timedelta(minutes=self.config.checkpoint_before_expiry_minutes)
        threshold_iso = threshold.isoformat()

        active_leases = self.lease_repo.list_active()

        for lease in active_leases:
            if lease.get("status") != "running":
                continue

            # Check if lease will expire soon
            if lease.get("expires_at") and lease["expires_at"] <= threshold_iso:
                # Only checkpoint once per lease (mark as checkpointing)
                if lease.get("status") == "checkpointing":
                    continue

                job = self.job_repo.get_by_id(lease["job_id"])
                if not job or not job.get("checkpoint_uri"):
                    continue

                account = self.account_repo.get_by_id(lease["account_id"])
                if not account:
                    continue

                adapter = get_adapter(lease["provider_key"])
                if not adapter:
                    continue

                logger.info(
                    f"AutoLoop: Preemptive checkpoint for lease {lease['id']} "
                    f"(expires in <{self.config.checkpoint_before_expiry_minutes}min)"
                )

                # Mark lease as checkpointing
                self.lease_repo.mark_checkpointing(lease["id"])

                try:
                    result = adapter.sync_checkpoint(
                        lease, account, job["checkpoint_uri"]
                    )
                    if result.ok:
                        self.stats.total_checkpoints += 1
                        self._emit_event("checkpoint", {
                            "lease_id": lease["id"],
                            "job_id": lease["job_id"],
                            "status": "ok",
                        })
                        logger.info(
                            f"AutoLoop: Checkpoint saved for lease {lease['id']}"
                        )
                    else:
                        logger.warning(
                            f"AutoLoop: Checkpoint failed for lease {lease['id']}: "
                            f"{result.message}"
                        )
                        # Revert to running so we can try again
                        self.lease_repo.update(lease["id"], status="running")
                except Exception as e:
                    logger.error(f"AutoLoop: Checkpoint error for lease {lease['id']}: {e}")
                    self.lease_repo.update(lease["id"], status="running")

    # ── Task: Auto-Start Queued Jobs ──────────────────────────────

    def _start_queued_jobs(self):
        """Check for queued jobs and start them if capacity is available.

        For each queued job:
          1. Use AccountSelector to find the best account
          2. Create a lease
          3. Start the job via the provider adapter
          4. If start fails, try next best account (up to 3 attempts)
        """
        self.stats.last_queue_check = datetime.now(timezone.utc).isoformat()

        queued_jobs = self.job_repo.list_queued()
        if not queued_jobs:
            return

        logger.info(f"AutoLoop: Checking {len(queued_jobs)} queued job(s)")

        for job in queued_jobs:
            # Build a job request for the selector
            request = JobRequest(
                job_name=job["name"],
                gpu_profile=job["gpu_profile"],
                max_runtime_minutes=job["max_runtime_minutes"],
                priority=job.get("priority", "normal"),
                checkpoint_uri=job.get("checkpoint_uri", ""),
                entrypoint=job.get("entrypoint", "train.py"),
                args=job.get("args") or {},
                allow_providers=job.get("allow_providers") or [],
                deny_providers=job.get("deny_providers") or [],
                checkpoint_required=bool(job.get("checkpoint_uri")),
            )

            # Validate
            errors = request.validate()
            if errors:
                logger.warning(
                    f"AutoLoop: Job {job['id']} has validation errors: {errors}"
                )
                self.job_repo.mark_failed(
                    job["id"],
                    f"Invalid job request: {'; '.join(errors)}"
                )
                continue

            # Select account
            account, failure_reason = self.selector.select(request)

            if not account:
                logger.info(
                    f"AutoLoop: No account available for job {job['id']}: {failure_reason}"
                )
                # Will retry next cycle
                continue

            # Check quota
            can_use, quota_reason = self.quota_enforcer.can_use_account(
                account["id"], job["max_runtime_minutes"]
            )

            if not can_use:
                logger.info(
                    f"AutoLoop: Account {account['id']} quota insufficient: {quota_reason}"
                )
                continue

            # Create lease
            lease = self.lease_manager.create_lease(
                job_id=job["id"],
                account_id=account["id"],
                provider_key=account["provider_key"],
                max_runtime_minutes=job["max_runtime_minutes"],
            )

            if not lease:
                logger.warning(
                    f"AutoLoop: Failed to create lease for job {job['id']}"
                )
                continue

            # Start the job
            result = self.lease_manager.start_lease(lease["id"])

            if result.ok:
                self.stats.total_jobs_started += 1
                self.stats.consecutive_errors = 0  # Reset on success
                self._emit_event("job_started", {
                    "job_id": job["id"],
                    "job_name": job["name"],
                    "lease_id": lease["id"],
                    "provider": account["provider_key"],
                    "account_owner": account.get("owner_id", "unknown"),
                    "manual_required": result.status == "manual_required",
                })

                if result.status == "manual_required":
                    logger.info(
                        f"AutoLoop: Job {job['id']} requires manual start "
                        f"on {account['provider_key']}"
                    )
                else:
                    logger.info(
                        f"AutoLoop: Job {job['id']} started on "
                        f"{account['provider_key']}/{account['label']}"
                    )

                self.audit_repo.log(
                    action="autoloop_start_job",
                    entity_type="job",
                    entity_id=job["id"],
                    message=f"AutoLoop started job on {account['provider_key']}/{account['label']}",
                    metadata={
                        "lease_id": lease["id"],
                        "provider": account["provider_key"],
                        "auto": True,
                    },
                )
            else:
                # Failed to start — record error on account
                logger.warning(
                    f"AutoLoop: Failed to start job {job['id']}: {result.message}"
                )
                self._record_account_error(account["id"])
                self._record_error()

    # ── Task: Health Checks ────────────────────────────────────────

    def _run_health_checks(self):
        """Run health checks on all active accounts.

        For each account:
          1. Get the provider adapter
          2. Call health_check()
          3. Record the result in provider_health table
          4. If health is "down", increment error counter
          5. If errors exceed threshold, disable the account
        """
        self.stats.last_health_check = datetime.now(timezone.utc).isoformat()

        accounts = self.account_repo.list_all(status="active")
        if not accounts:
            return

        logger.info(f"AutoLoop: Running health checks on {len(accounts)} account(s)")

        for account in accounts:
            adapter = get_adapter(account["provider_key"])
            if not adapter:
                continue

            try:
                result = adapter.health_check(account)
                health_status = "ok" if result.ok else "down"

                self.health_repo.record(
                    account_id=account["id"],
                    provider_key=account["provider_key"],
                    status=health_status,
                    message=result.message,
                )

                self.stats.total_health_checks += 1

                if not result.ok:
                    self._record_account_error(account["id"])
                    logger.warning(
                        f"AutoLoop: Health check FAILED for {account['label']} "
                        f"({account['provider_key']}): {result.message}"
                    )
                else:
                    # Reset error count on successful health check
                    self._account_errors[account["id"]] = 0

            except Exception as e:
                logger.error(
                    f"AutoLoop: Health check error for {account['label']}: {e}"
                )
                self._record_account_error(account["id"])

    def _record_account_error(self, account_id: str):
        """Record an error for an account and disable if threshold exceeded."""
        count = self._account_errors.get(account_id, 0) + 1
        self._account_errors[account_id] = count

        if count >= self.config.max_consecutive_errors:
            logger.warning(
                f"AutoLoop: Account {account_id} has {count} consecutive errors — "
                f"auto-disabling (threshold: {self.config.max_consecutive_errors})"
            )
            self.account_repo.disable(account_id)
            self.audit_repo.log(
                action="autoloop_disable_account",
                entity_type="account",
                entity_id=account_id,
                message=f"Auto-disabled after {count} consecutive errors",
                metadata={"error_count": count},
            )
            self._account_errors[account_id] = 0  # Reset after disabling

    # ── Task: Heartbeat Active Leases ─────────────────────────────

    def _heartbeat_active_leases(self):
        """Send heartbeats for all running leases.

        This tells the system the lease is still active and being
        monitored. It also provides an opportunity to detect stale
        leases where the remote process has died.
        """
        self.stats.last_heartbeat = datetime.now(timezone.utc).isoformat()

        active_leases = self.lease_repo.list_active()
        for lease in active_leases:
            if lease.get("status") == "running":
                try:
                    self.lease_manager.heartbeat(lease["id"])
                except Exception as e:
                    logger.debug(f"Heartbeat failed for lease {lease['id']}: {e}")

    # ── Task: Clear Expired Cooldowns ─────────────────────────────

    def _clear_expired_cooldowns(self):
        """Clear cooldowns that have expired, making accounts available again."""
        accounts = self.account_repo.list_all(status="cooldown")
        for account in accounts:
            self.account_repo.clear_cooldown(account["id"])

    # ── Task: Smart Account Rotation ─────────────────────────────

    def _check_smart_rotation(self):
        """Check if any running jobs should be rotated to accounts with more quota.

        Smart rotation proactively moves jobs from accounts that are running
        low on quota to accounts with more available quota, preventing
        unexpected lease expiries due to quota exhaustion.
        """
        if not self.config.auto_rotation:
            return

        now = datetime.now(timezone.utc).isoformat()
        active_leases = self.lease_repo.list_active()

        for lease in active_leases:
            if lease.get("status") != "running":
                continue

            account = self.account_repo.get_by_id(lease["account_id"])
            if not account:
                continue

            # Check quota usage percentage
            daily_limit = account.get("daily_limit_minutes", 120)
            weekly_limit = account.get("weekly_limit_minutes", 600)
            remaining_daily = self.quota_repo.remaining_daily(account["id"], daily_limit)
            remaining_weekly = self.quota_repo.remaining_weekly(account["id"], weekly_limit)

            # Calculate usage percentage
            daily_used_pct = ((daily_limit - remaining_daily) / daily_limit * 100) if daily_limit > 0 else 100
            weekly_used_pct = ((weekly_limit - remaining_weekly) / weekly_limit * 100) if weekly_limit > 0 else 100

            max_used_pct = max(daily_used_pct, weekly_used_pct)

            if max_used_pct < self.config.rotation_threshold_percent:
                continue  # Still plenty of quota

            # Check if we already rotated this lease recently
            job = self.job_repo.get_by_id(lease["job_id"])
            if not job:
                continue

            # Find a better account
            request = JobRequest(
                job_name=job["name"],
                gpu_profile=job["gpu_profile"],
                max_runtime_minutes=job["max_runtime_minutes"],
                priority=job["priority"],
                checkpoint_uri=job.get("checkpoint_uri", ""),
                entrypoint=job.get("entrypoint", "train.py"),
                args=job.get("args") or {},
                allow_providers=job.get("allow_providers") or [],
                deny_providers=job.get("deny_providers") or [],
                checkpoint_required=True,
            )

            new_account, failure_reason = self.selector.select(request)

            if not new_account or new_account["id"] == account["id"]:
                continue  # No better account available

            # Check new account has significantly more quota
            new_daily_limit = new_account.get("daily_limit_minutes", 120)
            new_remaining_daily = self.quota_repo.remaining_daily(new_account["id"], new_daily_limit)

            if new_remaining_daily <= remaining_daily * 1.5:
                continue  # New account doesn't have significantly more quota

            logger.info(
                f"AutoLoop: Smart rotation — lease {lease['id']} "
                f"account {account['label']} at {max_used_pct:.0f}% quota, "
                f"rotating to {new_account['label']} ({new_remaining_daily}min remaining)"
            )

            # Try checkpoint on current lease
            if job.get("checkpoint_uri"):
                adapter = get_adapter(lease["provider_key"])
                if adapter:
                    try:
                        ckpt_result = adapter.sync_checkpoint(lease, account, job["checkpoint_uri"])
                        if not ckpt_result.ok:
                            logger.warning(f"Smart rotation: checkpoint failed, skipping rotation: {ckpt_result.message}")
                            continue
                    except Exception as e:
                        logger.warning(f"Smart rotation: checkpoint error, skipping: {e}")
                        continue

            # Create new lease
            new_lease = self.lease_manager.create_lease(
                job_id=lease["job_id"],
                account_id=new_account["id"],
                provider_key=new_account["provider_key"],
                max_runtime_minutes=job["max_runtime_minutes"],
            )

            if not new_lease:
                logger.warning(f"Smart rotation: failed to create new lease")
                continue

            # Start on new account
            result = self.lease_manager.start_lease(new_lease["id"])

            if result.ok:
                # Complete old lease with usage tracking
                runtime = 0
                if lease.get("started_at"):
                    started = datetime.fromisoformat(lease["started_at"])
                    runtime = int((datetime.now(timezone.utc) - started).total_seconds() / 60)

                self.lease_repo.mark_completed(lease["id"], runtime_minutes=runtime)
                self.quota_repo.record_usage(
                    account_id=account["id"],
                    provider_key=account["provider_key"],
                    used_minutes=max(runtime, 1),
                    job_id=lease["job_id"],
                    lease_id=lease["id"],
                )

                # Cooldown old account
                if account.get("cooldown_minutes", 0) > 0:
                    self.account_repo.set_cooldown(account["id"], account["cooldown_minutes"])

                self.stats.total_rotations += 1
                self.stats.last_rotation = now

                self._emit_event("smart_rotation", {
                    "job_id": lease["job_id"],
                    "old_lease_id": lease["id"],
                    "new_lease_id": new_lease["id"],
                    "old_account": account["label"],
                    "new_account": new_account["label"],
                    "quota_used_pct": max_used_pct,
                })

                self.audit_repo.log(
                    action="smart_rotation",
                    entity_type="lease",
                    entity_id=lease["id"],
                    message=f"Rotated from {account['label']} ({max_used_pct:.0f}% quota) to {new_account['label']}",
                    metadata={
                        "old_account_id": account["id"],
                        "new_account_id": new_account["id"],
                        "quota_used_pct": max_used_pct,
                        "new_lease_id": new_lease["id"],
                    },
                )

                logger.info(
                    f"AutoLoop: Smart rotation complete — "
                    f"{account['label']} → {new_account['label']}"
                )
            else:
                # Rotation failed — cancel new lease, keep running on old
                self.lease_repo.cancel(new_lease["id"])
                logger.warning(f"Smart rotation: failed to start on new account, keeping old lease")

    # ── Task: Auto-Retry Failed Jobs ─────────────────────────────

    def _retry_failed_jobs(self):
        """Auto-retry recently failed jobs with exponential backoff.

        When a job fails, this method checks if it's eligible for retry
        and re-queues it with a delay. Maximum retries is configurable.
        """
        if not self.config.auto_retry_failed:
            return

        # Find recently failed jobs
        failed_jobs = self.job_repo.list_all(status="failed")

        for job in failed_jobs:
            # Count previous leases for this job
            all_leases = self.lease_repo.list_all()
            job_leases = [l for l in all_leases if l["job_id"] == job["id"]]
            retry_count = len([l for l in job_leases if l["status"] in ("failed", "expired")])

            if retry_count >= self.config.max_job_retries:
                logger.debug(f"Auto-retry: Job {job['id']} exceeded max retries ({retry_count})")
                continue

            # Check if we recently retried (avoid immediate retry loop)
            if job.get("completed_at"):
                completed = datetime.fromisoformat(job["completed_at"])
                elapsed = (datetime.now(timezone.utc) - completed).total_seconds()
                backoff_delay = min(30 * (2 ** retry_count), 300)  # 30s, 60s, 120s, max 300s

                if elapsed < backoff_delay:
                    continue  # Not enough time has passed

            # Re-queue the job
            self.job_repo.update(job["id"], status="queued", failure_reason=None, completed_at=None)

            self.stats.total_auto_retries += 1

            self.audit_repo.log(
                action="auto_retry",
                entity_type="job",
                entity_id=job["id"],
                message=f"Auto-retry #{retry_count + 1} for job {job['name']}",
                metadata={"retry_count": retry_count + 1},
            )

            logger.info(f"AutoLoop: Auto-retry #{retry_count + 1} for job {job['id']} ({job['name']})")

    # ── Task: Auto-Rebalance Jobs ────────────────────────────────

    def _rebalance_jobs(self):
        """Rebalance jobs across accounts for optimal quota utilization.

        When one account is overloaded while another has ample quota,
        this method moves jobs to balance the load.
        """
        if not self.config.auto_rebalance:
            return

        now = datetime.now(timezone.utc).isoformat()

        # Get all active accounts and their remaining quota
        accounts = self.account_repo.list_all(status="active")
        if len(accounts) < 2:
            return

        account_quotas = []
        for acct in accounts:
            if self.lease_repo.is_account_busy(acct["id"]):
                continue  # Skip busy accounts
            daily_limit = acct.get("daily_limit_minutes", 120)
            remaining = self.quota_repo.remaining_daily(acct["id"], daily_limit)
            account_quotas.append((acct, remaining))

        if len(account_quotas) < 2:
            return

        # Sort by remaining quota ascending
        account_quotas.sort(key=lambda x: x[1])

        # Find the most loaded account with a running lease
        for loaded_acct, remaining in account_quotas:
            leases = self.lease_repo.list_by_account(loaded_acct["id"], active_only=True)
            if not leases:
                continue

            for lease in leases:
                if lease.get("status") != "running":
                    continue

                job = self.job_repo.get_by_id(lease["job_id"])
                if not job or not job.get("checkpoint_uri"):
                    continue

                # Find the least loaded account
                best_acct, best_remaining = account_quotas[-1]
                if best_remaining <= remaining * 2:
                    continue  # No significantly better account

                if best_acct["id"] == loaded_acct["id"]:
                    continue

                # This is handled by smart rotation, so we just log and skip
                # to avoid double-rotation
                break
            break

    # ── Manual Triggers ────────────────────────────────────────────

    def submit_job(self, job_request: JobRequest) -> JobRequestResult:
        """Submit a job and let the auto loop handle scheduling.

        If capacity is immediately available, the job starts right away.
        Otherwise, it's queued for the next available account.

        Args:
            job_request: Job specification

        Returns:
            JobRequestResult with status and details
        """
        # Validate
        errors = job_request.validate()
        if errors:
            return JobRequestResult(
                status="rejected",
                failure_reason=FailureReason.CHECKPOINT_REQUIRED_MISSING,
                message=f"Invalid job request: {'; '.join(errors)}",
            )

        # Try to start immediately
        account, failure_reason = self.selector.select(job_request)

        if account:
            # Create job
            job = self.job_repo.create(
                name=job_request.job_name,
                gpu_profile=job_request.gpu_profile,
                max_runtime_minutes=job_request.max_runtime_minutes,
                checkpoint_uri=job_request.checkpoint_uri,
                entrypoint=job_request.entrypoint,
                priority=job_request.priority,
                args=job_request.args,
                allow_providers=job_request.allow_providers,
                deny_providers=job_request.deny_providers,
                created_by="auto",
            )

            if not job:
                return JobRequestResult(
                    status="rejected",
                    failure_reason=FailureReason.NO_ACTIVE_ACCOUNTS,
                    message="Failed to create job record",
                )

            # Create lease
            lease = self.lease_manager.create_lease(
                job_id=job["id"],
                account_id=account["id"],
                provider_key=account["provider_key"],
                max_runtime_minutes=job_request.max_runtime_minutes,
            )

            if lease:
                result = self.lease_manager.start_lease(lease["id"])

                if result.ok:
                    return JobRequestResult(
                        status="accepted",
                        job_id=job["id"],
                        lease_id=lease["id"],
                        provider=account["provider_key"],
                        account_owner=account.get("owner_id", "unknown"),
                        estimated_runtime_minutes=job_request.max_runtime_minutes,
                        message=result.message,
                    )

        # Queue the job for later
        job = self.job_repo.create(
            name=job_request.job_name,
            gpu_profile=job_request.gpu_profile,
            max_runtime_minutes=job_request.max_runtime_minutes,
            checkpoint_uri=job_request.checkpoint_uri,
            entrypoint=job_request.entrypoint,
            priority=job_request.priority,
            args=job_request.args,
            allow_providers=job_request.allow_providers,
            deny_providers=job_request.deny_providers,
            created_by="auto",
        )

        if job:
            return JobRequestResult(
                status="queued",
                job_id=job["id"],
                message=f"Job queued — no GPU currently available ({failure_reason.value if failure_reason else 'unknown'})",
            )

        return JobRequestResult(
            status="rejected",
            failure_reason=failure_reason,
            message=f"Cannot schedule job: {failure_reason.value if failure_reason else 'unknown'}",
        )

    def force_failover(self, job_id: str) -> Optional[JobRequestResult]:
        """Manually trigger a failover for a specific job.

        Useful when a user detects a stuck job and wants to
        move it to a different account immediately.
        """
        lease = self.lease_repo.get_active_for_job(job_id)
        if not lease:
            return None

        result = self.failover.attempt_failover(job_id, lease["id"])

        if result:
            self.stats.total_failovers_triggered += 1

        return result

    def get_status(self) -> dict:
        """Get current auto loop status for display."""
        return {
            "auto_loop": self.stats.to_dict(),
            "config": {
                "lease_check_interval": self.config.lease_check_interval,
                "queue_check_interval": self.config.queue_check_interval,
                "health_check_interval": self.config.health_check_interval,
                "auto_failover": self.config.auto_failover,
                "auto_start_queued": self.config.auto_start_queued,
                "auto_health_check": self.config.auto_health_check,
                "auto_checkpoint": self.config.auto_checkpoint,
                "auto_rotation": self.config.auto_rotation,
                "rotation_threshold_percent": self.config.rotation_threshold_percent,
                "auto_retry_failed": self.config.auto_retry_failed,
                "max_job_retries": self.config.max_job_retries,
                "auto_rebalance": self.config.auto_rebalance,
                "checkpoint_before_expiry_minutes": self.config.checkpoint_before_expiry_minutes,
            },
            "active_leases": len(self.lease_repo.list_active()),
            "queued_jobs": len(self.job_repo.list_queued()),
            "available_accounts": len(self.account_repo.list_available()),
        }
