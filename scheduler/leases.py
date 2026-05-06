"""Lease lifecycle management for FamilyGPU Orchestrator.

Manages the creation, tracking, and lifecycle of GPU leases.
A lease represents temporary permission to use one account for a job.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from db.repositories import (
    LeaseRepository, JobRepository, AccountRepository,
    AuditLogRepository, QuotaLedgerRepository,
)
from providers.registry import get_adapter
from providers.base import ProviderResult

logger = logging.getLogger("fgt.scheduler.leases")

# Lease statuses: pending -> starting -> running -> checkpointing -> completed/failed/expired/cancelled


class LeaseManager:
    """Manages GPU lease lifecycle."""

    def __init__(self):
        self.lease_repo = LeaseRepository()
        self.job_repo = JobRepository()
        self.account_repo = AccountRepository()
        self.audit_repo = AuditLogRepository()
        self.quota_repo = QuotaLedgerRepository()

    def create_lease(self, job_id: str, account_id: str,
                     provider_key: str, max_runtime_minutes: int) -> Optional[dict]:
        """Create a new lease for a job on an account.

        Args:
            job_id: Job to create lease for
            account_id: Account to use
            provider_key: Provider for the account
            max_runtime_minutes: Maximum runtime for this lease

        Returns:
            Lease dict or None on failure
        """
        # Check account is not already busy
        if self.lease_repo.is_account_busy(account_id):
            logger.error(f"Account {account_id} already has an active lease")
            return None

        # Calculate expiry time
        expires_at = (datetime.now(timezone.utc) + timedelta(minutes=max_runtime_minutes)).isoformat()

        lease = self.lease_repo.create(
            job_id=job_id,
            account_id=account_id,
            provider_key=provider_key,
            expires_at=expires_at,
        )

        if lease:
            self.audit_repo.log(
                action="create_lease",
                entity_type="lease",
                entity_id=lease["id"],
                message=f"Lease created for job {job_id} on account {account_id}",
                metadata={"provider_key": provider_key, "max_runtime_minutes": max_runtime_minutes},
            )

        return lease

    def start_lease(self, lease_id: str) -> ProviderResult:
        """Start the job on the provider via the lease.

        Calls the provider adapter's start_job method.
        """
        lease = self.lease_repo.get_by_id(lease_id)
        if not lease:
            return ProviderResult(ok=False, message=f"Lease {lease_id} not found")

        job = self.job_repo.get_by_id(lease["job_id"])
        account = self.account_repo.get_by_id(lease["account_id"])

        if not job or not account:
            return ProviderResult(ok=False, message="Job or account not found")

        # Get provider adapter
        adapter = get_adapter(lease["provider_key"])
        if not adapter:
            return ProviderResult(ok=False, message=f"No adapter for provider {lease['provider_key']}")

        # Update lease status to starting
        self.lease_repo.update(lease_id, status="starting")

        # Call provider adapter
        try:
            result = adapter.start_job(lease, job, account)
        except Exception as e:
            logger.error(f"Provider start_job failed: {e}")
            self.lease_repo.mark_failed(lease_id, str(e))
            return ProviderResult(ok=False, message=f"Provider error: {e}")

        if result.ok:
            if result.status == "manual_required":
                self.lease_repo.update(lease_id, status="pending")
                self.audit_repo.log(
                    action="start_job_manual",
                    entity_type="lease",
                    entity_id=lease_id,
                    message=result.message,
                )
            else:
                self.lease_repo.mark_running(lease_id, result.data.get("remote_job_ref"))
                self.job_repo.mark_started(job["id"])
                self.account_repo.record_use(account["id"])
                self.audit_repo.log(
                    action="start_job",
                    entity_type="lease",
                    entity_id=lease_id,
                    message=f"Job started on {account['provider_key']}/{account['label']}",
                )

        return result

    def complete_lease(self, lease_id: str, runtime_minutes: int = 0):
        """Mark a lease as completed and record usage."""
        lease = self.lease_repo.get_by_id(lease_id)
        if not lease:
            return

        self.lease_repo.mark_completed(lease_id, runtime_minutes)
        self.quota_repo.record_usage(
            account_id=lease["account_id"],
            provider_key=lease["provider_key"],
            used_minutes=runtime_minutes,
            job_id=lease["job_id"],
            lease_id=lease_id,
        )
        self.job_repo.mark_completed(lease["job_id"])

        # Put account into cooldown
        account = self.account_repo.get_by_id(lease["account_id"])
        if account and account.get("cooldown_minutes", 0) > 0:
            self.account_repo.set_cooldown(lease["account_id"], account["cooldown_minutes"])

        self.audit_repo.log(
            action="job_completed",
            entity_type="lease",
            entity_id=lease_id,
            message=f"Job completed after {runtime_minutes} minutes",
        )

    def fail_lease(self, lease_id: str, reason: str):
        """Mark a lease as failed."""
        lease = self.lease_repo.get_by_id(lease_id)
        if not lease:
            return

        # Record partial usage
        if lease.get("started_at"):
            started = datetime.fromisoformat(lease["started_at"])
            ended = datetime.now(timezone.utc)
            runtime = int((ended - started).total_seconds() / 60)
            if runtime > 0:
                self.quota_repo.record_usage(
                    account_id=lease["account_id"],
                    provider_key=lease["provider_key"],
                    used_minutes=runtime,
                    job_id=lease["job_id"],
                    lease_id=lease_id,
                )

        self.lease_repo.mark_failed(lease_id, reason)
        self.job_repo.mark_failed(lease["job_id"], reason)

        # Record error on account
        self.account_repo.record_use(lease["account_id"], error=reason)

        self.audit_repo.log(
            action="job_failed",
            entity_type="lease",
            entity_id=lease_id,
            message=f"Job failed: {reason}",
        )

    def expire_lease(self, lease_id: str):
        """Mark a lease as expired (time limit reached).

        Attempts checkpoint before expiry.
        """
        lease = self.lease_repo.get_by_id(lease_id)
        if not lease:
            return

        # Attempt checkpoint before marking expired
        job = self.job_repo.get_by_id(lease["job_id"])
        if job and job.get("checkpoint_uri"):
            adapter = get_adapter(lease["provider_key"])
            if adapter:
                try:
                    account = self.account_repo.get_by_id(lease["account_id"])
                    adapter.sync_checkpoint(lease, account, job["checkpoint_uri"])
                except Exception as e:
                    logger.warning(f"Checkpoint before expiry failed: {e}")

        # Record usage
        if lease.get("started_at"):
            started = datetime.fromisoformat(lease["started_at"])
            ended = datetime.now(timezone.utc)
            runtime = int((ended - started).total_seconds() / 60)
            if runtime > 0:
                self.quota_repo.record_usage(
                    account_id=lease["account_id"],
                    provider_key=lease["provider_key"],
                    used_minutes=runtime,
                    job_id=lease["job_id"],
                    lease_id=lease_id,
                )

        self.lease_repo.mark_expired(lease_id)

        # Put account into cooldown
        account = self.account_repo.get_by_id(lease["account_id"])
        if account and account.get("cooldown_minutes", 0) > 0:
            self.account_repo.set_cooldown(lease["account_id"], account["cooldown_minutes"])

        self.audit_repo.log(
            action="lease_expired",
            entity_type="lease",
            entity_id=lease_id,
            message="Lease expired — time limit reached",
        )

    def cancel_lease(self, lease_id: str):
        """Cancel a lease."""
        self.lease_repo.cancel(lease_id)
        lease = self.lease_repo.get_by_id(lease_id)
        if lease:
            self.job_repo.cancel(lease["job_id"])
            self.audit_repo.log(
                action="cancel_lease",
                entity_type="lease",
                entity_id=lease_id,
                message="Lease cancelled by user",
            )

    def heartbeat(self, lease_id: str):
        """Update heartbeat for a lease (indicates it's still alive)."""
        self.lease_repo.heartbeat(lease_id)

    def get_active_leases(self) -> list[dict]:
        """Get all active leases."""
        return self.lease_repo.list_active()

    def check_expired_leases(self):
        """Check for and expire any leases that have passed their expiry time."""
        now = datetime.now(timezone.utc).isoformat()
        active = self.lease_repo.list_active()
        for lease in active:
            if lease.get("expires_at") and lease["expires_at"] <= now:
                logger.info(f"Lease {lease['id']} expired — triggering expiry")
                self.expire_lease(lease["id"])
