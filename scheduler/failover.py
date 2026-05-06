"""Failover logic for FamilyGPU Orchestrator.

When a job fails or a lease expires before the job is complete,
the failover manager attempts to continue the job on a different
account/provider.
"""

import logging
from typing import Optional

from db.repositories import (
    JobRepository, LeaseRepository, AccountRepository,
    AuditLogRepository,
)
from scheduler.request import JobRequest, JobRequestResult, FailureReason
from scheduler.selector import AccountSelector
from scheduler.leases import LeaseManager

logger = logging.getLogger("fgt.scheduler.failover")


class FailoverManager:
    """Manages job failover when leases expire or fail.

    Failover flow:
    1. Detect failed/expired lease
    2. Check if job can be resumed (has checkpoint)
    3. Find a new account via AccountSelector
    4. Create new lease on new account
    5. Start job on new provider (resume from checkpoint)
    6. Log failover event
    """

    MAX_FAILOVER_ATTEMPTS = 3

    def __init__(self):
        self.job_repo = JobRepository()
        self.lease_repo = LeaseRepository()
        self.account_repo = AccountRepository()
        self.audit_repo = AuditLogRepository()
        self.selector = AccountSelector()
        self.lease_manager = LeaseManager()

    def attempt_failover(self, job_id: str, failed_lease_id: str) -> Optional[JobRequestResult]:
        """Attempt to failover a job to a new account.

        Args:
            job_id: The job that needs failover
            failed_lease_id: The lease that failed/expired

        Returns:
            JobRequestResult if failover succeeded, None if no account available
        """
        job = self.job_repo.get_by_id(job_id)
        if not job:
            logger.error(f"Failover: Job {job_id} not found")
            return None

        # Check if job has checkpoint (required for resume)
        if not job.get("checkpoint_uri"):
            logger.warning(f"Failover: Job {job_id} has no checkpoint_uri — cannot resume")
            self.job_repo.mark_failed(job_id, "Job failed and no checkpoint available for resume")
            return None

        # Count previous failover attempts
        previous_leases = self.lease_repo.list_all()
        failover_count = sum(
            1 for l in previous_leases
            if l["job_id"] == job_id and l["status"] in ("failed", "expired")
        )

        if failover_count >= self.MAX_FAILOVER_ATTEMPTS:
            logger.warning(
                f"Failover: Job {job_id} exceeded max failover attempts "
                f"({failover_count}/{self.MAX_FAILOVER_ATTEMPTS})"
            )
            self.job_repo.mark_failed(job_id, f"Exceeded max failover attempts ({self.MAX_FAILOVER_ATTEMPTS})")
            return None

        # Build a job request for the selector
        request = JobRequest(
            job_name=job["name"],
            gpu_profile=job["gpu_profile"],
            max_runtime_minutes=job["max_runtime_minutes"],
            priority=job["priority"],
            checkpoint_uri=job["checkpoint_uri"],
            entrypoint=job["entrypoint"],
            args=job.get("args") or {},
            allow_providers=job.get("allow_providers") or [],
            deny_providers=job.get("deny_providers") or [],
            checkpoint_required=True,  # Always required for failover
        )

        # Select a new account (excluding the one that just failed)
        account, failure_reason = self.selector.select(request)

        if not account:
            logger.warning(f"Failover: No account available for job {job_id}: {failure_reason}")
            self.job_repo.mark_failed(
                job_id,
                f"Failover failed — no account available: {failure_reason.value}"
            )
            return None

        # Create new lease
        lease = self.lease_manager.create_lease(
            job_id=job_id,
            account_id=account["id"],
            provider_key=account["provider_key"],
            max_runtime_minutes=job["max_runtime_minutes"],
        )

        if not lease:
            logger.error(f"Failover: Could not create lease for job {job_id}")
            self.job_repo.mark_failed(job_id, "Failover failed — could not create new lease")
            return None

        # Update job status back to running
        self.job_repo.update(job_id, status="running")

        # Log the failover
        self.audit_repo.log(
            action="failover",
            entity_type="job",
            entity_id=job_id,
            message=f"Failover from lease {failed_lease_id} to lease {lease['id']} "
                    f"on account {account['label']} ({account['provider_key']})",
            metadata={
                "failed_lease_id": failed_lease_id,
                "new_lease_id": lease["id"],
                "new_account_id": account["id"],
                "failover_attempt": failover_count + 1,
            },
        )

        # Start the job on the new provider
        result = self.lease_manager.start_lease(lease["id"])

        return JobRequestResult(
            status="accepted" if result.ok else "rejected",
            job_id=job_id,
            lease_id=lease["id"],
            provider=account["provider_key"],
            account_owner=account.get("owner_id", "unknown"),
            estimated_runtime_minutes=job["max_runtime_minutes"],
            failure_reason=None if result.ok else FailureReason.CREDENTIAL_INVALID,
            message=result.message,
        )
