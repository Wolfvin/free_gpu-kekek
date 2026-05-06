"""Agent API for FamilyGPU Orchestrator.

Provides the interface for AI agents to request GPU compute
without accessing credentials directly.

MVP starts with a local Python function, then adds HTTP endpoints.
"""

import logging
from typing import Optional

from db.connection import init_db
from db.repositories import JobRepository, LeaseRepository, AccountRepository, AuditLogRepository
from scheduler.request import JobRequest, JobRequestResult, FailureReason
from scheduler.selector import AccountSelector
from scheduler.leases import LeaseManager
from scheduler.failover import FailoverManager

logger = logging.getLogger("fgt.api")


class GPUSchedulerAPI:
    """Main API for requesting GPU compute.

    This is the ONLY interface for AI agents to request resources.
    Agents cannot access credentials, accounts, or provider internals.

    Usage:
        api = GPUSchedulerAPI()
        result = api.request_gpu(job_request)
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the API.

        Args:
            db_path: Optional path to SQLite database. If None, uses default.
        """
        init_db(db_path)
        self.job_repo = JobRepository()
        self.lease_repo = LeaseRepository()
        self.account_repo = AccountRepository()
        self.audit_repo = AuditLogRepository()
        self.selector = AccountSelector()
        self.lease_manager = LeaseManager()
        self.failover = FailoverManager()

    def request_gpu(self, job_request: JobRequest) -> JobRequestResult:
        """Request GPU compute for a training job.

        This is the primary entry point for AI agents.

        The agent provides a job request specifying what it needs
        (GPU profile, runtime, checkpoint URI, etc.) and the
        scheduler selects the best available account.

        The agent NEVER sees which account is used or the credentials.

        Args:
            job_request: Job specification from the agent

        Returns:
            JobRequestResult with status, job_id, lease_id, and provider info
        """
        # Validate request
        errors = job_request.validate()
        if errors:
            return JobRequestResult(
                status="rejected",
                failure_reason=FailureReason.CHECKPOINT_REQUIRED_MISSING,
                message=f"Invalid job request: {'; '.join(errors)}",
            )

        # Select best account
        account, failure_reason = self.selector.select(job_request)
        if not account:
            logger.warning(f"No account available for job {job_request.job_name}: {failure_reason}")
            return JobRequestResult(
                status="rejected",
                failure_reason=failure_reason,
                message=f"No GPU available: {failure_reason.value}",
            )

        # Create job in database
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
            created_by="agent",
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

        if not lease:
            self.job_repo.mark_failed(job["id"], "Failed to create lease")
            return JobRequestResult(
                status="rejected",
                failure_reason=FailureReason.ALL_ACCOUNTS_BUSY,
                message="Failed to create lease — account may be busy",
            )

        # Start the job via provider adapter
        result = self.lease_manager.start_lease(lease["id"])

        if result.ok:
            self.audit_repo.log(
                action="request_gpu",
                entity_type="job",
                entity_id=job["id"],
                message=f"GPU allocated: {account['provider_key']}/{account['label']}",
                metadata={"lease_id": lease["id"]},
            )

            return JobRequestResult(
                status="accepted",
                job_id=job["id"],
                lease_id=lease["id"],
                provider=account["provider_key"],
                account_owner=account.get("owner_id", "unknown"),
                estimated_runtime_minutes=job_request.max_runtime_minutes,
                message=result.message,
            )
        else:
            self.job_repo.mark_failed(job["id"], result.message)
            return JobRequestResult(
                status="rejected",
                job_id=job["id"],
                failure_reason=FailureReason.CREDENTIAL_INVALID,
                message=f"Failed to start job: {result.message}",
            )

    def get_job_status(self, job_id: str) -> Optional[dict]:
        """Get the status of a job.

        Agents can poll this to check progress.
        No credentials or account details are exposed.
        """
        job = self.job_repo.get_by_id(job_id)
        if not job:
            return None

        # Build a safe status response (no credential leakage)
        status = {
            "job_id": job["id"],
            "name": job["name"],
            "status": job["status"],
            "gpu_profile": job["gpu_profile"],
            "priority": job["priority"],
            "created_at": job["created_at"],
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "failure_reason": job.get("failure_reason"),
        }

        # Include lease info if active (no account credentials)
        lease = self.lease_repo.get_active_for_job(job_id)
        if lease:
            status["lease"] = {
                "lease_id": lease["id"],
                "provider": lease["provider_key"],
                "status": lease["status"],
                "started_at": lease.get("started_at"),
                "expires_at": lease.get("expires_at"),
            }
            # Include provider status label (AUTO vs MANUAL)
            from providers.registry import get_adapter
            adapter = get_adapter(lease["provider_key"])
            if adapter:
                status["lease"]["provider_type"] = "AUTO" if adapter.supports_auto else "MANUAL"

        return status

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: Job to cancel

        Returns:
            True if cancelled, False if job not found or already completed
        """
        job = self.job_repo.get_by_id(job_id)
        if not job:
            return False

        if job["status"] not in ("queued", "running", "starting"):
            return False

        # Cancel any active lease
        lease = self.lease_repo.get_active_for_job(job_id)
        if lease:
            self.lease_manager.cancel_lease(lease["id"])

        self.job_repo.cancel(job_id)
        self.audit_repo.log(
            action="cancel_job",
            entity_type="job",
            entity_id=job_id,
            message="Job cancelled by agent",
        )
        return True

    def list_jobs(self, status: Optional[str] = None) -> list[dict]:
        """List jobs (safe for agent consumption — no credentials)."""
        jobs = self.job_repo.list_all(status=status)
        return [
            {
                "job_id": j["id"],
                "name": j["name"],
                "status": j["status"],
                "gpu_profile": j["gpu_profile"],
                "priority": j["priority"],
                "created_at": j["created_at"],
            }
            for j in jobs
        ]

    def get_available_capacity(self) -> dict:
        """Get current available GPU capacity.

        Useful for agents to check before submitting a job.
        """
        accounts = self.account_repo.list_available()
        capacity = {
            "total_accounts": len(accounts),
            "by_provider": {},
            "by_profile": {},
        }

        from providers.registry import get_adapter

        for account in accounts:
            pk = account["provider_key"]
            adapter = get_adapter(pk)
            if not adapter:
                continue

            # Count by provider
            if pk not in capacity["by_provider"]:
                capacity["by_provider"][pk] = {
                    "name": adapter.display_name,
                    "auto": adapter.supports_auto,
                    "count": 0,
                }
            capacity["by_provider"][pk]["count"] += 1

            # Count by supported profile
            for profile in adapter.supported_profiles:
                if profile not in capacity["by_profile"]:
                    capacity["by_profile"][profile] = 0
                capacity["by_profile"][profile] += 1

        return capacity
