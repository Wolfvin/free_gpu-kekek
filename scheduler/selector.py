"""Account selector for FamilyGPU Orchestrator.

Selects the best account for a given job request based on:
  - Provider capability match (GPU profile)
  - Quota availability (daily + weekly)
  - Cooldown status
  - Active lease conflicts (one lease per account)
  - Owner policy (allow/deny providers)
  - Health status
  - Priority scoring
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from db.repositories import (
    AccountRepository, LeaseRepository, QuotaLedgerRepository,
    HealthRepository, AuditLogRepository,
)
from providers.registry import get_adapter
from scheduler.request import JobRequest, FailureReason
from scheduler.scoring import calculate_score

logger = logging.getLogger("fgt.scheduler.selector")


class AccountSelector:
    """Selects the best available account for a job request."""

    def __init__(self):
        self.account_repo = AccountRepository()
        self.lease_repo = LeaseRepository()
        self.quota_repo = QuotaLedgerRepository()
        self.health_repo = HealthRepository()
        self.audit_repo = AuditLogRepository()

    def select(self, job_request: JobRequest) -> Optional[tuple[dict, FailureReason]]:
        """Select the best account for a job request.

        Returns:
            (account_dict, None) on success, or (None, FailureReason) on failure.
        """
        accounts = self.account_repo.list_all(status="active")

        if not accounts:
            return None, FailureReason.NO_ACCOUNTS_CONFIGURED

        candidates = []

        for account in accounts:
            # 1. Check if account is active
            if account["status"] != "active":
                continue

            # 2. Check if account has an active lease (busy)
            if self.lease_repo.is_account_busy(account["id"]):
                continue

            # 3. Check cooldown
            if account.get("cooldown_until"):
                if account["cooldown_until"] > datetime.now(timezone.utc).isoformat():
                    continue

            # 4. Check provider is allowed by job request
            provider_key = account["provider_key"]
            if job_request.allow_providers and provider_key not in job_request.allow_providers:
                continue
            if job_request.deny_providers and provider_key in job_request.deny_providers:
                continue

            # 5. Check provider supports the requested GPU profile
            adapter = get_adapter(provider_key)
            if not adapter:
                continue
            if not adapter.supports_profile(job_request.gpu_profile):
                continue

            # 6. Check quota
            daily_limit = account.get("daily_limit_minutes", 120)
            weekly_limit = account.get("weekly_limit_minutes", 600)
            remaining_daily = self.quota_repo.remaining_daily(account["id"], daily_limit)
            remaining_weekly = self.quota_repo.remaining_weekly(account["id"], weekly_limit)

            if remaining_daily < job_request.max_runtime_minutes:
                continue
            if remaining_weekly < job_request.max_runtime_minutes:
                continue

            # 7. Check health
            health = self.health_repo.get_latest(account["id"])
            health_status = health.get("status", "unknown") if health else "unknown"
            if health_status == "down":
                continue

            # 8. Calculate score
            score = calculate_score(
                account=account,
                job_request=job_request.to_dict(),
                remaining_daily_minutes=remaining_daily,
                remaining_weekly_minutes=remaining_weekly,
                health_status=health_status,
                provider_adapter_info={
                    "supports_auto": adapter.supports_auto,
                    "provider_class": adapter.provider_class,
                },
            )

            candidates.append((score, account))

        if not candidates:
            # Determine specific failure reason
            return None, self._diagnose_failure(accounts, job_request)

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_account = candidates[0][1]

        logger.info(
            f"Selected account: {best_account['label']} "
            f"({best_account['provider_key']}) "
            f"score={candidates[0][0]:.1f}"
        )

        return best_account, None

    def _diagnose_failure(self, accounts: list[dict], job_request: JobRequest) -> FailureReason:
        """Determine why no account was selected."""
        if not accounts:
            return FailureReason.NO_ACCOUNTS_CONFIGURED

        has_active = any(a["status"] == "active" for a in accounts)
        if not has_active:
            return FailureReason.NO_ACTIVE_ACCOUNTS

        # Check if any account matches provider filter
        if job_request.allow_providers:
            matching = [a for a in accounts if a["provider_key"] in job_request.allow_providers]
            if not matching:
                return FailureReason.NO_PROVIDER_MATCH

        # Check if all accounts are busy
        active_accounts = [a for a in accounts if a["status"] == "active"]
        all_busy = all(self.lease_repo.is_account_busy(a["id"]) for a in active_accounts)
        if all_busy:
            return FailureReason.ALL_ACCOUNTS_BUSY

        # Check if all accounts are in cooldown
        now = datetime.now(timezone.utc).isoformat()
        all_cooling = all(
            a.get("cooldown_until") and a["cooldown_until"] > now
            for a in active_accounts
        )
        if all_cooling:
            return FailureReason.ALL_ACCOUNTS_IN_COOLDOWN

        # Check quota
        quota_repo = self.quota_repo
        all_daily_exhausted = all(
            quota_repo.remaining_daily(a["id"], a.get("daily_limit_minutes", 120)) < job_request.max_runtime_minutes
            for a in active_accounts
        )
        if all_daily_exhausted:
            return FailureReason.INSUFFICIENT_DAILY_QUOTA

        all_weekly_exhausted = all(
            quota_repo.remaining_weekly(a["id"], a.get("weekly_limit_minutes", 600)) < job_request.max_runtime_minutes
            for a in active_accounts
        )
        if all_weekly_exhausted:
            return FailureReason.INSUFFICIENT_WEEKLY_QUOTA

        # Check health
        health_repo = self.health_repo
        all_down = all(
            (lambda h: h and h.get("status") == "down")(health_repo.get_latest(a["id"]))
            for a in active_accounts
        )
        if all_down:
            return FailureReason.PROVIDER_HEALTH_FAILED

        # Default
        return FailureReason.NO_ACTIVE_ACCOUNTS
