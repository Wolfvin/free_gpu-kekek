"""Quota enforcement for FamilyGPU Orchestrator.

Checks daily and weekly quotas before allowing an account to be used.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from db.repositories import QuotaLedgerRepository, AccountRepository

logger = logging.getLogger("fgt.scheduler.quota")


class QuotaEnforcer:
    """Enforces quota limits on accounts."""

    def __init__(self):
        self.quota_repo = QuotaLedgerRepository()
        self.account_repo = AccountRepository()

    def can_use_account(self, account_id: str, max_runtime_minutes: int) -> tuple[bool, str]:
        """Check if an account has sufficient quota for a job.

        Args:
            account_id: Account ID to check
            max_runtime_minutes: Runtime needed

        Returns:
            (can_use, reason) — reason is empty if can_use is True
        """
        account = self.account_repo.get_by_id(account_id)
        if not account:
            return False, "Account not found"

        daily_limit = account.get("daily_limit_minutes", 120)
        weekly_limit = account.get("weekly_limit_minutes", 600)

        remaining_daily = self.quota_repo.remaining_daily(account_id, daily_limit)
        remaining_weekly = self.quota_repo.remaining_weekly(account_id, weekly_limit)

        if remaining_daily < max_runtime_minutes:
            return False, (
                f"Insufficient daily quota: {remaining_daily}min remaining, "
                f"{max_runtime_minutes}min needed (limit: {daily_limit}min)"
            )

        if remaining_weekly < max_runtime_minutes:
            return False, (
                f"Insufficient weekly quota: {remaining_weekly}min remaining, "
                f"{max_runtime_minutes}min needed (limit: {weekly_limit}min)"
            )

        return True, ""

    def get_remaining_daily(self, account_id: str) -> int:
        """Get remaining daily quota in minutes."""
        account = self.account_repo.get_by_id(account_id)
        if not account:
            return 0
        daily_limit = account.get("daily_limit_minutes", 120)
        return self.quota_repo.remaining_daily(account_id, daily_limit)

    def get_remaining_weekly(self, account_id: str) -> int:
        """Get remaining weekly quota in minutes."""
        account = self.account_repo.get_by_id(account_id)
        if not account:
            return 0
        weekly_limit = account.get("weekly_limit_minutes", 600)
        return self.quota_repo.remaining_weekly(account_id, weekly_limit)

    def record_usage(self, account_id: str, provider_key: str,
                     used_minutes: int, job_id: Optional[str] = None,
                     lease_id: Optional[str] = None):
        """Record usage after a lease ends."""
        self.quota_repo.record_usage(
            account_id=account_id,
            provider_key=provider_key,
            used_minutes=used_minutes,
            job_id=job_id,
            lease_id=lease_id,
        )
        logger.info(f"Recorded {used_minutes}min usage for account {account_id}")
