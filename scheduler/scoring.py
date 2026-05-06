"""Account scoring algorithm for FamilyGPU Orchestrator.

The scoring system determines which account is the best choice
for a given job request, considering quota, priority, health,
and provider capabilities.
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger("fgt.scheduler.scoring")


def calculate_score(
    account: dict,
    job_request: dict,
    remaining_daily_minutes: int,
    remaining_weekly_minutes: int,
    health_status: str,
    provider_adapter_info: dict,
) -> float:
    """Calculate a score for an account given a job request.

    Higher score = better match. The scheduler picks the highest-scoring account.

    Scoring factors:
      - Account priority (0-10) — weighted heavily
      - Remaining daily quota — more remaining = better
      - Remaining weekly quota — more remaining = better
      - Health status — "ok" gets bonus, "error" gets penalty
      - Provider supports auto — AUTO gets bonus for agent-friendliness
      - Automation level — full_auto gets bonus, manual gets none
      - Recent errors — penalty for accounts with recent failures
      - Job priority — high priority jobs get extra weight from high-priority accounts

    Args:
        account: Account dict from database
        job_request: JobRequest dict
        remaining_daily_minutes: Daily quota remaining
        remaining_weekly_minutes: Weekly quota remaining
        health_status: "ok", "degraded", "down", or "unknown"
        provider_adapter_info: Dict with supports_auto, provider_class, automation_level

    Returns:
        Float score (higher = better match)
    """
    score = 0.0

    # 1. Account priority (0-10, default 5)
    priority = account.get("priority", 5)
    score += priority * 10

    # 2. Remaining daily quota — cap at 240 minutes for scoring
    score += min(remaining_daily_minutes, 240) / 10

    # 3. Remaining weekly quota — cap at 1200 minutes for scoring
    score += min(remaining_weekly_minutes, 1200) / 50

    # 4. Health status bonus/penalty
    if health_status == "ok":
        score += 20
    elif health_status == "unknown":
        score += 5  # Not penalized, but not boosted either
    elif health_status == "degraded":
        score -= 5
    elif health_status == "down":
        score -= 30

    # 5. AUTO provider bonus (agent-friendly)
    if provider_adapter_info.get("supports_auto"):
        score += 15

    # 6. Provider class preference: A > B > C
    provider_class = provider_adapter_info.get("provider_class", "C")
    class_bonus = {"A": 10, "B": 5, "C": 0}
    score += class_bonus.get(provider_class, 0)

    # 7. Automation level bonus
    automation_level = provider_adapter_info.get("automation_level", "manual")
    auto_bonus = {"full_auto": 20, "partial_auto": 10, "manual": 0}
    score += auto_bonus.get(automation_level, 0)

    # 8. Recent error penalty
    if account.get("last_error_at"):
        score -= 10

    # 9. Job priority interaction
    job_priority = job_request.get("priority", "normal")
    if job_priority == "high":
        score += priority * 5  # High priority jobs get extra from high-priority accounts

    # 10. Long-running preference for provider_class A
    gpu_profile = job_request.get("gpu_profile", "small_gpu")
    if gpu_profile == "long_running" and provider_class == "A":
        score += 15  # Class A providers are better for long-running

    # 11. Penalize if remaining quota barely covers the job
    max_runtime = job_request.get("max_runtime_minutes", 180)
    if remaining_daily_minutes < max_runtime * 1.5:
        score -= 5  # Tight fit — might not have buffer

    logger.debug(
        f"Score for {account.get('id', '?')} "
        f"(priority={priority}, daily_left={remaining_daily_minutes}, "
        f"weekly_left={remaining_weekly_minutes}, health={health_status}): "
        f"{score:.1f}"
    )

    return score
