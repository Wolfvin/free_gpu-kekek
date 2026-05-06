"""Job request model and validation for FamilyGPU Orchestrator.

Defines the structure of a job request from an AI agent or user,
and the possible failure reasons when a request cannot be fulfilled.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


class FailureReason(Enum):
    """Explicit failure reasons when scheduler cannot allocate GPU."""
    NO_ACTIVE_ACCOUNTS = "no_active_accounts"
    NO_PROVIDER_MATCH = "no_provider_match"
    INSUFFICIENT_DAILY_QUOTA = "insufficient_daily_quota"
    INSUFFICIENT_WEEKLY_QUOTA = "insufficient_weekly_quota"
    ALL_ACCOUNTS_IN_COOLDOWN = "all_accounts_in_cooldown"
    ALL_ACCOUNTS_BUSY = "all_accounts_busy"
    PROVIDER_HEALTH_FAILED = "provider_health_failed"
    CREDENTIAL_INVALID = "credential_invalid"
    CHECKPOINT_REQUIRED_MISSING = "checkpoint_required_missing"
    NO_ACCOUNTS_CONFIGURED = "no_accounts_configured"


# ── GPU Profiles ────────────────────────────────────────────────

GPU_PROFILES = {
    "cpu_only": {"gpu_required": False, "description": "CPU only — no GPU needed"},
    "small_gpu": {"gpu_required": True, "description": "Entry-level GPU (T4, M4000, P100)"},
    "medium_gpu": {"gpu_required": True, "description": "Mid-range GPU (A100 40GB, V100)"},
    "high_vram_gpu": {"gpu_required": True, "description": "High VRAM GPU (A100 80GB, A6000)"},
    "long_running": {"gpu_required": True, "description": "Long-running session (12h+)"},
}


@dataclass
class JobRequest:
    """A training job request from an AI agent or user.

    This is the ONLY way for agents to request compute. Agents
    cannot select accounts or access credentials directly.
    """
    job_name: str
    gpu_profile: str = "small_gpu"
    max_runtime_minutes: int = 180
    priority: str = "normal"  # low, normal, high
    checkpoint_uri: str = ""
    entrypoint: str = "train.py"
    args: dict = field(default_factory=dict)
    allow_providers: list[str] = field(default_factory=list)
    deny_providers: list[str] = field(default_factory=list)
    checkpoint_required: bool = True

    def validate(self) -> list[str]:
        """Validate the job request. Returns list of error messages."""
        errors = []
        if not self.job_name:
            errors.append("job_name is required")
        if self.gpu_profile not in GPU_PROFILES:
            errors.append(f"Invalid gpu_profile: {self.gpu_profile}. Must be one of: {list(GPU_PROFILES.keys())}")
        if self.max_runtime_minutes <= 0:
            errors.append("max_runtime_minutes must be positive")
        if self.max_runtime_minutes > 1440:
            errors.append("max_runtime_minutes cannot exceed 1440 (24h)")
        if self.priority not in ("low", "normal", "high"):
            errors.append("priority must be low, normal, or high")
        if self.checkpoint_required and not self.checkpoint_uri:
            errors.append("checkpoint_uri is required when checkpoint_required is True")
        return errors

    def to_dict(self) -> dict:
        return {
            "job_name": self.job_name,
            "gpu_profile": self.gpu_profile,
            "max_runtime_minutes": self.max_runtime_minutes,
            "priority": self.priority,
            "checkpoint_uri": self.checkpoint_uri,
            "entrypoint": self.entrypoint,
            "args": self.args,
            "allow_providers": self.allow_providers,
            "deny_providers": self.deny_providers,
            "checkpoint_required": self.checkpoint_required,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JobRequest":
        return cls(
            job_name=d.get("job_name", ""),
            gpu_profile=d.get("gpu_profile", "small_gpu"),
            max_runtime_minutes=d.get("max_runtime_minutes", 180),
            priority=d.get("priority", "normal"),
            checkpoint_uri=d.get("checkpoint_uri", ""),
            entrypoint=d.get("entrypoint", "train.py"),
            args=d.get("args", {}),
            allow_providers=d.get("allow_providers", []),
            deny_providers=d.get("deny_providers", []),
            checkpoint_required=d.get("checkpoint_required", True),
        )


@dataclass
class JobRequestResult:
    """Result of a job request to the scheduler."""
    status: str = "rejected"  # accepted, queued, rejected
    job_id: str = ""
    lease_id: str = ""
    provider: str = ""
    account_owner: str = ""
    estimated_runtime_minutes: int = 0
    failure_reason: Optional[FailureReason] = None
    message: str = ""

    def to_dict(self) -> dict:
        result = {
            "status": self.status,
            "job_id": self.job_id,
            "lease_id": self.lease_id,
            "provider": self.provider,
            "account_owner": self.account_owner,
            "estimated_runtime_minutes": self.estimated_runtime_minutes,
            "message": self.message,
        }
        if self.failure_reason:
            result["failure_reason"] = self.failure_reason.value
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
