"""Base provider adapter interface for FamilyGPU Orchestrator.

All providers must implement this interface. The scheduler and TUI
interact with providers exclusively through this interface, ensuring
that no provider-specific logic leaks into the core system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProviderResult:
    """Structured result from provider operations.

    All provider methods return this or a dict that gets converted to this.
    """
    ok: bool = False
    status: str = "unknown"  # running, stopped, complete, error, manual_required, unknown
    message: str = ""
    manual: bool = False  # True if provider requires manual intervention
    data: dict = field(default_factory=dict)  # Extra provider-specific data

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "status": self.status,
            "message": self.message,
            "manual": self.manual,
            "data": self.data,
        }


class ProviderAdapter(ABC):
    """Base class for all provider adapters.

    Each adapter must implement:
      - validate_credentials(account) -> ProviderResult
      - health_check(account) -> ProviderResult
      - estimate_capacity(account) -> ProviderResult
      - start_job(lease, job, account) -> ProviderResult
      - stop_job(lease, account) -> ProviderResult
      - fetch_logs(lease, account) -> ProviderResult
      - sync_checkpoint(lease, account, checkpoint_uri) -> ProviderResult

    Provider class grouping:
      A - API/SSH Runnable (oracle_cloud, gcp, paperspace, lightning_ai, codesphere)
      B - Notebook/Interactive (google_colab, kaggle, sagemaker, deepnote)
      C - Special/Limited (huggingface, intel_devcloud, nvidia_vgpu)
    """

    provider_key: str = ""
    display_name: str = ""
    provider_class: str = ""  # "A", "B", or "C"

    # Automation level: "full_auto", "partial_auto", or "manual"
    automation_level: str = "manual"

    # Whether this provider supports fully automated job execution
    supports_auto: bool = False

    # GPU profiles this provider can support
    supported_profiles: list[str] = field(default_factory=lambda: ["cpu_only", "small_gpu"])

    @abstractmethod
    def validate_credentials(self, account: dict) -> ProviderResult:
        """Validate that stored credentials work with the provider.

        Args:
            account: Account dict from database (includes credential_ref)

        Returns:
            ProviderResult with ok=True if credentials are valid.
        """
        pass

    @abstractmethod
    def health_check(self, account: dict) -> ProviderResult:
        """Check if the provider is currently accessible for this account.

        Returns:
            ProviderResult with status='ok', 'degraded', or 'down'
        """
        pass

    @abstractmethod
    def estimate_capacity(self, account: dict) -> ProviderResult:
        """Estimate available capacity for this account.

        Returns:
            ProviderResult with data dict containing:
              - gpu_type: str
              - gpu_memory_gb: int (0 if unknown)
              - max_runtime_minutes: int
              - supports_checkpoint: bool
        """
        pass

    @abstractmethod
    def start_job(self, lease: dict, job: dict, account: dict) -> ProviderResult:
        """Start a training job on the provider.

        For providers that don't support automation, return:
          ProviderResult(ok=True, status='manual_required', manual=True,
                         message='This provider requires manual runtime start in MVP.')

        Args:
            lease: Lease dict from database
            job: Job dict from database
            account: Account dict from database

        Returns:
            ProviderResult with ok=True if job was started (or manual start needed).
        """
        pass

    @abstractmethod
    def stop_job(self, lease: dict, account: dict) -> ProviderResult:
        """Stop a running job on the provider.

        Args:
            lease: Lease dict from database
            account: Account dict from database

        Returns:
            ProviderResult with ok=True if job was stopped.
        """
        pass

    @abstractmethod
    def fetch_logs(self, lease: dict, account: dict) -> ProviderResult:
        """Fetch recent logs from the running job.

        Returns:
            ProviderResult with data={'logs': str} containing the log text.
        """
        pass

    @abstractmethod
    def sync_checkpoint(self, lease: dict, account: dict, checkpoint_uri: str) -> ProviderResult:
        """Sync checkpoint data to/from the provider.

        This is called before lease expiry to save training state,
        and after lease creation to restore training state.

        Args:
            lease: Lease dict from database
            account: Account dict from database
            checkpoint_uri: URI of the checkpoint location

        Returns:
            ProviderResult with ok=True if sync succeeded.
        """
        pass

    def supports_profile(self, gpu_profile: str) -> bool:
        """Check if this provider supports a given GPU profile."""
        return gpu_profile in self.supported_profiles or gpu_profile == "any"

    def get_label(self) -> str:
        """Get a human-readable label for this provider."""
        auto_label = "AUTO" if self.supports_auto else "MANUAL"
        return f"[{auto_label}] {self.display_name}"
