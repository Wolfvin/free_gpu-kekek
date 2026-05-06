"""NVIDIA vGPU Trial provider adapter for FamilyGPU Orchestrator.

Class C adapter — Special/Limited. Requires manual start.
NVIDIA vGPU Trial provides time-limited access to NVIDIA GPUs
via NVIDIA's cloud trial programs (NGC, LaunchPad, etc.).

For MVP: start_job returns manual_required. Full automation would
require NGC CLI or NVIDIA licensing API integration.

Auth: account.credentials = {
    "nvidia_license_key": "...",  # NVIDIA vGPU license key
}

NVIDIA vGPU Trial:
  - Time-limited GPU access (typically 30-90 day trials)
  - NVIDIA A100, V100, T4 GPUs available
  - NGC (NVIDIA GPU Cloud) container registry
  - LaunchPad for hands-on labs
  - License key required for vGPU activation
"""

import json
import logging
import os
import subprocess

from providers.base import ProviderAdapter, ProviderResult
from vault import redact_text, scan_for_secrets

logger = logging.getLogger("fgt.providers.nvidia_vgpu")

NGC_CLI_TIMEOUT = 30
NGC_API_BASE = "https://api.ngc.nvidia.com"


class NvidiaVgpuAdapter(ProviderAdapter):
    """NVIDIA vGPU Trial adapter — Special/Limited (Class C).

    NVIDIA vGPU Trial provides:
      - Time-limited access to NVIDIA data center GPUs
      - NGC container registry for optimized ML environments
      - NVIDIA LaunchPad for hands-on lab environments
      - License key activation for vGPU instances

    This is a special/limited provider because:
      - Trial access is time-limited (30-90 days)
      - License keys are single-use and expire
      - GPU allocation depends on trial availability
      - Full automation requires NGC CLI + licensing API
      - Not a persistent free tier — trials end

    For MVP, all operations are manual.
    """

    provider_key: str = "nvidia_vgpu"
    display_name: str = "NVIDIA vGPU Trial"
    provider_class: str = "C"
    supports_auto: bool = False
    supported_profiles: list[str] = [
        "cpu_only", "small_gpu", "medium_gpu",
    ]

    # ── Credential helpers ──────────────────────────────────────────

    def _get_license_key(self, account: dict) -> str:
        """Retrieve NVIDIA vGPU license key from account."""
        creds = account.get("credentials", {}) or {}
        return creds.get("nvidia_license_key", "")

    # ── API helpers ─────────────────────────────────────────────────

    def _check_ngc_cli(self) -> bool:
        """Check if NGC CLI is available."""
        try:
            result = subprocess.run(
                ["ngc", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _run_ngc_cli(self, args: list[str], license_key: str,
                     timeout: int = NGC_CLI_TIMEOUT) -> subprocess.CompletedProcess:
        """Run an NGC CLI command via subprocess.

        Args:
            args: Command arguments after 'ngc'
            license_key: NVIDIA license key (set as env var)
            timeout: Timeout in seconds

        Returns:
            CompletedProcess instance
        """
        env = os.environ.copy()
        env["NGC_API_KEY"] = license_key

        cmd = ["ngc"] + args
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            return result
        except subprocess.TimeoutExpired:
            logger.warning(f"NGC CLI timed out: {' '.join(args)}")
            return subprocess.CompletedProcess(
                args=cmd, returncode=-1,
                stdout="", stderr="Command timed out",
            )
        except FileNotFoundError:
            logger.debug("ngc CLI not found")
            return subprocess.CompletedProcess(
                args=cmd, returncode=-1,
                stdout="", stderr="ngc CLI not installed",
            )

    # ── ProviderAdapter interface ───────────────────────────────────

    def validate_credentials(self, account: dict) -> ProviderResult:
        """Validate NVIDIA vGPU license key.

        The license key is validated by checking its format and
        optionally testing against the NGC API.
        """
        license_key = self._get_license_key(account)
        if not license_key:
            return ProviderResult(
                ok=False, status="error",
                message="NVIDIA license key not configured. Set nvidia_license_key via /add.",
            )

        # Basic format check — NVIDIA license keys are typically 20+ chars
        if len(license_key) < 10:
            return ProviderResult(
                ok=False, status="error",
                message="NVIDIA license key appears invalid (too short). Please re-enter.",
            )

        # Try NGC API if CLI is available
        if self._check_ngc_cli():
            result = self._run_ngc_cli(["config", "validate"], license_key, timeout=10)
            if result.returncode == 0:
                return ProviderResult(
                    ok=True, status="ok",
                    message="NVIDIA NGC credentials validated successfully",
                )
            return ProviderResult(
                ok=False, status="error",
                message=f"NVIDIA NGC validation failed: {result.stderr.strip()}",
            )

        # Fallback: format-only validation
        return ProviderResult(
            ok=True, status="ok",
            message="NVIDIA license key configured (NGC CLI not available for full validation)",
            data={"note": "Full validation requires ngc CLI or browser login to NGC"},
        )

    def health_check(self, account: dict) -> ProviderResult:
        """Check if NVIDIA vGPU Trial / NGC is accessible."""
        license_key = self._get_license_key(account)
        if not license_key:
            return ProviderResult(
                ok=False, status="down",
                message="No NVIDIA license key configured",
            )

        # Try NGC API
        if self._check_ngc_cli():
            result = self._run_ngc_cli(["config", "validate"], license_key, timeout=10)
            if result.returncode == 0:
                return ProviderResult(
                    ok=True, status="ok",
                    message="NVIDIA NGC is accessible",
                )
            return ProviderResult(
                ok=False, status="down",
                message="NVIDIA NGC unreachable or license key invalid",
            )

        # Fallback: assume available if key is configured
        return ProviderResult(
            ok=True, status="ok",
            message="NVIDIA NGC is typically available (no API health check without ngc CLI)",
            data={
                "status_url": "https://status.nvidia.com/",
                "ngc_url": "https://ngc.nvidia.com/",
            },
        )

    def estimate_capacity(self, account: dict) -> ProviderResult:
        """Estimate GPU capacity for NVIDIA vGPU Trial.

        NVIDIA vGPU Trial capacity varies by program:
          - LaunchPad: A100 (40/80 GB), V100 (16/32 GB), T4 (16 GB)
          - NGC Free Tier: Limited container pulls, no GPU allocation
          - Trial GPUs: Typically T4 or V100
          - Session limits: 1-8 hours depending on trial program
          - License expiry: 30-90 days from activation

        NOTE: Actual GPU availability depends on the specific trial
        program and current allocation status.
        """
        return ProviderResult(
            ok=True, status="ok",
            message="NVIDIA vGPU Trial: V100/T4 (trial-dependent), time-limited access",
            data={
                "gpu_type": "NVIDIA V100 / T4 (trial-dependent)",
                "gpu_memory_gb": 16,  # V100 16GB / T4 16GB
                "max_runtime_minutes": 480,  # 8 hours typical trial session
                "supports_checkpoint": True,
                "trial_duration_days": "30-90",
                "possible_gpus": [
                    "A100 (40/80 GB) — LaunchPad only",
                    "V100 (16/32 GB) — most common trial",
                    "T4 (16 GB) — entry-level trial",
                ],
                "ngc_containers": True,
                "note": (
                    "GPU type and duration depend on specific trial program. "
                    "Check NGC portal for your allocation details."
                ),
            },
        )

    def start_job(self, lease: dict, job: dict, account: dict) -> ProviderResult:
        """Start a training job on NVIDIA vGPU Trial.

        For MVP: returns manual_required. Full automation would use
        NGC CLI to launch containers on allocated GPU instances.
        """
        license_key = self._get_license_key(account)
        if not license_key:
            return ProviderResult(
                ok=False, status="error",
                message="NVIDIA license key not configured",
            )

        gpu_profile = job.get("gpu_profile", "small_gpu")

        # Build NGC-specific instructions
        container_info = ""
        if gpu_profile != "cpu_only":
            container_info = (
                "\n"
                "Recommended NGC containers:\n"
                "  PyTorch:  nvcr.io/nvidia/pytorch:24.01-py3\n"
                "  TensorFlow: nvcr.io/nvidia/tensorflow:24.01-tf2-py3\n"
                "\n"
                "Pull and run:\n"
                "  docker pull nvcr.io/nvidia/pytorch:24.01-py3\n"
                "  docker run --gpus all -it nvcr.io/nvidia/pytorch:24.01-py3\n"
            )

        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="This provider requires manual runtime start in MVP.",
            data={
                "instructions": (
                    "1. Go to https://ngc.nvidia.com/\n"
                    "2. Activate your vGPU trial license\n"
                    "3. Launch a GPU instance from the NGC console\n"
                    "4. SSH into the instance or use Jupyter Lab\n"
                    "5. Upload your training script\n"
                    "6. Run: python train.py (or use NGC container)"
                    f"{container_info}"
                    "7. Use /confirm in TUI to start the countdown timer\n"
                    "\n"
                    "⚠ Remember: Trial access is time-limited!\n"
                    "Check your license expiry date in the NGC portal."
                ),
                "ngc_url": "https://ngc.nvidia.com/",
                "launchpad_url": "https://launchpad.nvidia.com/",
            },
        )

    def stop_job(self, lease: dict, account: dict) -> ProviderResult:
        """Stop a running job on NVIDIA vGPU Trial.

        For MVP: manual stop via SSH or console.
        """
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Stop via SSH (Ctrl+C) or terminate instance in NGC console",
            data={
                "instructions": (
                    "1. SSH into your GPU instance\n"
                    "2. Press Ctrl+C to stop the training process\n"
                    "3. Or terminate the instance in the NGC console\n"
                    "4. Use /confirm in TUI to mark the job as stopped\n"
                    "\n"
                    "Note: Terminating the instance stops all running jobs.\n"
                    "Your trial license time continues until expiry."
                ),
            },
        )

    def fetch_logs(self, lease: dict, account: dict) -> ProviderResult:
        """Fetch logs from NVIDIA vGPU Trial instance.

        For MVP: logs are only accessible via SSH.
        """
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Logs accessible via SSH or NGC console (manual in MVP).",
            data={
                "logs": "Manual check required — SSH into instance or check NGC console",
                "instructions": (
                    "1. SSH into your GPU instance\n"
                    "2. Check training output in the terminal\n"
                    "3. If logging to a file: cat /path/to/logfile\n"
                    "4. Or check NGC console for instance metrics and logs"
                ),
            },
        )

    def sync_checkpoint(self, lease: dict, account: dict,
                        checkpoint_uri: str) -> ProviderResult:
        """Sync checkpoints to/from NVIDIA vGPU Trial instance.

        Checkpoints can be transferred via SCP or NGC storage.
        """
        license_key = self._get_license_key(account)
        if not license_key:
            return ProviderResult(
                ok=False, status="error",
                message="NVIDIA license key not configured",
            )

        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Checkpoint sync via SCP / NGC storage (manual in MVP).",
            data={
                "checkpoint_uri": checkpoint_uri,
                "instructions": (
                    "1. Save checkpoints on the instance:\n"
                    "   torch.save(model.state_dict(), '/workspace/checkpoints/model.pt')\n"
                    "2. Download via SCP:\n"
                    "   scp user@instance:/workspace/checkpoints/model.pt ./\n"
                    "3. Or use NGC storage for persistence:\n"
                    "   ngc storage upload /workspace/checkpoints/ nvr-familygpu://checkpoints/\n"
                    f"4. Upload to next provider's checkpoint location: {checkpoint_uri}\n"
                    "\n"
                    "⚠ Instance storage is ephemeral — download checkpoints\n"
                    "before terminating the instance!"
                ),
            },
        )
