"""Intel Developer Cloud provider adapter for FamilyGPU Orchestrator.

Class C adapter — Special/Limited. Requires manual start.
Intel Developer Cloud provides access to Intel GPUs (Arc, Max)
and CPUs for development and testing.

For MVP: start_job returns manual_required. Full automation would
require SSH access or the Intel DevCloud CLI.

Auth: account.credentials = {
    "intel_token": "...",  # Intel Developer Cloud access token
}

Intel Developer Cloud free tier:
  - Intel Xeon CPUs (4th Gen)
  - Intel Arc A770 GPUs (16 GB VRAM) — limited availability
  - Intel Data Center GPU Max — limited free access
  - Jupyter Lab notebooks
  - SSH access available
  - Session limits: varies by queue
"""

import logging
import os
import subprocess

from providers.base import ProviderAdapter, ProviderResult
from vault import redact_text, scan_for_secrets

logger = logging.getLogger("fgt.providers.intel_devcloud")

INTEL_CLI_TIMEOUT = 30


class IntelDevcloudAdapter(ProviderAdapter):
    """Intel Developer Cloud adapter — Special/Limited (Class C).

    Intel Developer Cloud provides:
      - Free access to Intel Xeon CPUs (4th Gen Scalable)
      - Intel Arc A770 GPUs (16 GB VRAM) — limited free availability
      - Intel Data Center GPU Max (limited)
      - Jupyter Lab environment
      - SSH access for advanced users
      - OneAPI toolkit pre-installed

    This is a special/limited provider because:
      - GPU access is limited and queue-based
      - Free tier has strict time limits
      - Intel GPUs require specific optimizations (SYCL, oneAPI)
      - Not all PyTorch/TensorFlow ops are optimized for Intel GPUs

    For MVP, all operations are manual.
    """

    provider_key: str = "intel_devcloud"
    display_name: str = "Intel Developer Cloud"
    provider_class: str = "C"
    automation_level: str = "manual"
    supports_auto: bool = False
    supported_profiles: list[str] = [
        "cpu_only", "small_gpu",
    ]

    # ── Credential helpers ──────────────────────────────────────────

    def _get_token(self, account: dict) -> str:
        """Retrieve Intel Developer Cloud token from account."""
        creds = account.get("credentials", {}) or {}
        return creds.get("intel_token", "")

    # ── ProviderAdapter interface ───────────────────────────────────

    def validate_credentials(self, account: dict) -> ProviderResult:
        """Validate Intel Developer Cloud credentials.

        The Intel Developer Cloud uses browser-based auth with
        an access token. We verify the token exists and looks valid.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="Intel Developer Cloud token not configured. Set intel_token via /add.",
            )

        if len(token) < 8:
            return ProviderResult(
                ok=False, status="error",
                message="Intel token appears invalid (too short). Please re-enter.",
            )

        return ProviderResult(
            ok=True, status="ok",
            message="Intel Developer Cloud token configured (browser auth required for full validation)",
            data={"note": "Full token validation requires browser login to Intel Developer Cloud"},
        )

    def health_check(self, account: dict) -> ProviderResult:
        """Check if Intel Developer Cloud is accessible.

        No public health API — we verify the token is configured
        and return a link to check service status.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="down",
                message="No Intel Developer Cloud token configured",
            )

        return ProviderResult(
            ok=True, status="ok",
            message="Intel Developer Cloud is typically available (no API health check)",
            data={
                "status_url": "https://devcloud.intel.com/",
                "health_note": "Service availability depends on Intel infrastructure status",
            },
        )

    def estimate_capacity(self, account: dict) -> ProviderResult:
        """Estimate GPU capacity for Intel Developer Cloud.

        Intel Developer Cloud free tier:
          - CPU: Intel Xeon 4th Gen (Sapphire Rapids), up to 56 cores
          - GPU: Intel Arc A770 (16 GB VRAM) — limited free availability
          - Intel Data Center GPU Max (128 GB HBM) — very limited free access
          - Queue-based GPU allocation
          - Session limits vary by queue (typically 1-4 hours for free)
          - oneAPI / SYCL runtime pre-installed
        """
        return ProviderResult(
            ok=True, status="ok",
            message="Intel DevCloud: Arc A770 16 GB (limited), queue-based allocation",
            data={
                "gpu_type": "Intel Arc A770 / Data Center GPU Max",
                "gpu_memory_gb": 16,  # A770
                "max_runtime_minutes": 240,  # 4 hours typical
                "supports_checkpoint": True,
                "cpu_spec": "Intel Xeon 4th Gen (Sapphire Rapids)",
                "queue_based": True,
                "sycl_support": True,
                "note": (
                    "Intel GPUs require SYCL/oneAPI optimizations. "
                    "Standard CUDA code will NOT run. "
                    "Use intel_extension_for_pytorch or torch.xpu backend."
                ),
            },
        )

    def start_job(self, lease: dict, job: dict, account: dict) -> ProviderResult:
        """Start a training job on Intel Developer Cloud.

        For MVP: returns manual_required. Full automation would
        require SSH access or the Intel DevCloud CLI.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="Intel Developer Cloud token not configured",
            )

        gpu_profile = job.get("gpu_profile", "cpu_only")

        # Build Intel-specific instructions
        intel_setup = ""
        if gpu_profile != "cpu_only":
            intel_setup = (
                "\n"
                "⚠ IMPORTANT: Intel GPUs use SYCL/oneAPI, NOT CUDA!\n"
                "Install Intel Extension for PyTorch:\n"
                "  pip install intel-extension-for-pytorch\n"
                "\n"
                "Modify your script to use XPU:\n"
                "  import intel_extension_for_pytorch as ipex\n"
                "  device = 'xpu'  # instead of 'cuda'\n"
                "  model = model.to(device)\n"
            )

        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="This provider requires manual runtime start in MVP.",
            data={
                "instructions": (
                    "1. Go to https://devcloud.intel.com/\n"
                    "2. Sign in and launch a Jupyter Lab instance\n"
                    "3. Select GPU queue if available (Intel Arc A770)\n"
                    "4. Upload your training script via Jupyter file browser\n"
                    "5. Open a terminal and run: python train.py\n"
                    "6. Use /confirm in TUI to start the countdown timer"
                    f"{intel_setup}"
                ),
                "devcloud_url": "https://devcloud.intel.com/",
            },
        )

    def stop_job(self, lease: dict, account: dict) -> ProviderResult:
        """Stop a running job on Intel Developer Cloud.

        For MVP: manual stop via browser or SSH.
        """
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Stop in browser: File → Shut Down, or Ctrl+C in terminal",
            data={
                "instructions": (
                    "1. In Jupyter Lab, go to File → Shut Down\n"
                    "2. Or press Ctrl+C in the terminal running your script\n"
                    "3. Release your queue allocation if done\n"
                    "4. Use /confirm in TUI to mark the job as stopped"
                ),
            },
        )

    def fetch_logs(self, lease: dict, account: dict) -> ProviderResult:
        """Fetch logs from Intel Developer Cloud.

        For MVP: logs are only visible in the browser/terminal.
        """
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Logs not available via API. Check Jupyter terminal output.",
            data={
                "logs": "Manual check required — view terminal output in Jupyter Lab",
                "instructions": (
                    "1. Open your Jupyter Lab instance\n"
                    "2. Check the terminal output for training logs\n"
                    "3. If logging to a file, check your home directory for log files"
                ),
            },
        )

    def sync_checkpoint(self, lease: dict, account: dict,
                        checkpoint_uri: str) -> ProviderResult:
        """Sync checkpoints to/from Intel Developer Cloud.

        Intel DevCloud provides persistent home directory storage.
        Checkpoints can be saved locally and transferred via SCP
        or downloaded through Jupyter.
        """
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Checkpoint sync via home directory / SCP (manual in MVP).",
            data={
                "checkpoint_uri": checkpoint_uri,
                "instructions": (
                    "1. Save checkpoints to your home directory:\n"
                    "   torch.save(model.state_dict(), '~/checkpoints/model.pt')\n"
                    "2. Download via Jupyter file browser (right-click → Download)\n"
                    "   Or use SCP: scp devcloud:~/checkpoints/model.pt ./\n"
                    f"3. Upload to next provider's checkpoint location: {checkpoint_uri}\n"
                    "\n"
                    "Note: Home directory storage is persistent across sessions\n"
                    "but may have limited quota."
                ),
            },
        )
