"""Amazon SageMaker Studio Lab provider adapter for FamilyGPU Orchestrator.

Class B adapter — Notebook/Interactive. Requires manual start.
SageMaker Studio Lab provides free GPU/CPU notebook environments
with persistent storage and no AWS account requirement.

For MVP: start_job returns manual_required. Full automation would
require browser automation or the SageMaker Studio Lab API (if available).

Auth: account.credentials = {
    "sagemaker_token": "...",  # Studio Lab session token
}

SageMaker Studio Lab free tier:
  - GPU: NVIDIA T4 (16 GB VRAM)
  - CPU: 4 cores, 16 GB RAM
  - Persistent storage: 15 GB (GPU), 15 GB (CPU)
  - Session limit: 4 hours GPU, 12 hours CPU
  - Quota: resets after session ends (no weekly limit)
"""

import logging
import os

from providers.base import ProviderAdapter, ProviderResult
from vault import redact_text, scan_for_secrets

logger = logging.getLogger("fgt.providers.sagemaker")


class SageMakerAdapter(ProviderAdapter):
    """Amazon SageMaker Studio Lab adapter — Notebook/Interactive (Class B).

    SageMaker Studio Lab provides:
      - Free GPU notebooks (NVIDIA T4, 16 GB VRAM)
      - Free CPU notebooks (4 cores, 16 GB RAM)
      - Persistent project storage across sessions
      - No AWS account needed — sign up with email

    For MVP, all operations are manual. The user must:
      - Start/stop runtimes in the browser
      - Upload scripts manually
      - Manage checkpoints via project storage
    """

    provider_key: str = "sagemaker"
    display_name: str = "Amazon SageMaker Studio Lab"
    provider_class: str = "B"
    automation_level: str = "manual"
    supports_auto: bool = False
    supported_profiles: list[str] = [
        "cpu_only", "small_gpu",
    ]

    # ── Credential helpers ──────────────────────────────────────────

    def _get_token(self, account: dict) -> str:
        """Retrieve SageMaker Studio Lab token from account."""
        creds = account.get("credentials", {}) or {}
        return creds.get("sagemaker_token", "")

    # ── ProviderAdapter interface ───────────────────────────────────

    def validate_credentials(self, account: dict) -> ProviderResult:
        """Validate SageMaker Studio Lab credentials.

        SageMaker Studio Lab uses browser-based authentication.
        The token is stored for identification only — we verify
        it exists and looks reasonable.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="SageMaker Studio Lab token not configured. Set sagemaker_token via /add.",
            )

        # Basic format check — Studio Lab tokens are typically JWT-like
        if len(token) < 10:
            return ProviderResult(
                ok=False, status="error",
                message="SageMaker token appears invalid (too short). Please re-enter.",
            )

        return ProviderResult(
            ok=True, status="ok",
            message="SageMaker Studio Lab token configured (browser auth required for validation)",
            data={"note": "Full token validation requires browser login"},
        )

    def health_check(self, account: dict) -> ProviderResult:
        """Check if SageMaker Studio Lab is accessible.

        Since there's no public health API, we verify credentials
        are configured and return a link to check status.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="down",
                message="No SageMaker token configured",
            )

        return ProviderResult(
            ok=True, status="ok",
            message="SageMaker Studio Lab is typically available (no API health check)",
            data={
                "status_url": "https://studiolab.sagemaker.aws/",
                "health_note": "Service availability depends on AWS status",
            },
        )

    def estimate_capacity(self, account: dict) -> ProviderResult:
        """Estimate GPU capacity for SageMaker Studio Lab.

        SageMaker Studio Lab free tier:
          - GPU: NVIDIA T4 (16 GB VRAM), 4 CPU cores, 16 GB RAM
          - CPU: 4 CPU cores, 16 GB RAM
          - Persistent storage: 15 GB
          - GPU session: 4 hours max
          - CPU session: 12 hours max
          - No weekly quota limit (can restart after session ends)
        """
        return ProviderResult(
            ok=True, status="ok",
            message="SageMaker Studio Lab: T4 16 GB, 4h GPU sessions",
            data={
                "gpu_type": "NVIDIA T4",
                "gpu_memory_gb": 16,
                "max_runtime_minutes": 240,  # 4 hours GPU
                "supports_checkpoint": True,
                "cpu_spec": "4 cores, 16 GB RAM",
                "persistent_storage_gb": 15,
                "cpu_session_minutes": 720,  # 12 hours CPU
                "note": "No weekly quota; can restart sessions after they end",
            },
        )

    def start_job(self, lease: dict, job: dict, account: dict) -> ProviderResult:
        """Start a training job on SageMaker Studio Lab.

        For MVP: returns manual_required. Full automation would require
        browser automation to start the runtime and upload scripts.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="SageMaker Studio Lab token not configured",
            )

        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="This provider requires manual runtime start in MVP.",
            data={
                "instructions": (
                    "1. Go to https://studiolab.sagemaker.aws/\n"
                    "2. Start a GPU runtime (if not already running)\n"
                    "3. Open the project in JupyterLab\n"
                    "4. Upload your training script via the file browser\n"
                    "5. Open a terminal and run: python train.py\n"
                    "6. Use /confirm in TUI to start the countdown timer\n"
                    "\n"
                    "Tip: SageMaker Studio Lab has persistent storage —\n"
                    "your files persist across sessions in the same project."
                ),
                "studio_url": "https://studiolab.sagemaker.aws/",
            },
        )

    def stop_job(self, lease: dict, account: dict) -> ProviderResult:
        """Stop a running job on SageMaker Studio Lab.

        For MVP: manual stop via browser.
        """
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Stop in browser: File → Shut Down, then Stop Runtime",
            data={
                "instructions": (
                    "1. In SageMaker Studio Lab, go to File → Shut Down All Kernels\n"
                    "2. Click 'Stop Runtime' in the top-right corner\n"
                    "3. Confirm the shutdown\n"
                    "4. Use /confirm in TUI to mark the job as stopped\n"
                    "\n"
                    "Note: Your files are persisted in project storage."
                ),
            },
        )

    def fetch_logs(self, lease: dict, account: dict) -> ProviderResult:
        """Fetch logs from SageMaker Studio Lab.

        For MVP: logs are only visible in the browser.
        """
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Logs not available via API. Check Studio Lab terminal for output.",
            data={
                "logs": "Manual check required — view terminal output in Studio Lab",
                "instructions": (
                    "1. Open your SageMaker Studio Lab project\n"
                    "2. Check the terminal or notebook cell output for logs\n"
                    "3. If logging to a file, check /home/studio-lab-user/ for log files"
                ),
            },
        )

    def sync_checkpoint(self, lease: dict, account: dict,
                        checkpoint_uri: str) -> ProviderResult:
        """Sync checkpoints to/from SageMaker Studio Lab.

        Studio Lab has persistent project storage (15 GB).
        Checkpoints can be saved to project storage and
        downloaded/uploaded for cross-provider sync.
        """
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Checkpoint sync via persistent storage (manual in MVP).",
            data={
                "checkpoint_uri": checkpoint_uri,
                "instructions": (
                    "1. Save checkpoints to your project directory:\n"
                    "   torch.save(model.state_dict(), '/home/studio-lab-user/checkpoints/model.pt')\n"
                    "2. Download from Studio Lab file browser (right-click → Download)\n"
                    f"3. Upload to next provider's checkpoint location: {checkpoint_uri}\n"
                    "\n"
                    "Note: Studio Lab has 15 GB persistent storage —\n"
                    "checkpoints persist across sessions in the same project."
                ),
            },
        )
