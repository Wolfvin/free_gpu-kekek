"""Deepnote provider adapter for FamilyGPU Orchestrator.

Class B adapter — Notebook/Interactive. Requires manual start.
Deepnote provides collaborative cloud notebooks with optional GPU.

For MVP: start_job returns manual_required. Deepnote has an API
but full automation of notebook execution is not yet implemented.

Auth: account.credentials = {
    "deepnote_token": "...",  # Deepnote API token
}

Deepnote free tier:
  - CPU notebooks with 2 vCPU, 4 GB RAM
  - Small GPU available on paid plans
  - Collaborative real-time editing
  - Project-based organization
  - API available for programmatic access
"""

import json
import logging
import os
import subprocess

from providers.base import ProviderAdapter, ProviderResult
from vault import redact_text, scan_for_secrets

logger = logging.getLogger("fgt.providers.deepnote")

DEEPNOTE_API_BASE = "https://api.deepnote.com"
DEEPNOTE_CLI_TIMEOUT = 30


class DeepnoteAdapter(ProviderAdapter):
    """Deepnote adapter — Notebook/Interactive (Class B).

    Deepnote provides:
      - Cloud-based Jupyter notebooks
      - Collaborative real-time editing
      - Project-based organization with shared environments
      - API for programmatic notebook management
      - Small GPU on paid plans

    For MVP, all operations are manual. The user must:
      - Start/stop notebooks in the browser
      - Upload scripts manually
      - Manage checkpoints via project storage
    """

    provider_key: str = "deepnote"
    display_name: str = "Deepnote"
    provider_class: str = "B"
    automation_level: str = "manual"
    supports_auto: bool = False
    supported_profiles: list[str] = [
        "cpu_only", "small_gpu",
    ]

    # ── Credential helpers ──────────────────────────────────────────

    def _get_token(self, account: dict) -> str:
        """Retrieve Deepnote API token from account."""
        creds = account.get("credentials", {}) or {}
        return creds.get("deepnote_token", "")

    # ── API helpers ─────────────────────────────────────────────────

    def _api_request(self, method: str, path: str, token: str,
                     timeout: int = DEEPNOTE_CLI_TIMEOUT) -> dict | None:
        """Make a Deepnote API request using curl via subprocess.

        Returns parsed JSON response or None on failure.
        """
        url = f"{DEEPNOTE_API_BASE}{path}"
        try:
            result = subprocess.run(
                [
                    "curl", "-s", "-X", method,
                    "-H", f"Authorization: Bearer {token}",
                    "-H", "Content-Type: application/json",
                    url,
                ],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
            return None
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            logger.debug(f"Deepnote API request failed: {e}")
            return None
        except Exception as e:
            logger.debug(f"Deepnote API error: {e}")
            return None

    # ── ProviderAdapter interface ───────────────────────────────────

    def validate_credentials(self, account: dict) -> ProviderResult:
        """Validate Deepnote API token."""
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="Deepnote token not configured. Set deepnote_token via /add.",
            )

        # Try to validate by hitting the projects endpoint
        response = self._api_request("GET", "/v1/projects", token, timeout=10)
        if response and isinstance(response, (dict, list)):
            return ProviderResult(
                ok=True, status="ok",
                message="Deepnote API token validated successfully",
            )

        return ProviderResult(
            ok=False, status="error",
            message="Deepnote API authentication failed — invalid or expired token",
        )

    def health_check(self, account: dict) -> ProviderResult:
        """Check if Deepnote API is accessible."""
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="down",
                message="No Deepnote token configured",
            )

        response = self._api_request("GET", "/v1/projects", token, timeout=10)
        if response is not None:
            return ProviderResult(
                ok=True, status="ok",
                message="Deepnote API is accessible",
            )

        return ProviderResult(
            ok=False, status="down",
            message="Deepnote API unreachable or auth failed",
        )

    def estimate_capacity(self, account: dict) -> ProviderResult:
        """Estimate capacity for Deepnote.

        Deepnote free tier:
          - CPU: 2 vCPU, 4 GB RAM
          - Small GPU: available on paid plans only
          - Storage: included with projects
          - Session limits depend on plan
          - GPU on free tier is very limited / not guaranteed
        """
        return ProviderResult(
            ok=True, status="ok",
            message="Deepnote: 2 vCPU, 4 GB RAM (GPU on paid plans)",
            data={
                "gpu_type": "None (free tier) / Small GPU (paid)",
                "gpu_memory_gb": 0,  # Free tier has no GPU
                "max_runtime_minutes": 0,  # No hard limit published
                "supports_checkpoint": True,
                "cpu_spec": "2 vCPU, 4 GB RAM",
                "note": "GPU is only available on paid plans; free tier is CPU-only",
            },
        )

    def start_job(self, lease: dict, job: dict, account: dict) -> ProviderResult:
        """Start a training job on Deepnote.

        For MVP: returns manual_required. Full automation would use
        the Deepnote API to create a notebook and execute cells.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="Deepnote token not configured",
            )

        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="This provider requires manual runtime start in MVP.",
            data={
                "instructions": (
                    "1. Go to https://deepnote.com/\n"
                    "2. Open or create a project\n"
                    "3. Create a new notebook or open an existing one\n"
                    "4. Upload your training script to the project files\n"
                    "5. Run the script in a notebook cell: %run train.py\n"
                    "6. Use /confirm in TUI to start the countdown timer\n"
                    "\n"
                    "Tip: Deepnote supports real-time collaboration —\n"
                    "share the project with team members as needed."
                ),
                "deepnote_url": "https://deepnote.com/",
            },
        )

    def stop_job(self, lease: dict, account: dict) -> ProviderResult:
        """Stop a running job on Deepnote.

        For MVP: manual stop via browser.
        """
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Stop in browser: interrupt kernel or stop notebook",
            data={
                "instructions": (
                    "1. In Deepnote, click the 'Stop' button on the running cell\n"
                    "2. Or restart the kernel: Kernel → Restart\n"
                    "3. Use /confirm in TUI to mark the job as stopped"
                ),
            },
        )

    def fetch_logs(self, lease: dict, account: dict) -> ProviderResult:
        """Fetch logs from Deepnote.

        For MVP: logs are only visible in the browser.
        Deepnote API could potentially fetch notebook outputs.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="Deepnote token not configured",
            )

        # Try to fetch via API
        project_id = lease.get("provider_data", {}).get("project_id", "")
        if project_id:
            response = self._api_request(
                "GET", f"/v1/projects/{project_id}", token, timeout=10,
            )
            if response:
                # Could extract notebook outputs here in future
                pass

        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Logs not available via API in MVP. Check Deepnote notebook output.",
            data={
                "logs": "Manual check required — view cell outputs in Deepnote",
                "instructions": (
                    "1. Open your Deepnote notebook in the browser\n"
                    "2. Check cell outputs for training logs\n"
                    "3. Check the project terminal for detailed output"
                ),
            },
        )

    def sync_checkpoint(self, lease: dict, account: dict,
                        checkpoint_uri: str) -> ProviderResult:
        """Sync checkpoints to/from Deepnote.

        Deepnote has project-level file storage.
        Checkpoints can be saved to project files and
        downloaded/uploaded for cross-provider sync.
        """
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Checkpoint sync via project files (manual in MVP).",
            data={
                "checkpoint_uri": checkpoint_uri,
                "instructions": (
                    "1. Save checkpoints to your Deepnote project:\n"
                    "   torch.save(model.state_dict(), '/work/checkpoints/model.pt')\n"
                    "2. Download from Deepnote file browser (right-click → Download)\n"
                    f"3. Upload to next provider's checkpoint location: {checkpoint_uri}\n"
                    "\n"
                    "Note: Deepnote project files persist across sessions.\n"
                    "Use /work/ directory for persistent storage."
                ),
            },
        )
