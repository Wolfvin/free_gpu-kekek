"""Codesphere provider adapter for FamilyGPU Orchestrator.

Class A adapter — API-based automation. Uses the Codesphere API
to manage workspaces and deployments.

For MVP: start_job returns manual_required (full API workspace management is complex).
Credential validation and health check work via the whoami endpoint.

Auth: account.credentials = {
    "codesphere_token": "...",
}
Or set CODESPHERE_TOKEN env var.
"""

import os
import json
import logging
import subprocess
from typing import Optional

from providers.base import ProviderAdapter, ProviderResult
from vault import redact_text, scan_for_secrets
from handlers import with_retry

logger = logging.getLogger("fgt.providers.codesphere")

CODESPHERE_API_BASE = "https://api.codesphere.com/v1"


class CodesphereAdapter(ProviderAdapter):
    """Codesphere adapter — API-based automation.

    Codesphere provides:
      - Free tier: 1 workspace with 2 vCPU, 4 GB RAM
      - Cloud IDE with terminal access
      - Limited GPU support on paid plans
      - Team collaboration features

    For MVP, full job start/stop is manual. Credential validation,
    health checks, and capacity estimation work via the API.
    """

    provider_key: str = "codesphere"
    display_name: str = "Codesphere"
    provider_class: str = "A"
    automation_level: str = "partial_auto"
    supports_auto: bool = True
    supported_profiles: list[str] = [
        "cpu_only", "small_gpu",
    ]

    # ── Credential helpers ──────────────────────────────────────────

    def _get_token(self, account: dict) -> str:
        """Retrieve Codesphere token from account or env."""
        creds = account.get("credentials", {}) or {}
        val = creds.get("codesphere_token", "")
        if val:
            return val
        return os.environ.get("CODESPHERE_TOKEN", "")

    # ── API helpers ─────────────────────────────────────────────────

    def _api_request(self, method: str, path: str, token: str,
                     timeout: int = 15) -> Optional[dict]:
        """Make a Codesphere API request using curl via subprocess.

        Returns parsed JSON response or None on failure.
        """
        url = f"{CODESPHERE_API_BASE}{path}"
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
            logger.debug(f"Codesphere API request failed: {e}")
            return None
        except Exception as e:
            logger.debug(f"Codesphere API error: {e}")
            return None

    # ── ProviderAdapter interface ───────────────────────────────────

    @with_retry(max_retries=2, base_delay=2.0)
    def validate_credentials(self, account: dict) -> ProviderResult:
        """Validate Codesphere token by calling the user endpoint."""
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="Codesphere token not configured. Set codesphere_token via /add or CODESPHERE_TOKEN env var.",
            )

        response = self._api_request("GET", "/user", token)
        if response and isinstance(response, dict):
            username = response.get("email", response.get("username", "unknown"))
            return ProviderResult(
                ok=True, status="ok",
                message=f"Codesphere authenticated as {redact_text(username)}",
            )

        return ProviderResult(
            ok=False, status="error",
            message="Codesphere authentication failed — invalid or expired token",
        )

    @with_retry(max_retries=2, base_delay=2.0)
    def health_check(self, account: dict) -> ProviderResult:
        """Check if Codesphere API is accessible."""
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="down",
                message="No Codesphere token configured",
            )

        response = self._api_request("GET", "/user", token, timeout=10)
        if response:
            return ProviderResult(
                ok=True, status="ok",
                message="Codesphere API is accessible",
            )

        return ProviderResult(
            ok=False, status="down",
            message="Codesphere API unreachable or auth failed",
        )

    def estimate_capacity(self, account: dict) -> ProviderResult:
        """Estimate compute capacity for Codesphere.

        Codesphere free tier:
          - 1 workspace
          - 2 vCPU, 4 GB RAM
          - No GPU on free tier (GPU available on paid plans only)
          - Persistent storage included
        """
        token = self._get_token(account)
        gpu_info = {
            "gpu_type": "None (CPU-only free tier)",
            "gpu_memory_gb": 0,
            "max_runtime_minutes": 0,  # No hard limit on workspace runtime
            "supports_checkpoint": True,
        }

        if token:
            # Try to get workspace info from API
            response = self._api_request("GET", "/workspaces", token, timeout=10)
            if response and isinstance(response, list):
                for workspace in response:
                    if isinstance(workspace, dict):
                        ws_type = workspace.get("type", "")
                        state = workspace.get("status", workspace.get("state", ""))
                        if state in ("running", "active"):
                            gpu_info["workspace_type"] = ws_type
                            gpu_info["workspace_state"] = state
                            break
            elif response and isinstance(response, dict):
                workspaces = response.get("workspaces", response.get("data", []))
                if isinstance(workspaces, list):
                    for workspace in workspaces:
                        if isinstance(workspace, dict):
                            state = workspace.get("status", workspace.get("state", ""))
                            if state in ("running", "active"):
                                gpu_info["workspace_type"] = workspace.get("type", "")
                                gpu_info["workspace_state"] = state
                                break

        return ProviderResult(
            ok=True, status="ok",
            message=f"Capacity: {gpu_info['gpu_type']}, {gpu_info['gpu_memory_gb']} GB VRAM",
            data=gpu_info,
        )

    def start_job(self, lease: dict, job: dict, account: dict) -> ProviderResult:
        """Start a training job on Codesphere.

        For MVP: returns manual_required. Full Codesphere workspace management
        (creating workspaces, configuring environments, running commands) is
        complex and will be implemented in a future release.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="Codesphere token not configured",
            )

        # MVP: manual start required
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="This provider requires manual runtime start in MVP.",
            data={
                "instructions": (
                    "1. Go to https://codesphere.com\n"
                    "2. Open or create a workspace\n"
                    "3. Upload your training script to the workspace\n"
                    "4. Run the script in the workspace terminal\n"
                    "5. Use /confirm in TUI to start the countdown timer"
                ),
            },
        )

    def stop_job(self, lease: dict, account: dict) -> ProviderResult:
        """Stop a running job on Codesphere.

        For MVP: attempts API stop, falls back to manual instruction.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="Codesphere token not configured",
            )

        # Try to stop the workspace via API
        workspace_id = lease.get("provider_data", {}).get("workspace_id", "")
        if workspace_id:
            self._api_request(
                "POST",
                f"/workspaces/{workspace_id}/stop",
                token, timeout=10,
            )

        return ProviderResult(
            ok=True, status="stopped",
            message="Stop request sent. Verify in Codesphere console.",
        )

    @with_retry(max_retries=2, base_delay=1.0)
    def fetch_logs(self, lease: dict, account: dict) -> ProviderResult:
        """Fetch recent logs from Codesphere.

        For MVP: returns a message directing to the Codesphere console.
        Full log streaming via API will be added in a future release.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="Codesphere token not configured",
            )

        # Try to get workspace logs via API
        workspace_id = lease.get("provider_data", {}).get("workspace_id", "")
        if workspace_id:
            response = self._api_request(
                "GET",
                f"/workspaces/{workspace_id}/logs",
                token, timeout=10,
            )
            if response:
                logs = response.get("logs", "") if isinstance(response, dict) else str(response)
                return ProviderResult(
                    ok=True, status="ok",
                    message="Fetched logs from Codesphere",
                    data={"logs": redact_text(logs)},
                )

        return ProviderResult(
            ok=True, status="ok",
            message="Logs not available via API in MVP. Check console: https://codesphere.com",
            data={"logs": "Logs available in Codesphere workspace terminal"},
        )

    def sync_checkpoint(self, lease: dict, account: dict,
                        checkpoint_uri: str) -> ProviderResult:
        """Sync checkpoints to/from Codesphere.

        For MVP: returns manual instruction for checkpoint sync.
        Codesphere workspaces have persistent storage.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="Codesphere token not configured",
            )

        # MVP: manual checkpoint sync
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message=(
                "Checkpoint sync is manual in MVP. "
                "Codesphere workspaces have persistent storage by default."
            ),
            data={
                "instructions": (
                    "1. Save checkpoints within your workspace storage\n"
                    "2. Workspaces persist data across restarts\n"
                    "3. Use Codesphere deploy to share artifacts\n"
                    "4. Download files via the workspace file browser"
                ),
            },
        )
