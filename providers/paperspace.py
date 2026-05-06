"""Paperspace Gradient provider adapter for FamilyGPU Orchestrator.

Class A adapter — API-based automation. Uses the Paperspace Gradient API
to manage notebooks and machine types.

For MVP: start_job returns manual_required (full API job management is complex).
Credential validation and health check work via the whoami endpoint.

Auth: account.credentials = {
    "paperspace_api_key": "...",
}
Or set PAPERSPACE_API_KEY env var.
"""

import os
import json
import logging
import subprocess
from typing import Optional

from providers.base import ProviderAdapter, ProviderResult
from vault import redact_text, scan_for_secrets
from handlers import with_retry

logger = logging.getLogger("fgt.providers.paperspace")

PAPERSPACE_API_BASE = "https://api.paperspace.io"


class PaperspaceAdapter(ProviderAdapter):
    """Paperspace Gradient adapter — API-based automation.

    Paperspace Gradient provides:
      - Free GPU notebooks (M4000, P5000)
      - Free CPU notebooks
      - Gradient Notebooks API for programmatic control

    For MVP, full job start/stop is manual. Credential validation,
    health checks, and capacity estimation work via the API.
    """

    provider_key: str = "paperspace"
    display_name: str = "Paperspace Gradient"
    provider_class: str = "A"
    supports_auto: bool = True
    supported_profiles: list[str] = [
        "cpu_only", "small_gpu", "medium_gpu",
    ]

    # ── Credential helpers ──────────────────────────────────────────

    def _get_api_key(self, account: dict) -> str:
        """Retrieve Paperspace API key from account or env."""
        creds = account.get("credentials", {}) or {}
        val = creds.get("paperspace_api_key", "")
        if val:
            return val
        return os.environ.get("PAPERSPACE_API_KEY", "")

    # ── API helpers ─────────────────────────────────────────────────

    def _api_request(self, method: str, path: str, api_key: str,
                     timeout: int = 15) -> Optional[dict]:
        """Make a Paperspace API request using curl via subprocess.

        Returns parsed JSON response or None on failure.
        """
        url = f"{PAPERSPACE_API_BASE}{path}"
        try:
            result = subprocess.run(
                [
                    "curl", "-s", "-X", method,
                    "-H", f"Authorization: Bearer {api_key}",
                    "-H", "Content-Type: application/json",
                    url,
                ],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
            return None
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            logger.debug(f"Paperspace API request failed: {e}")
            return None
        except Exception as e:
            logger.debug(f"Paperspace API error: {e}")
            return None

    # ── ProviderAdapter interface ───────────────────────────────────

    @with_retry(max_retries=2, base_delay=2.0)
    def validate_credentials(self, account: dict) -> ProviderResult:
        """Validate Paperspace API key by calling the whoami endpoint."""
        api_key = self._get_api_key(account)
        if not api_key:
            return ProviderResult(
                ok=False, status="error",
                message="Paperspace API key not configured. Set paperspace_api_key via /add or PAPERSPACE_API_KEY env var.",
            )

        response = self._api_request("GET", "/api/v1/user", api_key)
        if response and isinstance(response, dict):
            # Successful whoami response contains user info
            username = response.get("email", response.get("username", "unknown"))
            return ProviderResult(
                ok=True, status="ok",
                message=f"Paperspace API authenticated as {redact_text(username)}",
            )

        return ProviderResult(
            ok=False, status="error",
            message="Paperspace API authentication failed — invalid or expired API key",
        )

    @with_retry(max_retries=2, base_delay=2.0)
    def health_check(self, account: dict) -> ProviderResult:
        """Check if Paperspace API is accessible."""
        api_key = self._get_api_key(account)
        if not api_key:
            return ProviderResult(
                ok=False, status="down",
                message="No Paperspace API key configured",
            )

        response = self._api_request("GET", "/api/v1/user", api_key, timeout=10)
        if response:
            return ProviderResult(
                ok=True, status="ok",
                message="Paperspace API is accessible",
            )

        return ProviderResult(
            ok=False, status="down",
            message="Paperspace API unreachable or auth failed",
        )

    def estimate_capacity(self, account: dict) -> ProviderResult:
        """Estimate GPU capacity for Paperspace Gradient.

        Paperspace Gradient free tier:
          - Free GPU: M4000 (8 GB) or P5000 (16 GB) on Free-GPU machines
          - Free CPU: C5 machine type
          - Session limits apply (typically 6-12 hours for free tier)
          - Machine types: Free-GPU, Free-CPU
        """
        api_key = self._get_api_key(account)
        gpu_info = {
            "gpu_type": "M4000 / P5000 (free tier)",
            "gpu_memory_gb": 8,  # M4000 default
            "max_runtime_minutes": 360,  # 6 hours typical free tier limit
            "supports_checkpoint": True,
        }

        if api_key:
            # Try to get machine info from API
            response = self._api_request("GET", "/api/v1/machines", api_key, timeout=10)
            if response and isinstance(response, list):
                for machine in response:
                    if isinstance(machine, dict):
                        machine_type = machine.get("machineType", "")
                        state = machine.get("state", "")
                        if "free" in machine_type.lower():
                            # Determine GPU from machine type
                            if "p5000" in machine_type.lower():
                                gpu_info["gpu_type"] = "P5000 (free tier)"
                                gpu_info["gpu_memory_gb"] = 16
                            elif "m4000" in machine_type.lower():
                                gpu_info["gpu_type"] = "M4000 (free tier)"
                                gpu_info["gpu_memory_gb"] = 8
                            gpu_info["data"] = {"machine_type": machine_type, "state": state}
                            break

        return ProviderResult(
            ok=True, status="ok",
            message=f"Capacity: {gpu_info['gpu_type']}, {gpu_info['gpu_memory_gb']} GB VRAM",
            data=gpu_info,
        )

    def start_job(self, lease: dict, job: dict, account: dict) -> ProviderResult:
        """Start a training job on Paperspace Gradient.

        For MVP: returns manual_required. Full Gradient API job management
        (creating notebooks, mounting storage, injecting scripts) is complex
        and will be implemented in a future release.
        """
        api_key = self._get_api_key(account)
        if not api_key:
            return ProviderResult(
                ok=False, status="error",
                message="Paperspace API key not configured",
            )

        # MVP: manual start required
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="This provider requires manual runtime start in MVP.",
            data={
                "instructions": (
                    "1. Go to https://gradient.run/notebooks\n"
                    "2. Create a new notebook with Free-GPU machine\n"
                    "3. Upload your training script\n"
                    "4. Run the script in the notebook terminal\n"
                    "5. Use /confirm in TUI to start the countdown timer"
                ),
            },
        )

    def stop_job(self, lease: dict, account: dict) -> ProviderResult:
        """Stop a running job on Paperspace Gradient.

        For MVP: attempts API stop, falls back to manual instruction.
        """
        api_key = self._get_api_key(account)
        if not api_key:
            return ProviderResult(
                ok=False, status="error",
                message="Paperspace API key not configured",
            )

        # Try to find and stop running machines via API
        response = self._api_request("GET", "/api/v1/machines", api_key, timeout=10)
        if response and isinstance(response, list):
            for machine in response:
                if isinstance(machine, dict) and machine.get("state") == "running":
                    machine_id = machine.get("id", "")
                    if machine_id:
                        self._api_request(
                            "POST",
                            f"/api/v1/machines/{machine_id}/stop",
                            api_key, timeout=10,
                        )

        return ProviderResult(
            ok=True, status="stopped",
            message="Stop request sent. Verify in Gradient console.",
        )

    @with_retry(max_retries=2, base_delay=1.0)
    def fetch_logs(self, lease: dict, account: dict) -> ProviderResult:
        """Fetch recent logs from Paperspace Gradient.

        For MVP: returns a message directing to the Gradient console.
        Full log streaming via API will be added in a future release.
        """
        api_key = self._get_api_key(account)
        if not api_key:
            return ProviderResult(
                ok=False, status="error",
                message="Paperspace API key not configured",
            )

        # Try to get notebook logs via API
        notebook_id = lease.get("provider_data", {}).get("notebook_id", "")
        if notebook_id:
            response = self._api_request(
                "GET",
                f"/api/v1/notebooks/{notebook_id}/logs",
                api_key, timeout=10,
            )
            if response:
                logs = response.get("logs", "") if isinstance(response, dict) else str(response)
                return ProviderResult(
                    ok=True, status="ok",
                    message="Fetched logs from Paperspace",
                    data={"logs": redact_text(logs)},
                )

        return ProviderResult(
            ok=True, status="ok",
            message="Logs not available via API in MVP. Check console: https://gradient.run/notebooks",
            data={"logs": "Logs available in Gradient console UI"},
        )

    def sync_checkpoint(self, lease: dict, account: dict,
                        checkpoint_uri: str) -> ProviderResult:
        """Sync checkpoints to/from Paperspace Gradient.

        For MVP: returns manual instruction for checkpoint sync.
        Gradient supports /storage mounted volumes for persistence.
        """
        api_key = self._get_api_key(account)
        if not api_key:
            return ProviderResult(
                ok=False, status="error",
                message="Paperspace API key not configured",
            )

        # MVP: manual checkpoint sync
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message=(
                "Checkpoint sync is manual in MVP. "
                "Use /storage in your Gradient notebook to persist checkpoints."
            ),
            data={
                "instructions": (
                    "1. Save checkpoints to /storage in your notebook\n"
                    "2. Download from Gradient console\n"
                    "3. Upload to next session's /storage\n"
                    "Or use: gradient experiments with --checkpoint-uri"
                ),
            },
        )
