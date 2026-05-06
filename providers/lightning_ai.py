"""Lightning AI provider adapter for FamilyGPU Orchestrator.

Class A adapter — API-based automation. Uses the Lightning AI API
to manage studios and compute instances.

For MVP: start_job returns manual_required (full API studio management is complex).
Credential validation and health check work via the whoami endpoint.

Auth: account.credentials = {
    "lightning_token": "...",
}
Or set LIGHTNING_TOKEN env var.
"""

import os
import json
import logging
import subprocess
from typing import Optional

from providers.base import ProviderAdapter, ProviderResult
from vault import redact_text, scan_for_secrets
from handlers import with_retry

logger = logging.getLogger("fgt.providers.lightning_ai")

LIGHTNING_API_BASE = "https://lightning.ai/api/v1"


class LightningAIAdapter(ProviderAdapter):
    """Lightning AI adapter — API-based automation.

    Lightning AI provides:
      - Free GPU credits (15 GPU hours/month on free tier)
      - Studios (cloud IDE with GPU access)
      - CPU and GPU machine types

    For MVP, full job start/stop is manual. Credential validation,
    health checks, and capacity estimation work via the API.
    """

    provider_key: str = "lightning_ai"
    display_name: str = "Lightning AI"
    provider_class: str = "A"
    automation_level: str = "partial_auto"
    supports_auto: bool = True
    supported_profiles: list[str] = [
        "cpu_only", "small_gpu", "medium_gpu",
    ]

    # ── Credential helpers ──────────────────────────────────────────

    def _get_token(self, account: dict) -> str:
        """Retrieve Lightning AI token from account or env."""
        creds = account.get("credentials", {}) or {}
        val = creds.get("lightning_token", "")
        if val:
            return val
        return os.environ.get("LIGHTNING_TOKEN", "")

    # ── API helpers ─────────────────────────────────────────────────

    def _api_request(self, method: str, path: str, token: str,
                     timeout: int = 15) -> Optional[dict]:
        """Make a Lightning AI API request using curl via subprocess.

        Returns parsed JSON response or None on failure.
        """
        url = f"{LIGHTNING_API_BASE}{path}"
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
            logger.debug(f"Lightning AI API request failed: {e}")
            return None
        except Exception as e:
            logger.debug(f"Lightning AI API error: {e}")
            return None

    # ── ProviderAdapter interface ───────────────────────────────────

    @with_retry(max_retries=2, base_delay=2.0)
    def validate_credentials(self, account: dict) -> ProviderResult:
        """Validate Lightning AI token by calling the whoami/user endpoint."""
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="Lightning AI token not configured. Set lightning_token via /add or LIGHTNING_TOKEN env var.",
            )

        response = self._api_request("GET", "/user", token)
        if response and isinstance(response, dict):
            username = response.get("username", response.get("email", "unknown"))
            return ProviderResult(
                ok=True, status="ok",
                message=f"Lightning AI authenticated as {redact_text(username)}",
            )

        return ProviderResult(
            ok=False, status="error",
            message="Lightning AI authentication failed — invalid or expired token",
        )

    @with_retry(max_retries=2, base_delay=2.0)
    def health_check(self, account: dict) -> ProviderResult:
        """Check if Lightning AI API is accessible."""
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="down",
                message="No Lightning AI token configured",
            )

        response = self._api_request("GET", "/user", token, timeout=10)
        if response:
            return ProviderResult(
                ok=True, status="ok",
                message="Lightning AI API is accessible",
            )

        return ProviderResult(
            ok=False, status="down",
            message="Lightning AI API unreachable or auth failed",
        )

    def estimate_capacity(self, account: dict) -> ProviderResult:
        """Estimate GPU capacity for Lightning AI.

        Lightning AI free tier:
          - 15 GPU hours/month
          - CPU: unlimited on CPU studios
          - GPU: T4 (16 GB) available on free tier
          - Studios: persistent cloud IDE environments
        """
        token = self._get_token(account)
        gpu_info = {
            "gpu_type": "T4 (free tier)",
            "gpu_memory_gb": 16,
            "max_runtime_minutes": 0,  # Limited by monthly quota, not per-session
            "supports_checkpoint": True,
        }

        if token:
            # Try to get credit/quota info from API
            response = self._api_request("GET", "/user/credits", token, timeout=10)
            if response and isinstance(response, dict):
                credits = response.get("gpu_credits_remaining", response.get("credits", "unknown"))
                gpu_info["credits_remaining"] = credits

        return ProviderResult(
            ok=True, status="ok",
            message=f"Capacity: {gpu_info['gpu_type']}, {gpu_info['gpu_memory_gb']} GB VRAM",
            data=gpu_info,
        )

    def start_job(self, lease: dict, job: dict, account: dict) -> ProviderResult:
        """Start a training job on Lightning AI.

        For MVP: returns manual_required. Full Lightning AI studio management
        (creating studios, configuring machines, injecting scripts) is complex
        and will be implemented in a future release.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="Lightning AI token not configured",
            )

        # MVP: manual start required
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="This provider requires manual runtime start in MVP.",
            data={
                "instructions": (
                    "1. Go to https://lightning.ai\n"
                    "2. Create a new Studio with GPU machine\n"
                    "3. Upload or paste your training script\n"
                    "4. Run the script in the Studio terminal\n"
                    "5. Use /confirm in TUI to start the countdown timer"
                ),
            },
        )

    def stop_job(self, lease: dict, account: dict) -> ProviderResult:
        """Stop a running job on Lightning AI.

        For MVP: attempts API stop, falls back to manual instruction.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="Lightning AI token not configured",
            )

        # Try to stop running studios via API
        studio_id = lease.get("provider_data", {}).get("studio_id", "")
        if studio_id:
            self._api_request(
                "POST",
                f"/studios/{studio_id}/stop",
                token, timeout=10,
            )

        return ProviderResult(
            ok=True, status="stopped",
            message="Stop request sent. Verify in Lightning AI console.",
        )

    @with_retry(max_retries=2, base_delay=1.0)
    def fetch_logs(self, lease: dict, account: dict) -> ProviderResult:
        """Fetch recent logs from Lightning AI.

        For MVP: returns a message directing to the Lightning AI console.
        Full log streaming via API will be added in a future release.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="Lightning AI token not configured",
            )

        # Try to get studio logs via API
        studio_id = lease.get("provider_data", {}).get("studio_id", "")
        if studio_id:
            response = self._api_request(
                "GET",
                f"/studios/{studio_id}/logs",
                token, timeout=10,
            )
            if response:
                logs = response.get("logs", "") if isinstance(response, dict) else str(response)
                return ProviderResult(
                    ok=True, status="ok",
                    message="Fetched logs from Lightning AI",
                    data={"logs": redact_text(logs)},
                )

        return ProviderResult(
            ok=True, status="ok",
            message="Logs not available via API in MVP. Check console: https://lightning.ai",
            data={"logs": "Logs available in Lightning AI Studio UI"},
        )

    def sync_checkpoint(self, lease: dict, account: dict,
                        checkpoint_uri: str) -> ProviderResult:
        """Sync checkpoints to/from Lightning AI.

        For MVP: returns manual instruction for checkpoint sync.
        Lightning AI Studios have persistent /teamspace storage.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="Lightning AI token not configured",
            )

        # MVP: manual checkpoint sync
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message=(
                "Checkpoint sync is manual in MVP. "
                "Use /teamspace/snapshots in your Studio for persistence."
            ),
            data={
                "instructions": (
                    "1. Save checkpoints to /teamspace in your Studio\n"
                    "2. Studios persist data across restarts automatically\n"
                    "3. Use Lightning Cloud Storage for cross-studio sharing"
                ),
            },
        )
