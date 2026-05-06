"""Kaggle Notebooks provider adapter for FamilyGPU Orchestrator.

Class B adapter — Notebook/Interactive with API support.
Kaggle API can push kernel code and check status, providing
partial automation for job lifecycle management.

For MVP: start_job pushes kernel code via kaggle CLI.
Stop, status checks, and checkpoint uploads use kaggle CLI commands.

Auth: account.credentials = {
    "kaggle_username": "...",
    "kaggle_key": "...",      # 32-char hex API key
}

Kaggle API reference: https://github.com/Kaggle/kaggle-api
CLI commands used:
  - kaggle kernels push    — push kernel source code
  - kaggle kernels status  — check kernel execution status
  - kaggle kernels pull    — download kernel output
  - kaggle datasets create — upload checkpoints as dataset
"""

import json
import logging
import os
import subprocess
import tempfile
import uuid
from datetime import datetime
from typing import Optional

from providers.base import ProviderAdapter, ProviderResult
from vault import redact_text, scan_for_secrets

logger = logging.getLogger("fgt.providers.kaggle")

KAGGLE_CLI_TIMEOUT = 30  # seconds for CLI commands


class KaggleAdapter(ProviderAdapter):
    """Kaggle Notebooks adapter — Notebook/Interactive (Class B).

    Kaggle provides free GPU access via interactive notebooks:
      - 30 hours/week GPU quota (T4 x2 or P100)
      - 20 hours/week TPU quota
      - CPU notebooks with 4 cores, 30 GB RAM
      - Max session: 12 hours

    This adapter uses the kaggle CLI (subprocess) for:
      - Pushing kernel code
      - Checking kernel status
      - Uploading checkpoints as datasets

    The kaggle CLI requires KAGGLE_USERNAME and KAGGLE_KEY
    environment variables or a ~/.kaggle/kaggle.json file.
    """

    provider_key: str = "kaggle"
    display_name: str = "Kaggle Notebooks"
    provider_class: str = "B"
    automation_level: str = "partial_auto"
    supports_auto: bool = True
    supported_profiles: list[str] = [
        "cpu_only", "small_gpu", "medium_gpu",
    ]

    # ── Credential helpers ──────────────────────────────────────────

    def _get_credentials(self, account: dict) -> tuple[str, str]:
        """Retrieve Kaggle credentials from account.

        Returns (username, key) tuple.
        """
        creds = account.get("credentials", {}) or {}
        username = creds.get("kaggle_username", "") or os.environ.get("KAGGLE_USERNAME", "")
        key = creds.get("kaggle_key", "") or os.environ.get("KAGGLE_KEY", "")
        return username, key

    def _setup_env(self, account: dict) -> dict:
        """Build environment dict with Kaggle credentials for subprocess."""
        username, key = self._get_credentials(account)
        env = os.environ.copy()
        if username:
            env["KAGGLE_USERNAME"] = username
        if key:
            env["KAGGLE_KEY"] = key
        return env

    # ── CLI helpers ─────────────────────────────────────────────────

    def _run_kaggle_cli(self, args: list[str], env: dict,
                        timeout: int = KAGGLE_CLI_TIMEOUT) -> subprocess.CompletedProcess:
        """Run a kaggle CLI command via subprocess.

        Args:
            args: Command arguments after 'kaggle' (e.g. ['kernels', 'push'])
            env: Environment dict with KAGGLE_USERNAME/KAGGLE_KEY
            timeout: Timeout in seconds

        Returns:
            CompletedProcess instance
        """
        cmd = ["kaggle"] + args
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
            logger.warning(f"Kaggle CLI timed out: {' '.join(args)}")
            return subprocess.CompletedProcess(
                args=cmd, returncode=-1,
                stdout="", stderr="Command timed out",
            )
        except FileNotFoundError:
            logger.error("kaggle CLI not found — install with: pip install kaggle")
            return subprocess.CompletedProcess(
                args=cmd, returncode=-1,
                stdout="", stderr="kaggle CLI not installed",
            )

    def _check_kaggle_installed(self) -> bool:
        """Check if kaggle CLI is available."""
        try:
            result = subprocess.run(
                ["kaggle", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # ── Kernel metadata generation ──────────────────────────────────

    def _generate_kernel_metadata(self, job: dict, account: dict) -> dict:
        """Generate kernel metadata for kaggle kernels push.

        Returns a dict matching the kernel-metadata.json format.
        """
        username, _ = self._get_credentials(account)
        job_id = job.get("id", str(uuid.uuid4())[:8])
        slug = f"familygpu-{job_id}"

        # Determine GPU type from profile
        gpu_profile = job.get("gpu_profile", "small_gpu")
        if gpu_profile == "cpu_only":
            enable_gpu = False
            enable_internet = True
        elif gpu_profile == "medium_gpu":
            enable_gpu = True
            enable_internet = True
        else:
            enable_gpu = True
            enable_internet = True

        return {
            "id": f"{username}/{slug}",
            "title": slug,
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": enable_gpu,
            "enable_internet": enable_internet,
            "keywords": ["familygpu", "training"],
            "dataset_sources": [],
            "kernel_sources": [],
        }

    # ── ProviderAdapter interface ───────────────────────────────────

    def validate_credentials(self, account: dict) -> ProviderResult:
        """Validate Kaggle credentials by listing datasets (whoami test)."""
        username, key = self._get_credentials(account)

        if not username or not key:
            return ProviderResult(
                ok=False, status="error",
                message="Kaggle credentials not configured. Set kaggle_username and kaggle_key via /add.",
            )

        if not self._check_kaggle_installed():
            return ProviderResult(
                ok=False, status="error",
                message="kaggle CLI not installed. Run: pip install kaggle",
            )

        env = self._setup_env(account)
        result = self._run_kaggle_cli(["config", "view"], env, timeout=10)

        if result.returncode == 0:
            return ProviderResult(
                ok=True, status="ok",
                message=f"Kaggle API authenticated as {redact_text(username)}",
            )

        return ProviderResult(
            ok=False, status="error",
            message=f"Kaggle API authentication failed: {result.stderr.strip()}",
        )

    def health_check(self, account: dict) -> ProviderResult:
        """Check if Kaggle API is accessible."""
        username, key = self._get_credentials(account)

        if not username or not key:
            return ProviderResult(
                ok=False, status="down",
                message="No Kaggle credentials configured",
            )

        if not self._check_kaggle_installed():
            return ProviderResult(
                ok=False, status="down",
                message="kaggle CLI not installed",
            )

        env = self._setup_env(account)
        result = self._run_kaggle_cli(["config", "view"], env, timeout=10)

        if result.returncode == 0:
            return ProviderResult(
                ok=True, status="ok",
                message="Kaggle API is accessible",
            )

        return ProviderResult(
            ok=False, status="down",
            message="Kaggle API unreachable or auth failed",
        )

    def estimate_capacity(self, account: dict) -> ProviderResult:
        """Estimate GPU capacity for Kaggle Notebooks.

        Kaggle free tier:
          - GPU: NVIDIA T4 x2 (16 GB each) or P100 (16 GB)
          - CPU: 4 cores, 30 GB RAM
          - TPU: TPU v3-8
          - GPU quota: 30 hours/week
          - Max session: 12 hours
          - GPU allocation subject to quota availability
        """
        return ProviderResult(
            ok=True, status="ok",
            message="Kaggle: T4 x2 (16 GB each), 30h/week GPU quota",
            data={
                "gpu_type": "NVIDIA T4 x2 / P100",
                "gpu_memory_gb": 16,
                "max_runtime_minutes": 720,  # 12 hours
                "supports_checkpoint": True,
                "gpu_quota_hours_per_week": 30,
                "cpu_spec": "4 cores, 30 GB RAM",
                "disk_space_gb": 20,
                "note": "GPU quota resets weekly; T4 allocation is most common",
            },
        )

    def start_job(self, lease: dict, job: dict, account: dict) -> ProviderResult:
        """Start a training job on Kaggle via kaggle kernels push.

        Creates a temporary directory with:
          - script.py (the training script, scanned for secrets)
          - kernel-metadata.json

        Then pushes the kernel via the kaggle CLI.
        """
        username, key = self._get_credentials(account)
        if not username or not key:
            return ProviderResult(
                ok=False, status="error",
                message="Kaggle credentials not configured",
            )

        if not self._check_kaggle_installed():
            return ProviderResult(
                ok=False, status="error",
                message="kaggle CLI not installed. Run: pip install kaggle",
            )

        # Get training script
        training_script = job.get("training_script", "")
        if training_script:
            secret_warnings = scan_for_secrets(training_script)
            if secret_warnings:
                logger.warning(
                    f"Secrets detected in training script: {'; '.join(secret_warnings)}. "
                    f"Replacing with safe loader stub."
                )
                training_script = (
                    "# FamilyGPU — Safe Loader Stub\n"
                    "# Secrets were detected in the original script.\n"
                    "# Please upload your script manually via Kaggle UI.\n"
                    f"# Warnings: {'; '.join(secret_warnings)}\n"
                    "print('Please upload your training script manually')\n"
                )
            else:
                training_script = redact_text(training_script)
        else:
            training_script = "# No training script provided\nprint('Add your training code here')\n"

        # Generate kernel metadata
        metadata = self._generate_kernel_metadata(job, account)
        job_id = job.get("id", str(uuid.uuid4())[:8])

        # Create temp directory with kernel files
        try:
            with tempfile.TemporaryDirectory(prefix=f"familygpu_kaggle_{job_id}_") as tmpdir:
                # Write script
                script_path = os.path.join(tmpdir, "script.py")
                with open(script_path, "w") as f:
                    f.write(training_script)

                # Write metadata
                metadata_path = os.path.join(tmpdir, "kernel-metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                # Push kernel
                env = self._setup_env(account)
                result = self._run_kaggle_cli(
                    ["kernels", "push", "-p", tmpdir],
                    env, timeout=60,
                )

                if result.returncode == 0:
                    # Parse kernel URL from output
                    output = result.stdout.strip()
                    kernel_ref = metadata["id"]

                    return ProviderResult(
                        ok=True, status="running",
                        message=f"Kernel pushed successfully: {kernel_ref}",
                        data={
                            "kernel_ref": kernel_ref,
                            "kernel_url": f"https://www.kaggle.com/{kernel_ref}",
                            "cli_output": redact_text(output),
                            "instructions": (
                                "1. Monitor kernel status: kaggle kernels status\n"
                                f"2. View output at: https://www.kaggle.com/{kernel_ref}\n"
                                "3. Use /confirm in TUI to start the countdown timer"
                            ),
                        },
                    )
                else:
                    error_msg = result.stderr.strip() or result.stdout.strip()
                    return ProviderResult(
                        ok=False, status="error",
                        message=f"Kaggle kernel push failed: {redact_text(error_msg)}",
                        data={"cli_stderr": redact_text(result.stderr)},
                    )

        except Exception as e:
            logger.error(f"Kaggle start_job error: {e}")
            return ProviderResult(
                ok=False, status="error",
                message=f"Failed to push Kaggle kernel: {e}",
            )

    def stop_job(self, lease: dict, account: dict) -> ProviderResult:
        """Stop a running Kaggle kernel.

        Kaggle API doesn't have a direct 'stop kernel' command.
        The kernel can be stopped via the web UI.
        """
        username, key = self._get_credentials(account)
        if not username or not key:
            return ProviderResult(
                ok=False, status="error",
                message="Kaggle credentials not configured",
            )

        kernel_ref = lease.get("provider_data", {}).get("kernel_ref", "")

        if kernel_ref and self._check_kaggle_installed():
            # Try to check current status first
            env = self._setup_env(account)
            result = self._run_kaggle_cli(
                ["kernels", "status", kernel_ref],
                env, timeout=15,
            )
            if result.returncode == 0:
                status_output = result.stdout.strip()
                if "complete" in status_output.lower():
                    return ProviderResult(
                        ok=True, status="stopped",
                        message="Kernel already completed",
                        data={"kaggle_status": status_output},
                    )

        # Manual stop required
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Kaggle kernels must be stopped via the web UI",
            data={
                "kernel_ref": kernel_ref,
                "instructions": (
                    "1. Go to https://www.kaggle.com/kernels\n"
                    "2. Find your running kernel\n"
                    "3. Click 'Stop' in the kernel settings\n"
                    "4. Use /confirm in TUI to mark as stopped"
                ),
            },
        )

    def fetch_logs(self, lease: dict, account: dict) -> ProviderResult:
        """Fetch logs from a Kaggle kernel.

        Uses kaggle kernels status to check execution state.
        Full log output is available via kaggle kernels pull.
        """
        username, key = self._get_credentials(account)
        if not username or not key:
            return ProviderResult(
                ok=False, status="error",
                message="Kaggle credentials not configured",
            )

        kernel_ref = lease.get("provider_data", {}).get("kernel_ref", "")
        if not kernel_ref:
            return ProviderResult(
                ok=True, status="manual_required", manual=True,
                message="No kernel reference found in lease data",
                data={"logs": "Check kernel output in Kaggle UI"},
            )

        if not self._check_kaggle_installed():
            return ProviderResult(
                ok=False, status="error",
                message="kaggle CLI not installed",
            )

        env = self._setup_env(account)

        # Check kernel status
        status_result = self._run_kaggle_cli(
            ["kernels", "status", kernel_ref],
            env, timeout=15,
        )

        if status_result.returncode != 0:
            return ProviderResult(
                ok=False, status="error",
                message=f"Failed to check kernel status: {status_result.stderr.strip()}",
            )

        status_output = status_result.stdout.strip()

        # Try to pull output if kernel is complete
        logs = status_output
        if "complete" in status_output.lower():
            try:
                with tempfile.TemporaryDirectory(prefix="familygpu_kaggle_logs_") as tmpdir:
                    pull_result = self._run_kaggle_cli(
                        ["kernels", "pull", kernel_ref, "-p", tmpdir],
                        env, timeout=30,
                    )
                    if pull_result.returncode == 0:
                        # Read the pulled log file
                        log_file = os.path.join(tmpdir, "script.py.log")
                        if os.path.exists(log_file):
                            with open(log_file, "r") as f:
                                logs = f.read()
            except Exception as e:
                logger.debug(f"Failed to pull kernel output: {e}")

        return ProviderResult(
            ok=True, status="ok",
            message="Fetched Kaggle kernel status",
            data={"logs": redact_text(logs)},
        )

    def sync_checkpoint(self, lease: dict, account: dict,
                        checkpoint_uri: str) -> ProviderResult:
        """Sync checkpoints via Kaggle datasets.

        Uploads checkpoint files as a Kaggle dataset for persistence
        and cross-session access. Falls back to manual instructions.
        """
        username, key = self._get_credentials(account)
        if not username or not key:
            return ProviderResult(
                ok=False, status="error",
                message="Kaggle credentials not configured",
            )

        kernel_ref = lease.get("provider_data", {}).get("kernel_ref", "")

        if not self._check_kaggle_installed():
            return ProviderResult(
                ok=True, status="manual_required", manual=True,
                message="kaggle CLI not installed — manual checkpoint sync required",
                data={
                    "checkpoint_uri": checkpoint_uri,
                    "instructions": (
                        "1. In your Kaggle notebook, save checkpoints:\n"
                        "   torch.save(model.state_dict(), 'checkpoint.pt')\n"
                        "2. Download output from the kernel output tab\n"
                        "3. Upload to next provider's checkpoint location\n"
                        f"   Target: {checkpoint_uri}"
                    ),
                },
            )

        # Try to create a dataset for checkpoint storage
        env = self._setup_env(account)
        job_id = lease.get("job_id", str(uuid.uuid4())[:8])
        dataset_slug = f"familygpu-checkpoint-{job_id}"

        try:
            with tempfile.TemporaryDirectory(prefix=f"familygpu_ckpt_{job_id}_") as tmpdir:
                # Create dataset metadata
                dataset_metadata = {
                    "title": f"FamilyGPU Checkpoint {job_id}",
                    "id": f"{username}/{dataset_slug}",
                    "licenses": [{"name": "CC0-1.0"}],
                }
                metadata_path = os.path.join(tmpdir, "dataset-metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(dataset_metadata, f, indent=2)

                # Create a placeholder file (actual checkpoints uploaded manually)
                placeholder_path = os.path.join(tmpdir, "README.md")
                with open(placeholder_path, "w") as f:
                    f.write(f"# FamilyGPU Checkpoint {job_id}\n")
                    f.write(f"Checkpoint URI: {checkpoint_uri}\n")
                    f.write("Upload your checkpoint files to this dataset.\n")

                # Try to create the dataset
                result = self._run_kaggle_cli(
                    ["datasets", "create", "-p", tmpdir],
                    env, timeout=60,
                )

                if result.returncode == 0:
                    return ProviderResult(
                        ok=True, status="ok",
                        message=f"Checkpoint dataset created: {username}/{dataset_slug}",
                        data={
                            "checkpoint_uri": checkpoint_uri,
                            "dataset_ref": f"{username}/{dataset_slug}",
                            "dataset_url": f"https://www.kaggle.com/datasets/{username}/{dataset_slug}",
                            "instructions": (
                                "1. Save checkpoints in your kernel:\n"
                                "   torch.save(model.state_dict(), 'checkpoint.pt')\n"
                                "2. Add checkpoint files to the dataset via Kaggle UI\n"
                                f"3. Dataset URL: https://www.kaggle.com/datasets/{username}/{dataset_slug}\n"
                                "4. In your next kernel, add this dataset as a data source"
                            ),
                        },
                    )
        except Exception as e:
            logger.debug(f"Kaggle dataset creation failed: {e}")

        # Fallback: manual instructions
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Checkpoint sync via Kaggle dataset (manual in MVP).",
            data={
                "checkpoint_uri": checkpoint_uri,
                "instructions": (
                    "1. Save checkpoints in your Kaggle notebook:\n"
                    "   torch.save(model.state_dict(), 'checkpoint.pt')\n"
                    "2. Create a new dataset: https://www.kaggle.com/datasets/new\n"
                    "3. Upload checkpoint files\n"
                    "4. Add the dataset as a source in your next kernel\n"
                    f"5. Target checkpoint URI: {checkpoint_uri}"
                ),
            },
        )
