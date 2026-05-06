"""Oracle Cloud Free Tier provider adapter for FamilyGPU Orchestrator.

Class A adapter — fully automated via SSH. Supports always-free Ampere A1
compute instances with GPU profiles up to high_vram_gpu.

Auth: account.credentials = {
    "oci_vm_host": "129.x.x.x",
    "oci_vm_user": "opc",
    "oci_ssh_key": "~/.ssh/id_rsa",
}
Or set OCI_VM_HOST, OCI_VM_USER, OCI_SSH_KEY env vars.
"""

import os
import shlex
import logging
import subprocess
from pathlib import Path
from typing import Optional

from providers.base import ProviderAdapter, ProviderResult
from vault import redact_text, scan_for_secrets
from handlers import _safe_ssh_args, validate_hostname, validate_ssh_user, with_retry

logger = logging.getLogger("fgt.providers.oracle_cloud")


class OracleCloudAdapter(ProviderAdapter):
    """Oracle Cloud Free Tier adapter — SSH-based automation.

    Uses SSH + SCP to push scripts, start jobs via nohup, and manage
    process lifecycle through PID files on the remote host.
    """

    provider_key: str = "oracle_cloud"
    display_name: str = "Oracle Cloud Free Tier"
    provider_class: str = "A"
    supports_auto: bool = True
    supported_profiles: list[str] = [
        "cpu_only", "small_gpu", "medium_gpu", "high_vram_gpu", "long_running",
    ]

    # ── Credential helpers ──────────────────────────────────────────

    def _get_credential(self, account: dict, key: str, env_var: str = "",
                        default: str = "") -> str:
        """Retrieve a credential from account dict, then env, then default."""
        creds = account.get("credentials", {}) or {}
        val = creds.get(key, "")
        if val:
            return val
        if env_var:
            val = os.environ.get(env_var, "")
            if val:
                return val
        return default

    def _ssh_config(self, account: dict) -> tuple:
        """Return (host, user, key_file) from account credentials."""
        host = self._get_credential(account, "oci_vm_host", "OCI_VM_HOST")
        user = self._get_credential(account, "oci_vm_user", "OCI_VM_USER", "opc")
        key_file = self._get_credential(account, "oci_ssh_key", "OCI_SSH_KEY", "~/.ssh/id_rsa")
        return host, user, key_file

    def _validated_ssh(self, account: dict) -> Optional[tuple]:
        """Validate SSH args and return (host, user, expanded_key) or None."""
        host, user, key_file = self._ssh_config(account)
        if not host:
            return None
        safe = _safe_ssh_args(host, user, key_file)
        return safe  # None if invalid, (host, user, expanded_key) if valid

    # ── ProviderAdapter interface ───────────────────────────────────

    @with_retry(max_retries=2, base_delay=2.0)
    def validate_credentials(self, account: dict) -> ProviderResult:
        """Validate Oracle Cloud SSH credentials by running a whoami check."""
        host, user, key_file = self._ssh_config(account)
        if not host:
            return ProviderResult(
                ok=False, status="error",
                message="OCI VM host not configured. Set oci_vm_host via /add or OCI_VM_HOST env var.",
            )

        safe = _safe_ssh_args(host, user, key_file)
        if not safe:
            return ProviderResult(
                ok=False, status="error",
                message=f"Invalid SSH config. Host: {redact_text(host)!r}, User: {user!r}",
            )

        safe_host, safe_user, expanded_key = safe
        try:
            result = subprocess.run(
                ["ssh", "-i", expanded_key, "-o", "ConnectTimeout=10",
                 "-o", "StrictHostKeyChecking=accept-new",
                 f"{safe_user}@{safe_host}", "echo ok"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and "ok" in result.stdout:
                return ProviderResult(
                    ok=True, status="ok",
                    message=f"SSH connection to {safe_host} successful",
                )
            stderr = redact_text(result.stderr.strip())
            return ProviderResult(
                ok=False, status="error",
                message=f"SSH connection failed: {stderr}",
            )
        except subprocess.TimeoutExpired:
            return ProviderResult(
                ok=False, status="error",
                message=f"SSH connection to {safe_host} timed out",
            )
        except Exception as e:
            return ProviderResult(
                ok=False, status="error",
                message=f"SSH validation error: {e}",
            )

    @with_retry(max_retries=2, base_delay=2.0)
    def health_check(self, account: dict) -> ProviderResult:
        """Check if the Oracle Cloud VM is reachable and responsive."""
        safe = self._validated_ssh(account)
        if not safe:
            return ProviderResult(
                ok=False, status="down",
                message="No valid SSH config for Oracle Cloud",
            )

        safe_host, safe_user, expanded_key = safe
        try:
            result = subprocess.run(
                ["ssh", "-i", expanded_key, "-o", "ConnectTimeout=10",
                 f"{safe_user}@{safe_host}", "uptime"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                uptime_info = result.stdout.strip()
                return ProviderResult(
                    ok=True, status="ok",
                    message=f"VM is up: {uptime_info}",
                    data={"uptime": uptime_info},
                )
            return ProviderResult(
                ok=False, status="down",
                message="VM unreachable via SSH",
            )
        except subprocess.TimeoutExpired:
            return ProviderResult(
                ok=False, status="down",
                message=f"VM {safe_host} timed out",
            )
        except Exception as e:
            return ProviderResult(
                ok=False, status="error",
                message=f"Health check error: {e}",
            )

    def estimate_capacity(self, account: dict) -> ProviderResult:
        """Estimate GPU capacity for Oracle Cloud Free Tier.

        Oracle Cloud Always Free tier includes:
          - Up to 4 Ampere A1 cores (shared, arm64)
          - 24 GB RAM
          - No always-free GPU by default, but A1 shapes support small GPU workloads
          - GPU instances available via free-tier promotional credits
        """
        safe = self._validated_ssh(account)
        gpu_info = {
            "gpu_type": "Ampere A1 (arm64) / A10 GPU",
            "gpu_memory_gb": 0,
            "max_runtime_minutes": 0,  # No hard limit on always-free
            "supports_checkpoint": True,
        }

        if safe:
            safe_host, safe_user, expanded_key = safe
            try:
                result = subprocess.run(
                    ["ssh", "-i", expanded_key, "-o", "ConnectTimeout=5",
                     f"{safe_user}@{safe_host}",
                     "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no-gpu'"],
                    capture_output=True, text=True, timeout=15,
                )
                output = result.stdout.strip()
                if output and output != "no-gpu" and "no-gpu" not in output.lower():
                    # Parse GPU info: e.g. "NVIDIA A10, 24576 MiB"
                    parts = output.split(",")
                    if len(parts) >= 2:
                        gpu_info["gpu_type"] = parts[0].strip()
                        mem_str = parts[1].strip()
                        try:
                            mem_mib = int(mem_str.split()[0])
                            gpu_info["gpu_memory_gb"] = mem_mib // 1024
                        except (ValueError, IndexError):
                            pass
            except Exception as e:
                logger.debug(f"GPU detection failed for Oracle Cloud: {e}")

        # Always-free: no time limit; GPU shapes: promotional limits vary
        if gpu_info["gpu_memory_gb"] == 0:
            gpu_info["gpu_type"] = "Ampere A1 (CPU-only, arm64)"

        gpu_info["max_runtime_minutes"] = 0  # No hard limit

        return ProviderResult(
            ok=True, status="ok",
            message=f"Capacity: {gpu_info['gpu_type']}, {gpu_info['gpu_memory_gb']} GB VRAM",
            data=gpu_info,
        )

    @with_retry(max_retries=2, base_delay=2.0)
    def start_job(self, lease: dict, job: dict, account: dict) -> ProviderResult:
        """Start a training job on Oracle Cloud via SSH.

        SCPs the training script to the remote host, then launches it
        with nohup, capturing output to ~/training.log and PID to ~/training.pid.
        """
        safe = self._validated_ssh(account)
        if not safe:
            return ProviderResult(
                ok=False, status="error",
                message="No valid SSH config for Oracle Cloud",
            )

        safe_host, safe_user, expanded_key = safe

        # Determine script path from job dict
        script_path = job.get("script_path", "train.py")
        script = Path(script_path)
        if not script.exists():
            return ProviderResult(
                ok=False, status="error",
                message=f"Script not found: {script_path}",
            )

        # Scan for secrets before pushing
        try:
            script_content = script.read_text()
            secret_warnings = scan_for_secrets(script_content)
            if secret_warnings:
                logger.warning(
                    "Secrets detected in %s: %s", script_path,
                    "; ".join(secret_warnings),
                )
        except Exception:
            pass  # Non-blocking

        try:
            # SCP script to remote — preserve filename
            remote_script_name = script.name
            scp_result = subprocess.run(
                ["scp", "-i", expanded_key, "-o", "StrictHostKeyChecking=accept-new",
                 str(script), f"{safe_user}@{safe_host}:~/{remote_script_name}"],
                capture_output=True, text=True, timeout=30,
            )
            if scp_result.returncode != 0:
                return ProviderResult(
                    ok=False, status="error",
                    message=f"SCP failed: {redact_text(scp_result.stderr)}",
                )

            # SCP checkpoints if available (for resume support)
            checkpoint_dir = job.get("checkpoint_dir", "./checkpoints")
            ckpt_path = Path(checkpoint_dir)
            if ckpt_path.exists() and any(ckpt_path.iterdir()):
                subprocess.run(
                    ["scp", "-i", expanded_key, "-r",
                     str(ckpt_path), f"{safe_user}@{safe_host}:~/checkpoints"],
                    capture_output=True, text=True, timeout=30,
                )

            # Start job via nohup with PID tracking
            quoted_name = shlex.quote(remote_script_name)
            ssh_result = subprocess.run(
                ["ssh", "-i", expanded_key,
                 f"{safe_user}@{safe_host}",
                 f"nohup python ~/{quoted_name} > ~/training.log 2>&1 & echo $! > ~/training.pid"],
                capture_output=True, text=True, timeout=30,
            )

            if ssh_result.returncode == 0:
                return ProviderResult(
                    ok=True, status="running",
                    message=f"Training started on {safe_host}",
                    data={"host": safe_host, "script": remote_script_name},
                )
            return ProviderResult(
                ok=False, status="error",
                message=f"SSH start failed: {redact_text(ssh_result.stderr)}",
            )

        except subprocess.TimeoutExpired:
            return ProviderResult(
                ok=False, status="error",
                message=f"SSH/SCP to {safe_host} timed out",
            )
        except Exception as e:
            return ProviderResult(
                ok=False, status="error",
                message=f"Start job error: {e}",
            )

    @with_retry(max_retries=2, base_delay=1.0)
    def stop_job(self, lease: dict, account: dict) -> ProviderResult:
        """Stop the running training job via PID file on Oracle Cloud."""
        safe = self._validated_ssh(account)
        if not safe:
            return ProviderResult(
                ok=False, status="error",
                message="No valid SSH config for Oracle Cloud",
            )

        safe_host, safe_user, expanded_key = safe
        try:
            subprocess.run(
                ["ssh", "-i", expanded_key,
                 f"{safe_user}@{safe_host}",
                 "kill $(cat ~/training.pid 2>/dev/null) 2>/dev/null; rm -f ~/training.pid"],
                capture_output=True, text=True, timeout=10,
            )
            return ProviderResult(
                ok=True, status="stopped",
                message="Training process killed via PID file",
            )
        except Exception as e:
            return ProviderResult(
                ok=False, status="error",
                message=f"Stop job error: {e}",
            )

    @with_retry(max_retries=2, base_delay=1.0)
    def fetch_logs(self, lease: dict, account: dict) -> ProviderResult:
        """Fetch the last 100 lines of training.log from Oracle Cloud."""
        safe = self._validated_ssh(account)
        if not safe:
            return ProviderResult(
                ok=False, status="error",
                message="No valid SSH config for Oracle Cloud",
            )

        safe_host, safe_user, expanded_key = safe
        try:
            result = subprocess.run(
                ["ssh", "-i", expanded_key,
                 f"{safe_user}@{safe_host}",
                 "cat ~/training.log | tail -100"],
                capture_output=True, text=True, timeout=15,
            )
            logs = redact_text(result.stdout) if result.stdout else ""
            return ProviderResult(
                ok=True, status="ok",
                message=f"Fetched {len(logs.splitlines())} log lines",
                data={"logs": logs},
            )
        except Exception as e:
            return ProviderResult(
                ok=False, status="error",
                message=f"Fetch logs error: {e}",
            )

    @with_retry(max_retries=2, base_delay=2.0)
    def sync_checkpoint(self, lease: dict, account: dict,
                        checkpoint_uri: str) -> ProviderResult:
        """Sync checkpoints to/from Oracle Cloud via SCP.

        The checkpoint_uri determines direction:
          - "pull:<local_path>" — SCP remote ~/checkpoints to local path
          - "push:<local_path>" — SCP local path to remote ~/checkpoints
          - Otherwise — default: push local ./checkpoints to remote
        """
        safe = self._validated_ssh(account)
        if not safe:
            return ProviderResult(
                ok=False, status="error",
                message="No valid SSH config for Oracle Cloud",
            )

        safe_host, safe_user, expanded_key = safe

        # Parse direction from checkpoint_uri
        direction = "push"
        local_path = "./checkpoints"
        if checkpoint_uri.startswith("pull:"):
            direction = "pull"
            local_path = checkpoint_uri[5:] or "./checkpoints"
        elif checkpoint_uri.startswith("push:"):
            direction = "push"
            local_path = checkpoint_uri[5:] or "./checkpoints"

        try:
            if direction == "push":
                local = Path(local_path)
                if not local.exists():
                    return ProviderResult(
                        ok=True, status="ok",
                        message="No local checkpoints to push",
                    )
                result = subprocess.run(
                    ["scp", "-i", expanded_key, "-r",
                     str(local), f"{safe_user}@{safe_host}:~/checkpoints"],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode == 0:
                    return ProviderResult(
                        ok=True, status="ok",
                        message=f"Checkpoints pushed to {safe_host}:~/checkpoints",
                    )
                return ProviderResult(
                    ok=False, status="error",
                    message=f"Checkpoint push failed: {redact_text(result.stderr)}",
                )
            else:  # pull
                local = Path(local_path)
                local.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(
                    ["scp", "-i", expanded_key, "-r",
                     f"{safe_user}@{safe_host}:~/checkpoints/", str(local)],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode == 0:
                    return ProviderResult(
                        ok=True, status="ok",
                        message=f"Checkpoints pulled from {safe_host}:~/checkpoints",
                    )
                return ProviderResult(
                    ok=False, status="error",
                    message=f"Checkpoint pull failed: {redact_text(result.stderr)}",
                )
        except Exception as e:
            return ProviderResult(
                ok=False, status="error",
                message=f"Checkpoint sync error: {e}",
            )
