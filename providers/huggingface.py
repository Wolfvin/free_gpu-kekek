"""HuggingFace Spaces provider adapter for FamilyGPU Orchestrator.

Class C adapter — Special/Limited.

WARNING: HuggingFace Spaces (ZeroGPU) is designed for inference and
demos, NOT for long-running training jobs. This adapter exists to
support lightweight training experiments and demo deployments only.

Do NOT use this provider for:
  - Long-running training jobs (>1 hour)
  - Full fine-tuning of large models
  - Jobs requiring persistent GPU allocation
  - Any production training workload

Acceptable use cases:
  - Lightweight fine-tuning experiments (<30 min)
  - Model inference / demo deployment
  - Quick test runs to verify training scripts
  - ZeroGPU short-burst compute tasks

Auth: account.credentials = {
    "hf_token": "...",  # HuggingFace API token (write access)
}

HuggingFace Spaces uses the huggingface_hub library for API access.
ZeroGPU provides short GPU bursts for Gradio apps.
"""

import json
import logging
import os
import subprocess
import tempfile
import uuid
from typing import Optional

from providers.base import ProviderAdapter, ProviderResult
from vault import redact_text, scan_for_secrets

logger = logging.getLogger("fgt.providers.huggingface")

HF_CLI_TIMEOUT = 30

# ── Warning banner included in all user-facing messages ─────────────

_TRAINING_WARNING = (
    "⚠ WARNING: HuggingFace Spaces (ZeroGPU) is for inference/demos, "
    "NOT for long-running training! Use only for short experiments (<30 min)."
)


def _generate_gradio_wrapper(job: dict, lease: dict) -> str:
    """Generate a Gradio app wrapper for deploying on HF Spaces.

    Scans the training script for secrets before embedding.
    If secrets are detected, the script is replaced with a safe loader.

    Args:
        job: Job dict containing training_script and metadata
        lease: Lease dict for context

    Returns:
        Python source code for the Gradio app
    """
    job_id = job.get("id", str(uuid.uuid4())[:8])
    gpu_profile = job.get("gpu_profile", "small_gpu")
    training_script = job.get("training_script", "")

    # Determine if we can safely embed the training script
    script_section = ""
    if training_script:
        secret_warnings = scan_for_secrets(training_script)
        if secret_warnings:
            # SECRETS DETECTED — do NOT embed the raw script
            logger.warning(
                f"Secrets detected in training script for HF Spaces job {job_id}: "
                f"{'; '.join(secret_warnings)}. Using safe loader stub."
            )
            script_section = f'''
    # ⚠ Secrets detected in original script — not embedding.
    # Warnings: {'; '.join(secret_warnings)}
    # Please upload your script manually via the HF Spaces repo.
    def run_training():
        return "Please upload your training script manually via the HF Spaces repository. See README for instructions."
'''
        else:
            # Safe to embed — redact as final precaution
            safe_script = redact_text(training_script)
            # Indent the script for embedding inside a function
            indented = "\n".join("    " + line for line in safe_script.splitlines())
            script_section = f'''
    # Training script (auto-embedded by FamilyGPU)
    def run_training():
{indented}
        return "Training completed (check logs for details)"
'''
    else:
        script_section = '''
    def run_training():
        return "No training script provided. Add your code to this Space."
'''

    # Build the Gradio app
    gradio_app = f'''"""
FamilyGPU Training Space — Auto-generated Gradio wrapper.
Job ID: {job_id}
GPU Profile: {gpu_profile}

{_TRAINING_WARNING}
"""

import gradio as gr
import os

{script_section.strip()}

# ── Gradio Interface ──────────────────────────────────────────────

def start_training():
    """Start the training process."""
    try:
        result = run_training()
        return result
    except Exception as e:
        return f"Training error: {{e}}"

def check_gpu():
    """Check GPU availability."""
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return "GPU available:\\n" + result.stdout[:500]
        return "No GPU detected (running in CPU mode)"
    except Exception:
        return "GPU check failed"

with gr.Blocks(title="FamilyGPU Training Space") as demo:
    gr.Markdown("# FamilyGPU Training Space")
    gr.Markdown("{_TRAINING_WARNING}")
    gr.Markdown(f"Job ID: `{job_id}` | Profile: `{gpu_profile}`")

    with gr.Row():
        train_btn = gr.Button("▶ Start Training", variant="primary")
        gpu_btn = gr.Button("🔍 Check GPU")

    output = gr.Textbox(label="Output", lines=10, max_lines=30)

    train_btn.click(fn=start_training, outputs=output)
    gpu_btn.click(fn=check_gpu, outputs=output)

if __name__ == "__main__":
    demo.launch()
'''
    return gradio_app


def _generate_readme(job: dict) -> str:
    """Generate README.md for the HF Space."""
    job_id = job.get("id", str(uuid.uuid4())[:8])
    gpu_profile = job.get("gpu_profile", "small_gpu")

    sdk = "gradio"
    # ZeroGPU requires specific hardware setting
    if gpu_profile != "cpu_only":
        sdk = "gradio"

    return f'''---
title: FamilyGPU Training Space
emoji: 🔥
colorFrom: blue
colorTo: green
sdk: {sdk}
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# FamilyGPU Training Space

{_TRAINING_WARNING}

- Job ID: `{job_id}`
- GPU Profile: `{gpu_profile}`

## Usage

1. Click "Start Training" to run the embedded training script
2. Click "Check GPU" to verify GPU availability
3. For longer training, use a different provider (Colab, Kaggle, etc.)

## Checkpoints

Save checkpoints to the `/data/` directory (persistent storage).
'''


class HuggingFaceAdapter(ProviderAdapter):
    """HuggingFace Spaces adapter — Special/Limited (Class C).

    HuggingFace Spaces provides:
      - Free CPU hosting for Gradio/Streamlit apps
      - ZeroGPU: short GPU bursts for inference tasks
      - Persistent storage on paid plans
      - Git-based deployment via huggingface_hub

    IMPORTANT: Spaces are NOT designed for long-running training.
    ZeroGPU provides short GPU bursts (~60s) for inference, not
    sustained training workloads.

    This adapter:
      - Pushes a Gradio app wrapper to HF Spaces
      - Scans scripts for secrets before embedding
      - Returns explicit warnings about training limitations
      - Uses huggingface_hub for API operations
    """

    provider_key: str = "huggingface"
    display_name: str = "HuggingFace Spaces"
    provider_class: str = "C"
    automation_level: str = "manual"
    supports_auto: bool = False
    supported_profiles: list[str] = [
        "cpu_only", "small_gpu",
    ]

    # ── Credential helpers ──────────────────────────────────────────

    def _get_token(self, account: dict) -> str:
        """Retrieve HuggingFace API token from account."""
        creds = account.get("credentials", {}) or {}
        return creds.get("hf_token", "") or os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")

    # ── CLI helpers ─────────────────────────────────────────────────

    def _run_hf_cli(self, args: list[str], token: str,
                    timeout: int = HF_CLI_TIMEOUT) -> subprocess.CompletedProcess:
        """Run a huggingface-cli command via subprocess.

        Args:
            args: Command arguments after 'huggingface-cli'
            token: HF API token (set as env var)
            timeout: Timeout in seconds

        Returns:
            CompletedProcess instance
        """
        env = os.environ.copy()
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token

        cmd = ["huggingface-cli"] + args
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
            logger.warning(f"HF CLI timed out: {' '.join(args)}")
            return subprocess.CompletedProcess(
                args=cmd, returncode=-1,
                stdout="", stderr="Command timed out",
            )
        except FileNotFoundError:
            logger.debug("huggingface-cli not found — will try Python API")
            return subprocess.CompletedProcess(
                args=cmd, returncode=-1,
                stdout="", stderr="huggingface-cli not installed",
            )

    def _check_hf_hub_installed(self) -> bool:
        """Check if huggingface_hub is available."""
        try:
            import huggingface_hub
            return True
        except ImportError:
            return False

    # ── ProviderAdapter interface ───────────────────────────────────

    def validate_credentials(self, account: dict) -> ProviderResult:
        """Validate HuggingFace API token."""
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="HuggingFace token not configured. Set hf_token via /add.",
            )

        # Try huggingface_hub API
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            user_info = api.whoami()
            if user_info and isinstance(user_info, dict):
                username = user_info.get("name", "unknown")
                return ProviderResult(
                    ok=True, status="ok",
                    message=f"HuggingFace API authenticated as {redact_text(username)}",
                )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"HF API validation error: {e}")

        # Fallback: try CLI
        result = self._run_hf_cli(["whoami"], token, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return ProviderResult(
                ok=True, status="ok",
                message=f"HuggingFace CLI authenticated: {redact_text(result.stdout.strip())}",
            )

        return ProviderResult(
            ok=False, status="error",
            message="HuggingFace authentication failed — invalid or expired token",
        )

    def health_check(self, account: dict) -> ProviderResult:
        """Check if HuggingFace API is accessible."""
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="down",
                message="No HuggingFace token configured",
            )

        # Try to hit HF API
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.whoami()
            return ProviderResult(
                ok=True, status="ok",
                message="HuggingFace API is accessible",
            )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"HF health check error: {e}")

        # Fallback: try CLI
        result = self._run_hf_cli(["whoami"], token, timeout=10)
        if result.returncode == 0:
            return ProviderResult(
                ok=True, status="ok",
                message="HuggingFace API is accessible (via CLI)",
            )

        return ProviderResult(
            ok=False, status="down",
            message="HuggingFace API unreachable or auth failed",
        )

    def estimate_capacity(self, account: dict) -> ProviderResult:
        """Estimate GPU capacity for HuggingFace Spaces.

        ⚠ WARNING: HuggingFace Spaces is NOT designed for training!

        HuggingFace Spaces free tier:
          - CPU: 2 vCPU, 16 GB RAM (unlimited duration)
          - ZeroGPU: T4 (16 GB) for short bursts (~60s per request)
          - Storage: 50 GB persistent (paid plans)
          - No persistent GPU for training
        """
        return ProviderResult(
            ok=True, status="ok",
            message=f"HF Spaces: CPU only for training. {_TRAINING_WARNING}",
            data={
                "gpu_type": "None (ZeroGPU is for inference only)",
                "gpu_memory_gb": 0,
                "max_runtime_minutes": 0,  # No sustained GPU training
                "supports_checkpoint": False,
                "cpu_spec": "2 vCPU, 16 GB RAM",
                "zerogpu_note": "ZeroGPU provides T4 (16 GB) for ~60s inference bursts only",
                "warning": _TRAINING_WARNING,
            },
        )

    def start_job(self, lease: dict, job: dict, account: dict) -> ProviderResult:
        """Start a training job on HuggingFace Spaces.

        ⚠ WARNING: This is NOT suitable for long-running training!

        Pushes a Gradio app wrapper to a HF Space. The wrapper
        includes the training script (if safe to embed) and a
        simple UI for triggering training.

        Before embedding the training script, scan_for_secrets()
        is called. If secrets are found, the script is replaced
        with a safe loader stub.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="HuggingFace token not configured",
            )

        job_id = job.get("id", str(uuid.uuid4())[:8])

        # Generate the Gradio app wrapper (scans for secrets internally)
        app_code = _generate_gradio_wrapper(job, lease)
        readme_content = _generate_readme(job)

        # Try to push via huggingface_hub API
        space_id = f"familygpu-training-{job_id}"
        try:
            from huggingface_hub import HfApi, create_repo

            api = HfApi(token=token)
            user_info = api.whoami()
            username = user_info.get("name", "unknown") if isinstance(user_info, dict) else "unknown"
            space_id = f"{username}/familygpu-training-{job_id}"

            # Create the Space repo
            try:
                create_repo(
                    repo_id=space_id,
                    repo_type="space",
                    space_sdk="gradio",
                    token=token,
                    private=True,
                    exist_ok=True,
                )
            except Exception as e:
                logger.debug(f"Space creation error (may already exist): {e}")

            # Upload app.py
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", prefix="hf_app_", delete=False
            ) as f:
                f.write(app_code)
                app_path = f.name

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", prefix="hf_readme_", delete=False
            ) as f:
                f.write(readme_content)
                readme_path = f.name

            try:
                api.upload_file(
                    path_or_fileobj=app_path,
                    path_in_repo="app.py",
                    repo_id=space_id,
                    repo_type="space",
                    token=token,
                )
                api.upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=space_id,
                    repo_type="space",
                    token=token,
                )
            finally:
                os.unlink(app_path)
                os.unlink(readme_path)

            space_url = f"https://huggingface.co/spaces/{space_id}"

            return ProviderResult(
                ok=True, status="manual_required", manual=True,
                message=f"{_TRAINING_WARNING} Space pushed successfully.",
                data={
                    "space_id": space_id,
                    "space_url": space_url,
                    "warning": _TRAINING_WARNING,
                    "instructions": (
                        f"1. Open your Space: {space_url}\n"
                        "2. Wait for it to build (usually 2-5 minutes)\n"
                        "3. Click 'Start Training' in the Gradio UI\n"
                        "4. Use /confirm in TUI to start the countdown timer\n"
                        "\n"
                        "⚠ This Space is for SHORT experiments only!\n"
                        "For real training, use Colab, Kaggle, or a Class A provider."
                    ),
                },
            )

        except ImportError:
            logger.warning("huggingface_hub not installed — cannot push Space")
        except Exception as e:
            logger.error(f"Failed to push HF Space: {e}")

        # Fallback: manual instructions
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message=f"{_TRAINING_WARNING} This provider requires manual runtime start in MVP.",
            data={
                "warning": _TRAINING_WARNING,
                "app_code": app_code,
                "readme_content": readme_content,
                "instructions": (
                    "1. Go to https://huggingface.co/new-space\n"
                    "2. Create a Gradio Space (choose CPU or ZeroGPU)\n"
                    "3. Upload app.py and README.md (generated above)\n"
                    "4. Wait for the Space to build\n"
                    "5. Click 'Start Training' in the Gradio UI\n"
                    "6. Use /confirm in TUI to start the countdown timer\n"
                    "\n"
                    "⚠ Remember: Spaces are for demos/inference, NOT long training!"
                ),
            },
        )

    def stop_job(self, lease: dict, account: dict) -> ProviderResult:
        """Stop a running job on HuggingFace Spaces.

        Spaces don't have a 'stop' concept — they run continuously.
        The user should pause the Space or delete it.
        """
        token = self._get_token(account)
        space_id = lease.get("provider_data", {}).get("space_id", "")

        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Stop the Space via HuggingFace UI (pause or delete)",
            data={
                "space_id": space_id,
                "warning": _TRAINING_WARNING,
                "instructions": (
                    "1. Go to your Space settings\n"
                    "2. Click 'Pause this Space' to stop it\n"
                    "   Or delete it if no longer needed\n"
                    f"3. Space URL: https://huggingface.co/spaces/{space_id}\n"
                    "4. Use /confirm in TUI to mark the job as stopped"
                ),
            },
        )

    def fetch_logs(self, lease: dict, account: dict) -> ProviderResult:
        """Fetch logs from HuggingFace Space.

        Space logs are available via the HF API or UI.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="HuggingFace token not configured",
            )

        space_id = lease.get("provider_data", {}).get("space_id", "")
        if space_id:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=token)
                # Try to get Space runtime info
                runtime = api.space_info(repo_id=space_id)
                if runtime:
                    stage = getattr(runtime, "stage", "unknown")
                    return ProviderResult(
                        ok=True, status="ok",
                        message=f"Space status: {stage}",
                        data={
                            "logs": f"Space stage: {stage}. Check UI for detailed logs.",
                            "space_url": f"https://huggingface.co/spaces/{space_id}",
                            "stage": stage,
                        },
                    )
            except Exception as e:
                logger.debug(f"Failed to fetch Space info: {e}")

        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Logs not available via API. Check Space logs in browser.",
            data={
                "logs": "Manual check required — view Space logs in HF UI",
                "warning": _TRAINING_WARNING,
                "instructions": (
                    "1. Go to your Space page\n"
                    "2. Click 'Logs' tab to view runtime logs\n"
                    f"3. Space URL: https://huggingface.co/spaces/{space_id}"
                ),
            },
        )

    def sync_checkpoint(self, lease: dict, account: dict,
                        checkpoint_uri: str) -> ProviderResult:
        """Sync checkpoints from HuggingFace Space.

        Spaces can write to /data/ directory (persistent on paid plans).
        For free tier, checkpoints are lost when the Space restarts.
        Best approach: upload checkpoints as a HF Dataset.
        """
        token = self._get_token(account)
        if not token:
            return ProviderResult(
                ok=False, status="error",
                message="HuggingFace token not configured",
            )

        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Checkpoint sync via HF Datasets (manual in MVP).",
            data={
                "checkpoint_uri": checkpoint_uri,
                "warning": _TRAINING_WARNING,
                "instructions": (
                    "1. Save checkpoints in your Space to /data/:\n"
                    "   torch.save(model.state_dict(), '/data/checkpoint.pt')\n"
                    "2. Download from Space files (Settings → Files)\n"
                    "3. Or create a HF Dataset and upload:\n"
                    "   huggingface-cli dataset create familygpu-checkpoints\n"
                    f"4. Target checkpoint URI: {checkpoint_uri}\n"
                    "\n"
                    "⚠ Free Spaces lose /data/ on restart —\n"
                    "upload to a HF Dataset for persistence."
                ),
            },
        )
