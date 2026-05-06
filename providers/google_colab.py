"""Google Colab provider adapter for FamilyGPU Orchestrator.

Class B adapter — Notebook/Interactive. Requires manual start.
Generates .ipynb notebooks for upload to Colab.

For MVP: start_job generates a notebook and returns manual_required.
Stop, logs, and checkpoint sync are all manual operations.

Auth: account.credentials = {
    "email": "...",  # For identification only, not API auth
}

Security: Before embedding training scripts into generated notebooks,
scan_for_secrets() is called. If secrets are detected, the script
content is replaced with a safe loader stub instead of embedding.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

from providers.base import ProviderAdapter, ProviderResult
from vault import redact_text, scan_for_secrets

logger = logging.getLogger("fgt.providers.google_colab")

# ── Notebook generation constants ──────────────────────────────────

COLAB_SETUP_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": ""},
    "outputs": [],
    "source": [
        "# FamilyGPU Orchestrator — Google Colab Setup\n",
        "# Auto-generated notebook. DO NOT EDIT this cell.\n",
        "import os, sys\n",
        "print('FamilyGPU: Colab runtime initialized')\n",
        "\n",
        "# Check GPU availability\n",
        "gpu_info = !nvidia-smi 2>/dev/null\n",
        "if gpu_info:\n",
        "    print('GPU available:')\n",
        "    print('\\n'.join(gpu_info))\n",
        "else:\n",
        "    print('No GPU detected — using CPU-only mode')\n",
    ],
}

SAFE_LOADER_STUB = [
    "# FamilyGPU Orchestrator — Training Script Loader\n",
    "# WARNING: Your training script contains secrets that cannot be embedded.\n",
    "# Please upload your script manually and run it from the next cell.\n",
    "#\n",
    "# Option 1: Upload via Colab file browser (folder icon on left)\n",
    "# Option 2: Clone from a private repo:\n",
    "#   !git clone <YOUR_REPO_URL> training_script\n",
    "#   %run training_script/train.py\n",
    "#\n",
    "# Option 3: Use Google Drive:\n",
    "#   from google.colab import drive\n",
    "#   drive.mount('/content/drive')\n",
    "#   %run /content/drive/MyDrive/path/to/train.py\n",
    "print('Please load your training script manually (see instructions above)')\n",
]

CHECKPOINT_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": ""},
    "outputs": [],
    "source": [
        "# FamilyGPU — Checkpoint Sync via Google Drive\n",
        "from google.colab import drive\n",
        "import shutil, os\n",
        "\n",
        "# Mount Google Drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# Save checkpoints to Drive\n",
        "CHECKPOINT_DIR = '/content/drive/MyDrive/familygpu_checkpoints'\n",
        "os.makedirs(CHECKPOINT_DIR, exist_ok=True)\n",
        "print(f'Checkpoint directory: {CHECKPOINT_DIR}')\n",
        "print('Save your model checkpoints here to persist across sessions.')\n",
    ],
}


def _generate_notebook(job: dict, lease: dict) -> dict:
    """Generate a Jupyter notebook dict for Colab upload.

    Scans the training script for secrets before embedding.
    If secrets are found, replaces script content with a safe loader stub.

    Args:
        job: Job dict containing training_script and other metadata
        lease: Lease dict for context

    Returns:
        Notebook dict ready for json.dumps()
    """
    cells = []
    job_id = job.get("id", str(uuid.uuid4())[:8])

    # ── Title cell ────────────────────────────────────────────────
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": f"title-{job_id}"},
        "source": [
            f"# FamilyGPU Training Job: {job.get('name', job_id)}\n",
            f"Generated: {datetime.utcnow().isoformat()}Z\n",
            f"Provider: Google Colab | Profile: {job.get('gpu_profile', 'unknown')}\n",
        ],
    })

    # ── Setup cell (with unique ID) ──────────────────────────────
    setup_cell = dict(COLAB_SETUP_CELL)
    setup_cell["metadata"] = {"id": f"setup-{job_id}"}
    cells.append(setup_cell)

    # ── Training script cell ─────────────────────────────────────
    training_script = job.get("training_script", "")
    if training_script:
        secret_warnings = scan_for_secrets(training_script)
        if secret_warnings:
            # SECRETS DETECTED — do NOT embed the raw script
            logger.warning(
                f"Secrets detected in training script for job {job_id}: "
                f"{'; '.join(secret_warnings)}. Using safe loader stub."
            )
            script_source = list(SAFE_LOADER_STUB)
            script_source.insert(0, f"# Security warnings: {'; '.join(secret_warnings)}\n")
        else:
            # Safe to embed — redact as a final precaution
            safe_script = redact_text(training_script)
            script_source = safe_script.splitlines(keepends=True)
            if script_source and not script_source[0].startswith("# FamilyGPU"):
                script_source.insert(0, "# FamilyGPU — Training Script\n")
    else:
        script_source = [
            "# No training script provided in job config.\n",
            "# Add your training code here.\n",
            "print('No training script provided — add your code in this cell')\n",
        ]

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": f"train-{job_id}"},
        "outputs": [],
        "source": script_source,
    })

    # ── Checkpoint sync cell ─────────────────────────────────────
    ckpt_cell = dict(CHECKPOINT_CELL)
    ckpt_cell["metadata"] = {"id": f"checkpoint-{job_id}"}
    cells.append(ckpt_cell)

    # ── Build notebook structure ─────────────────────────────────
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": f"familygpu_{job_id}.ipynb"},
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "accelerator": "GPU" if job.get("gpu_profile") != "cpu_only" else "None",
        },
        "cells": cells,
    }
    return notebook


class GoogleColabAdapter(ProviderAdapter):
    """Google Colab adapter — Notebook/Interactive (Class B).

    Google Colab provides free GPU access (T4, occasionally V100/A100)
    via interactive Jupyter notebooks. This adapter:
      - Generates .ipynb notebooks for manual upload
      - Scans scripts for secrets before embedding
      - Provides checkpoint sync via Google Drive instructions
      - Requires manual runtime start/stop in browser

    Limitations:
      - Sessions timeout after 12 hours (free) or 24 hours (Pro)
      - GPU allocation is not guaranteed (varies by demand)
      - No programmatic API for starting/stopping runtimes
    """

    provider_key: str = "google_colab"
    display_name: str = "Google Colab"
    provider_class: str = "B"
    automation_level: str = "manual"
    supports_auto: bool = False
    supported_profiles: list[str] = [
        "cpu_only", "small_gpu", "medium_gpu",
    ]

    # ── Credential helpers ──────────────────────────────────────────

    def _get_email(self, account: dict) -> str:
        """Retrieve email from account credentials (for identification only)."""
        creds = account.get("credentials", {}) or {}
        return creds.get("email", "")

    # ── ProviderAdapter interface ───────────────────────────────────

    def validate_credentials(self, account: dict) -> ProviderResult:
        """Validate Google Colab credentials.

        Google Colab doesn't use API keys — we only verify that
        an email is configured for identification purposes.
        """
        email = self._get_email(account)
        if not email:
            return ProviderResult(
                ok=False, status="error",
                message="Google Colab requires an email for identification. Set 'email' via /add.",
            )

        return ProviderResult(
            ok=True, status="ok",
            message=f"Google Colab identified as {redact_text(email)}",
            data={"note": "Colab uses browser auth — no API key validation possible"},
        )

    def health_check(self, account: dict) -> ProviderResult:
        """Check Google Colab accessibility.

        Since Colab has no API, we can only verify the email is set
        and provide a link to check service status.
        """
        email = self._get_email(account)
        if not email:
            return ProviderResult(
                ok=False, status="down",
                message="No email configured for Google Colab",
            )

        # Colab has no health endpoint — assume it's available
        return ProviderResult(
            ok=True, status="ok",
            message="Google Colab is typically available (no API health check possible)",
            data={
                "status_url": "https://status.cloud.google.com/",
                "colab_url": "https://colab.research.google.com/",
            },
        )

    def estimate_capacity(self, account: dict) -> ProviderResult:
        """Estimate GPU capacity for Google Colab.

        Google Colab free tier:
          - CPU: 2-core Intel Xeon, ~13 GB RAM
          - GPU: NVIDIA T4 (16 GB VRAM) — most common free allocation
          - Occasionally: V100 (16 GB) or A100 (40 GB) for short periods
          - Max runtime: ~12 hours (free), ~24 hours (Pro)
          - GPU allocation is NOT guaranteed (varies by demand)
        """
        return ProviderResult(
            ok=True, status="ok",
            message="Colab free tier: T4 16 GB (subject to availability)",
            data={
                "gpu_type": "NVIDIA T4 (typical free tier)",
                "gpu_memory_gb": 16,
                "max_runtime_minutes": 720,  # 12 hours
                "supports_checkpoint": True,
                "gpu_availability": "not_guaranteed",
                "alternative_gpus": ["V100 (16 GB)", "A100 (40 GB) — rare"],
                "note": "GPU allocation varies by demand; CPU fallback is always available",
            },
        )

    def start_job(self, lease: dict, job: dict, account: dict) -> ProviderResult:
        """Start a training job on Google Colab.

        Generates a .ipynb notebook file for manual upload.
        The notebook includes:
          1. Setup cell (GPU detection)
          2. Training script (scanned for secrets)
          3. Checkpoint sync cell (Google Drive mount)

        Returns manual_required since Colab requires browser-based start.
        """
        email = self._get_email(account)
        if not email:
            return ProviderResult(
                ok=False, status="error",
                message="Google Colab email not configured",
            )

        # Generate the notebook
        notebook = _generate_notebook(job, lease)
        job_id = job.get("id", str(uuid.uuid4())[:8])
        notebook_json = json.dumps(notebook, indent=2)

        # Store notebook in data for retrieval
        notebook_filename = f"familygpu_colab_{job_id}.ipynb"

        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="This provider requires manual runtime start in MVP.",
            data={
                "notebook": notebook_json,
                "notebook_filename": notebook_filename,
                "instructions": (
                    "1. Download the generated notebook file\n"
                    "2. Go to https://colab.research.google.com/\n"
                    "3. Upload the notebook: File → Upload notebook\n"
                    "4. Change runtime type: Runtime → Change runtime type → GPU (T4)\n"
                    "5. Run all cells: Runtime → Run all\n"
                    "6. Use /confirm in TUI to start the countdown timer\n"
                    "7. Save checkpoints to Google Drive (see checkpoint cell in notebook)"
                ),
                "colab_url": "https://colab.research.google.com/",
            },
        )

    def stop_job(self, lease: dict, account: dict) -> ProviderResult:
        """Stop a running job on Google Colab.

        Colab has no API for stopping runtimes.
        The user must manually stop the runtime in the browser.
        """
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Stop in browser: Runtime → Disconnect and delete runtime",
            data={
                "instructions": (
                    "1. In your Colab browser tab, go to Runtime menu\n"
                    "2. Select 'Disconnect and delete runtime'\n"
                    "3. Confirm the disconnect\n"
                    "4. Use /confirm in TUI to mark the job as stopped"
                ),
            },
        )

    def fetch_logs(self, lease: dict, account: dict) -> ProviderResult:
        """Fetch logs from Google Colab.

        Colab has no log API — logs are only visible in the browser.
        """
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Logs not available via API. Check your Colab browser tab for output.",
            data={
                "logs": "Manual check required — view cell outputs in Colab browser",
                "instructions": (
                    "1. Open your Colab notebook in the browser\n"
                    "2. Check cell outputs for training logs\n"
                    "3. For detailed logs, check any log files written to /content/"
                ),
            },
        )

    def sync_checkpoint(self, lease: dict, account: dict,
                        checkpoint_uri: str) -> ProviderResult:
        """Sync checkpoints to/from Google Colab via Google Drive.

        Checkpoint sync is manual — the notebook includes a Google Drive
        mount cell. Users save checkpoints to Drive and download them.
        """
        return ProviderResult(
            ok=True, status="manual_required", manual=True,
            message="Checkpoint sync via Google Drive mount (manual in MVP).",
            data={
                "checkpoint_uri": checkpoint_uri,
                "instructions": (
                    "1. Run the checkpoint cell in your Colab notebook to mount Drive\n"
                    "2. Save your model to /content/drive/MyDrive/familygpu_checkpoints/\n"
                    "3. Download checkpoints from Google Drive to local machine\n"
                    f"4. Upload to next provider's checkpoint location: {checkpoint_uri}\n"
                    "\n"
                    "Python snippet for saving:\n"
                    "  import torch\n"
                    "  torch.save(model.state_dict(), '/content/drive/MyDrive/familygpu_checkpoints/model.pt')\n"
                    "\n"
                    "Python snippet for loading:\n"
                    "  model.load_state_dict(torch.load('/content/drive/MyDrive/familygpu_checkpoints/model.pt'))"
                ),
            },
        )
