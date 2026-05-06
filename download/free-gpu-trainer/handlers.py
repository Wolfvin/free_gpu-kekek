"""Real platform handlers — actual API integrations for free GPU platforms.

Each handler implements:
  - push_code(account, script_path)  → push training code to the platform
  - start_session(account)           → start a notebook/runtime session
  - check_status(account)            → check if session is still running
  - stop_session(account)            → stop a running session
  - get_gpu_info(account)            → get GPU type and availability
  - is_available(account)            → check if platform is accessible right now

All methods return dicts with 'ok', 'message', and optional data.
"""

import os
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

logger = logging.getLogger("fgt.handler")


# ── Base Handler ───────────────────────────────────────────────────

class PlatformHandler(ABC):
    """Base class for all platform handlers."""

    key: str = ""
    name: str = ""

    @abstractmethod
    def push_code(self, account, script_path: str, checkpoint_dir: str = "./checkpoints") -> dict:
        """Push training code to the platform."""
        pass

    @abstractmethod
    def start_session(self, account) -> dict:
        """Start a training session."""
        pass

    @abstractmethod
    def check_status(self, account) -> dict:
        """Check session status."""
        pass

    @abstractmethod
    def stop_session(self, account) -> dict:
        """Stop a running session."""
        pass

    def is_available(self, account) -> dict:
        """Check if the platform is accessible right now."""
        return {"ok": True, "message": "Unknown", "available": None}

    def get_gpu_info(self, account) -> dict:
        """Get GPU info."""
        return {"ok": True, "gpu": "Unknown"}


# ── Kaggle Handler ─────────────────────────────────────────────────

class KaggleHandler(PlatformHandler):
    """Kaggle Notebooks API handler.

    Uses the kaggle Python package to:
    - Push notebooks via `kaggle kernels push`
    - Check status via `kaggle kernels status`
    - Pull output via `kaggle kernels output`

    Auth: Set KAGGLE_USERNAME and KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json
    """

    key = "kaggle"
    name = "Kaggle Notebooks"

    def _get_client(self, account):
        """Get authenticated Kaggle API client."""
        try:
            import os
            # Set env vars before importing so kaggle doesn't prompt
            if account.api_key:
                os.environ["KAGGLE_KEY"] = account.api_key
            if account.token:
                try:
                    creds = json.loads(account.token)
                    os.environ["KAGGLE_USERNAME"] = creds.get("username", "")
                    os.environ["KAGGLE_KEY"] = creds.get("key", "")
                except (json.JSONDecodeError, TypeError):
                    pass

            # Check if credentials exist before importing
            has_creds = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
            if not has_creds:
                # Also check ~/.kaggle/kaggle.json
                kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
                if kaggle_json.exists():
                    has_creds = True

            if not has_creds:
                return None

            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            return api
        except ImportError:
            return None
        except Exception as e:
            logger.debug(f"Kaggle auth failed for {account.name}: {e}")
            return None

    def push_code(self, account, script_path: str, checkpoint_dir: str = "./checkpoints") -> dict:
        """Push a training script as a Kaggle kernel.

        Creates a kernel metadata JSON and pushes it via the API.
        """
        api = self._get_client(account)
        if not api:
            return {"ok": False, "message": "Kaggle auth failed. Set KAGGLE_USERNAME + KAGGLE_KEY env vars."}

        script = Path(script_path)
        if not script.exists():
            return {"ok": False, "message": f"Script not found: {script_path}"}

        # Create kernel metadata
        kernel_meta = {
            "id": f"{os.environ.get('KAGGLE_USERNAME', 'user')}/{account.name}-training",
            "title": f"{account.name}-training",
            "code_file": str(script),
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "competition_sources": [],
            "dataset_sources": [],
            "kernel_sources": [],
        }

        # Write metadata
        meta_path = script.parent / "kernel-metadata.json"
        with open(meta_path, "w") as f:
            json.dump(kernel_meta, f, indent=2)

        try:
            # Push the kernel
            result = subprocess.run(
                ["kaggle", "kernels", "push", "-p", str(script.parent)],
                capture_output=True, text=True, timeout=60,
                env={**os.environ, "KAGGLE_USERNAME": os.environ.get("KAGGLE_USERNAME", ""),
                     "KAGGLE_KEY": os.environ.get("KAGGLE_KEY", "")},
            )
            if result.returncode == 0:
                return {"ok": True, "message": f"Kernel pushed: {result.stdout.strip()}"}
            else:
                return {"ok": False, "message": f"Push failed: {result.stderr.strip()}"}
        except FileNotFoundError:
            return {"ok": False, "message": "kaggle CLI not found. Install: pip install kaggle"}
        except Exception as e:
            return {"ok": False, "message": f"Push error: {e}"}

    def start_session(self, account) -> dict:
        """Start a Kaggle kernel session (push = start on Kaggle)."""
        return self.push_code(account, "train.py")

    def check_status(self, account) -> dict:
        """Check kernel status via kaggle API."""
        api = self._get_client(account)
        if not api:
            return {"ok": False, "message": "Auth failed", "status": "unknown"}

        username = os.environ.get("KAGGLE_USERNAME", "user")
        slug = f"{account.name}-training"
        try:
            result = subprocess.run(
                ["kaggle", "kernels", "status", f"{username}/{slug}"],
                capture_output=True, text=True, timeout=30,
            )
            status_text = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
            return {"ok": True, "status": status_text, "message": status_text}
        except Exception as e:
            return {"ok": False, "message": str(e), "status": "error"}

    def stop_session(self, account) -> dict:
        """Kaggle doesn't support stopping kernels via API — they auto-timeout."""
        return {"ok": True, "message": "Kaggle kernels auto-stop after session limit"}

    def is_available(self, account) -> dict:
        """Check if Kaggle API is accessible."""
        api = self._get_client(account)
        if api:
            return {"ok": True, "available": True, "message": "Kaggle API authenticated"}
        return {"ok": False, "available": False, "message": "Kaggle auth failed"}


# ── HuggingFace Handler ────────────────────────────────────────────

class HuggingFaceHandler(PlatformHandler):
    """HuggingFace Spaces/ZeroGPU handler.

    Uses huggingface_hub to:
    - Create/manage Spaces with GPU hardware
    - Upload training code
    - Start/stop Spaces

    Auth: Set HF_TOKEN env var or pass token in account config
    """

    key = "huggingface"
    name = "Hugging Face Spaces"

    def _get_client(self, account):
        """Get authenticated HuggingFace API client."""
        try:
            from huggingface_hub import HfApi
            token = account.token or os.environ.get("HF_TOKEN")
            api = HfApi(token=token)
            return api
        except ImportError:
            return None
        except Exception as e:
            logger.warning(f"HF auth failed for {account.name}: {e}")
            return None

    def push_code(self, account, script_path: str, checkpoint_dir: str = "./checkpoints") -> dict:
        """Push training code to a HuggingFace Space."""
        api = self._get_client(account)
        if not api:
            return {"ok": False, "message": "HF auth failed. Set HF_TOKEN env var."}

        script = Path(script_path)
        if not script.exists():
            return {"ok": False, "message": f"Script not found: {script_path}"}

        # Generate app.py that wraps the training script as a Gradio app
        app_code = self._generate_gradio_app(script_path, checkpoint_dir)

        repo_name = f"{account.name}-trainer"
        username = None
        try:
            who = api.whoami()
            username = who.get("name", "user")
        except Exception:
            pass

        repo_id = f"{username}/{repo_name}" if username else repo_name

        try:
            # Try to create the Space (will fail if exists, that's ok)
            try:
                from huggingface_hub import create_repo
                create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", token=account.token or os.environ.get("HF_TOKEN"), exist_ok=True)
            except Exception:
                pass

            # Upload the training script and app
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(app_code)
                app_path = f.name

            api.upload_file(path_or_fileobj=app_path, path_in_repo="app.py", repo_id=repo_id, repo_type="space")
            api.upload_file(path_or_fileobj=str(script), path_in_repo=script.name, repo_id=repo_id, repo_type="space")

            os.unlink(app_path)

            return {"ok": True, "message": f"Pushed to Space: {repo_id}", "repo_id": repo_id}
        except Exception as e:
            return {"ok": False, "message": f"Push error: {e}"}

    def _generate_gradio_app(self, script_path: str, checkpoint_dir: str) -> str:
        """Generate a Gradio app.py that runs the training script."""
        return f'''import gradio as gr
import subprocess
import threading
import os

def train(progress=gr.Progress()):
    """Run the training script."""
    result = subprocess.run(
        ["python", "{script_path}", "--checkpoint-dir", "{checkpoint_dir}"],
        capture_output=True, text=True
    )
    return result.stdout[-2000:] if result.stdout else result.stderr[-2000:]

with gr.Blocks() as demo:
    gr.Markdown("# Free GPU Trainer - HuggingFace Space")
    btn = gr.Button("Start Training")
    output = gr.Textbox(label="Output", lines=20)
    btn.click(fn=train, outputs=output)

if __name__ == "__main__":
    demo.launch()
'''

    def start_session(self, account) -> dict:
        """Start a HF Space (push code = starts the Space)."""
        return self.push_code(account, "train.py")

    def check_status(self, account) -> dict:
        """Check Space status."""
        api = self._get_client(account)
        if not api:
            return {"ok": False, "message": "Auth failed", "status": "unknown"}

        try:
            from huggingface_hub import SpaceRuntime
            info = api.space_info(repo_id=f"{api.whoami().get('name', 'user')}/{account.name}-trainer")
            runtime = info.runtime if hasattr(info, 'runtime') else None
            stage = runtime.stage if runtime else "unknown"
            return {"ok": True, "status": stage, "message": f"Space status: {stage}"}
        except Exception as e:
            return {"ok": False, "message": str(e), "status": "error"}

    def stop_session(self, account) -> dict:
        """Stop a HF Space by pausing it."""
        api = self._get_client(account)
        if not api:
            return {"ok": False, "message": "Auth failed"}
        try:
            # Pause the space to free GPU
            from huggingface_hub import pause_space
            username = api.whoami().get("name", "user")
            pause_space(f"{username}/{account.name}-trainer", token=account.token or os.environ.get("HF_TOKEN"))
            return {"ok": True, "message": "Space paused (GPU freed)"}
        except Exception as e:
            return {"ok": False, "message": f"Stop error: {e}"}

    def is_available(self, account) -> dict:
        api = self._get_client(account)
        if api:
            return {"ok": True, "available": True, "message": "HF API authenticated"}
        return {"ok": False, "available": False, "message": "HF auth failed"}


# ── Google Colab Handler ───────────────────────────────────────────

class GoogleColabHandler(PlatformHandler):
    """Google Colab handler.

    Colab has NO public API for creating/managing runtimes.
    Strategy: Generate notebook + use google-colabtools CLI if available,
    otherwise generate a .ipynb file the user manually uploads.

    For automation, we use the approach of:
    1. Generating a .ipynb notebook file
    2. Using selenium or colab-cli to push it (if installed)
    3. Otherwise, telling the user to upload it manually
    """

    key = "google_colab"
    name = "Google Colab"

    def push_code(self, account, script_path: str, checkpoint_dir: str = "./checkpoints") -> dict:
        """Generate a Colab notebook from the training script."""
        script = Path(script_path)
        if not script.exists():
            return {"ok": False, "message": f"Script not found: {script_path}"}

        script_content = script.read_text()

        # Generate .ipynb notebook
        notebook = self._generate_notebook(script_content, checkpoint_dir, account.name)
        nb_path = Path(f"colab_{account.name}_training.ipynb")
        nb_path.write_text(json.dumps(notebook, indent=2))

        # Try to push via colab-cli if available
        try:
            result = subprocess.run(
                ["colab-cli", "upload", str(nb_path)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return {"ok": True, "message": f"Uploaded to Colab: {result.stdout.strip()}"}
        except FileNotFoundError:
            pass

        return {
            "ok": True,
            "message": f"Notebook generated: {nb_path} — open in Colab and run",
            "notebook_path": str(nb_path),
            "manual": True,
            "url": "https://colab.research.google.com/",
        }

    def _generate_notebook(self, script_content: str, checkpoint_dir: str, account_name: str) -> dict:
        """Generate a Jupyter notebook dict for Colab."""
        cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Free GPU Trainer — {account_name}\n",
                    "# Auto-generated notebook — just click Runtime > Run all\n",
                    "# Make sure to enable GPU: Runtime > Change runtime type > T4 GPU"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {"id": "setup"},
                "source": [
                    "# Setup\n",
                    "!pip install -q torch torchvision transformers accelerate peft datasets 2>/dev/null\n",
                    "\n",
                    "import torch\n",
                    "print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')\n",
                    "print(f'CUDA: {torch.cuda.is_available()}')\n",
                    "\n",
                    "# Mount Google Drive for checkpoints\n",
                    "try:\n",
                    "    from google.colab import drive\n",
                    "    drive.mount('/content/drive')\n",
                    "    import os\n",
                    "    os.environ['CHECKPOINT_DIR'] = '/content/drive/MyDrive/checkpoints'\n",
                    "except:\n",
                    "    import os\n",
                    "    os.environ['CHECKPOINT_DIR'] = './checkpoints'\n",
                ],
                "execution_count": None,
                "outputs": [],
            },
            {
                "cell_type": "code",
                "metadata": {"id": "training"},
                "source": [
                    "# Training Script\n",
                    script_content,
                ],
                "execution_count": None,
                "outputs": [],
            },
        ]

        return {
            "nbformat": 4,
            "nbformat_minor": 0,
            "metadata": {
                "colab": {"provenance": [], "gpuType": "T4"},
                "kernelspec": {"name": "python3", "display_name": "Python 3"},
                "accelerator": "GPU",
            },
            "cells": cells,
        }

    def start_session(self, account) -> dict:
        """Generate notebook for manual upload."""
        return self.push_code(account, "train.py")

    def check_status(self, account) -> dict:
        """Colab has no status API — user checks manually."""
        return {"ok": True, "status": "unknown", "message": "Colab has no status API — check browser"}

    def stop_session(self, account) -> dict:
        return {"ok": True, "message": "Stop Colab manually in browser (Runtime > Disconnect)"}

    def is_available(self, account) -> dict:
        return {"ok": True, "available": True, "message": "Colab notebooks can be created anytime"}


# ── Oracle Cloud Handler ───────────────────────────────────────────

class OracleCloudHandler(PlatformHandler):
    """Oracle Cloud Free Tier handler.

    Uses OCI Python SDK to manage always-free Ampere A1 compute instances.
    Can run Ollama/transformers on the free tier VM.

    Auth: Set OCI config via ~/.oci/config or env vars
    """

    key = "oracle_cloud"
    name = "Oracle Cloud Free Tier"

    def push_code(self, account, script_path: str, checkpoint_dir: str = "./checkpoints") -> dict:
        """Push training code via SSH to Oracle Cloud VM."""
        script = Path(script_path)
        if not script.exists():
            return {"ok": False, "message": f"Script not found: {script_path}"}

        # Try SSH-based push (user must configure SSH key)
        host = os.environ.get("OCI_VM_HOST", "")
        user = os.environ.get("OCI_VM_USER", "opc")
        key_file = os.environ.get("OCI_SSH_KEY", "~/.ssh/id_rsa")

        if not host:
            return {
                "ok": True,
                "message": "Set OCI_VM_HOST env var to push code. Manual: scp train.py opc@<vm-ip>:~/",
                "manual": True,
            }

        try:
            # SCP the training script
            result = subprocess.run(
                ["scp", "-i", os.path.expanduser(key_file), str(script), f"{user}@{host}:~/train.py"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return {"ok": True, "message": f"Pushed to {host}"}
            return {"ok": False, "message": f"SCP failed: {result.stderr}"}
        except Exception as e:
            return {"ok": False, "message": f"Push error: {e}"}

    def start_session(self, account) -> dict:
        """Start training via SSH on Oracle Cloud VM."""
        host = os.environ.get("OCI_VM_HOST", "")
        user = os.environ.get("OCI_VM_USER", "opc")
        key_file = os.environ.get("OCI_SSH_KEY", "~/.ssh/id_rsa")

        if not host:
            return {
                "ok": True,
                "message": "Set OCI_VM_HOST. Manual: ssh opc@<vm-ip> 'python train.py'",
                "manual": True,
            }

        try:
            result = subprocess.run(
                ["ssh", "-i", os.path.expanduser(key_file), f"{user}@{host}",
                 "nohup python ~/train.py > ~/training.log 2>&1 &"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return {"ok": True, "message": f"Training started on {host}"}
            return {"ok": False, "message": f"SSH failed: {result.stderr}"}
        except Exception as e:
            return {"ok": False, "message": f"Start error: {e}"}

    def check_status(self, account) -> dict:
        host = os.environ.get("OCI_VM_HOST", "")
        if not host:
            return {"ok": True, "status": "unknown", "message": "No OCI_VM_HOST configured"}
        user = os.environ.get("OCI_VM_USER", "opc")
        key_file = os.environ.get("OCI_SSH_KEY", "~/.ssh/id_rsa")
        try:
            result = subprocess.run(
                ["ssh", "-i", os.path.expanduser(key_file), f"{user}@{host}",
                 "ps aux | grep train.py | grep -v grep || echo 'not running'"],
                capture_output=True, text=True, timeout=10,
            )
            running = "train.py" in result.stdout and "not running" not in result.stdout
            return {"ok": True, "status": "running" if running else "stopped", "message": result.stdout.strip()}
        except Exception as e:
            return {"ok": False, "message": str(e), "status": "error"}

    def stop_session(self, account) -> dict:
        host = os.environ.get("OCI_VM_HOST", "")
        if not host:
            return {"ok": True, "message": "No OCI_VM_HOST — stop manually"}
        user = os.environ.get("OCI_VM_USER", "opc")
        key_file = os.environ.get("OCI_SSH_KEY", "~/.ssh/id_rsa")
        try:
            result = subprocess.run(
                ["ssh", "-i", os.path.expanduser(key_file), f"{user}@{host}", "pkill -f train.py"],
                capture_output=True, text=True, timeout=10,
            )
            return {"ok": True, "message": "Training process killed"}
        except Exception as e:
            return {"ok": False, "message": str(e)}


# ── Generic SSH Handler (for GCP, AWS, etc.) ──────────────────────

class SSHHandler(PlatformHandler):
    """Generic SSH-based handler for cloud VMs (GCP, Oracle, etc)."""

    def __init__(self, key: str, name: str):
        self.key = key
        self.name = name

    def _ssh_config(self, account) -> tuple:
        """Get SSH host, user, key from env vars."""
        prefix = self.key.upper()
        host = os.environ.get(f"{prefix}_VM_HOST", "")
        user = os.environ.get(f"{prefix}_VM_USER", "ubuntu")
        key_file = os.environ.get(f"{prefix}_SSH_KEY", "~/.ssh/id_rsa")
        return host, user, key_file

    def push_code(self, account, script_path: str, checkpoint_dir: str = "./checkpoints") -> dict:
        host, user, key_file = self._ssh_config(account)
        if not host:
            return {"ok": True, "message": f"Set {self.key.upper()}_VM_HOST env var", "manual": True}
        script = Path(script_path)
        if not script.exists():
            return {"ok": False, "message": f"Script not found: {script_path}"}
        try:
            result = subprocess.run(
                ["scp", "-i", os.path.expanduser(key_file), str(script), f"{user}@{host}:~/train.py"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return {"ok": True, "message": f"Pushed to {host}"}
            return {"ok": False, "message": f"SCP failed: {result.stderr}"}
        except Exception as e:
            return {"ok": False, "message": str(e)}

    def start_session(self, account) -> dict:
        host, user, key_file = self._ssh_config(account)
        if not host:
            return {"ok": True, "message": f"Set {self.key.upper()}_VM_HOST env var", "manual": True}
        try:
            result = subprocess.run(
                ["ssh", "-i", os.path.expanduser(key_file), f"{user}@{host}",
                 "nohup python ~/train.py > ~/training.log 2>&1 &"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return {"ok": True, "message": f"Training started on {host}"}
            return {"ok": False, "message": f"SSH failed: {result.stderr}"}
        except Exception as e:
            return {"ok": False, "message": str(e)}

    def check_status(self, account) -> dict:
        host, user, key_file = self._ssh_config(account)
        if not host:
            return {"ok": True, "status": "unknown", "message": "No host configured"}
        try:
            result = subprocess.run(
                ["ssh", "-i", os.path.expanduser(key_file), f"{user}@{host}",
                 "ps aux | grep train.py | grep -v grep || echo 'not running'"],
                capture_output=True, text=True, timeout=10,
            )
            running = "train.py" in result.stdout and "not running" not in result.stdout
            return {"ok": True, "status": "running" if running else "stopped", "message": result.stdout.strip()}
        except Exception as e:
            return {"ok": False, "message": str(e), "status": "error"}

    def stop_session(self, account) -> dict:
        host, user, key_file = self._ssh_config(account)
        if not host:
            return {"ok": True, "message": "No host — stop manually"}
        try:
            subprocess.run(
                ["ssh", "-i", os.path.expanduser(key_file), f"{user}@{host}", "pkill -f train.py"],
                capture_output=True, text=True, timeout=10,
            )
            return {"ok": True, "message": "Training process killed"}
        except Exception as e:
            return {"ok": False, "message": str(e)}


# ── Notebook Generator Handler (fallback for notebook platforms) ───

class NotebookHandler(PlatformHandler):
    """Handler for notebook-based platforms that don't have push APIs.

    Generates .ipynb files for manual upload.
    Works for: SageMaker Studio Lab, Paperspace, Deepnote, Lightning AI, Codesphere
    """

    def __init__(self, key: str, name: str, url: str):
        self.key = key
        self.name = name
        self.url = url

    def push_code(self, account, script_path: str, checkpoint_dir: str = "./checkpoints") -> dict:
        script = Path(script_path)
        if not script.exists():
            return {"ok": False, "message": f"Script not found: {script_path}"}

        script_content = script.read_text()
        notebook = self._generate_notebook(script_content, checkpoint_dir, account.name)
        nb_path = Path(f"{self.key}_{account.name}_training.ipynb")
        nb_path.write_text(json.dumps(notebook, indent=2))

        return {
            "ok": True,
            "message": f"Notebook: {nb_path} — upload to {self.name}",
            "notebook_path": str(nb_path),
            "manual": True,
            "url": self.url,
        }

    def _generate_notebook(self, script_content: str, checkpoint_dir: str, account_name: str) -> dict:
        gpu_setup = {
            "sagemaker": "# SageMaker Studio Lab: Enable GPU in Runtime settings\n",
            "paperspace": "# Paperspace Gradient: GPU is auto-configured\n",
            "deepnote": "# Deepnote: Enable GPU in Environment settings\n",
            "lightning_ai": "# Lightning AI: GPU Studios auto-configured\n",
            "codesphere": "# Codesphere: Enable shared GPU in settings\n",
            "intel_devcloud": "# Intel DevCloud: oneAPI environment\nsource /opt/intel/oneapi/setvars.sh 2>/dev/null || true\n",
            "nvidia_vgpu": "# NVIDIA vGPU: GPU passthrough configured\n",
        }.get(self.key, "")

        cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# Free GPU Trainer — {self.name} ({account_name})\n", "# Auto-generated — run all cells"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    gpu_setup,
                    "!pip install -q torch torchvision transformers accelerate peft datasets 2>/dev/null\n",
                    "import torch\n",
                    f"print(f'GPU: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}}')\n",
                    "print(f'CUDA: {torch.cuda.is_available()}')\n",
                ],
                "execution_count": None,
                "outputs": [],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Training Script\n", script_content],
                "execution_count": None,
                "outputs": [],
            },
        ]
        return {
            "nbformat": 4, "nbformat_minor": 0,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "GPU"},
            "cells": cells,
        }

    def start_session(self, account) -> dict:
        return self.push_code(account, "train.py")

    def check_status(self, account) -> dict:
        return {"ok": True, "status": "unknown", "message": f"{self.name} has no status API — check browser"}

    def stop_session(self, account) -> dict:
        return {"ok": True, "message": f"Stop {self.name} manually in browser"}


# ── Handler Registry ───────────────────────────────────────────────

HANDLERS: dict[str, PlatformHandler] = {
    "google_colab": GoogleColabHandler(),
    "kaggle": KaggleHandler(),
    "huggingface": HuggingFaceHandler(),
    "oracle_cloud": OracleCloudHandler(),
    "gcp": SSHHandler("gcp", "Google Cloud Platform"),
    "paperspace": NotebookHandler("paperspace", "Paperspace Gradient", "https://gradient.paperspace.com/"),
    "sagemaker": NotebookHandler("sagemaker", "Amazon SageMaker Studio Lab", "https://studiolab.sagemaker.aws/"),
    "lightning_ai": NotebookHandler("lightning_ai", "Lightning AI", "https://lightning.ai/"),
    "codesphere": NotebookHandler("codesphere", "Codesphere", "https://codesphere.com/"),
    "intel_devcloud": NotebookHandler("intel_devcloud", "Intel Developer Cloud", "https://devcloud.intel.com/"),
    "deepnote": NotebookHandler("deepnote", "Deepnote", "https://deepnote.com/"),
    "nvidia_vgpu": NotebookHandler("nvidia_vgpu", "NVIDIA vGPU Trial", "https://www.nvidia.com/"),
}


def get_handler(platform_key: str) -> Optional[PlatformHandler]:
    """Get the handler for a platform."""
    return HANDLERS.get(platform_key)
