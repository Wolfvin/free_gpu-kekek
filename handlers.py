"""Real platform handlers — actual API integrations for free GPU platforms.

Each handler implements:
  - push_code(account, script_path, checkpoint_dir)  → push training code to platform
  - start_session(account, entry_script)              → start a notebook/runtime session
  - check_status(account)                             → check if session is still running
  - stop_session(account)                             → stop a running session
  - get_gpu_info(account)                             → get GPU type and availability
  - is_available(account)                             → check if platform is accessible right now

All methods return dicts with 'ok', 'message', and optional data.
Credentials are read from account.credentials dict (set via /add in TUI).
"""

import os
import re
import shlex
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod
from functools import wraps

logger = logging.getLogger("fgt.handler")


# ── Credential Helper ─────────────────────────────────────────────

def _cred(account, key: str, env_var: str = "", default: str = "") -> str:
    """Get a credential value: first from account.credentials, then env var, then default."""
    val = account.credentials.get(key, "")
    if val:
        return val
    if env_var:
        val = os.environ.get(env_var, "")
        if val:
            return val
    return default


# ── Input Validation ──────────────────────────────────────────────

# Valid hostname: alphanumeric + dots + hyphens, no leading hyphen, no spaces
_RE_HOSTNAME = re.compile(r'^[a-zA-Z0-9]([a-zA-Z0-9.\-]*[a-zA-Z0-9])?$')
# Valid IP address (v4)
_RE_IPV4 = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
# Valid SSH username: alphanumeric + underscore + hyphen
_RE_SSH_USER = re.compile(r'^[a-zA-Z0-9_][a-zA-Z0-9_.\-]*$')
# Valid account name: alphanumeric + hyphen + underscore
_RE_ACCOUNT_NAME = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_\-]*$')


def validate_hostname(host: str) -> bool:
    """Validate a hostname or IP address."""
    if not host:
        return False
    if _RE_IPV4.match(host):
        return True
    if _RE_HOSTNAME.match(host) and len(host) <= 253:
        return True
    return False


def validate_ssh_user(user: str) -> bool:
    """Validate an SSH username."""
    return bool(_RE_SSH_USER.match(user)) and len(user) <= 64


def validate_account_name(name: str) -> tuple[bool, str]:
    """Validate an account name. Returns (is_valid, error_message)."""
    if not name:
        return False, "Account name cannot be empty"
    if len(name) > 64:
        return False, "Account name too long (max 64 characters)"
    if not _RE_ACCOUNT_NAME.match(name):
        return False, "Account name can only contain letters, numbers, hyphens, and underscores (must start with alphanumeric)"
    return True, ""


# ── Retry Logic ───────────────────────────────────────────────────

def with_retry(max_retries: int = 3, base_delay: float = 1.0, backoff: float = 2.0):
    """Decorator for API calls with exponential backoff retry.

    Retries on:
      - subprocess.TimeoutExpired
      - ConnectionError / ConnectionRefusedError
      - Any exception with 'rate' or 'timeout' in the message

    Does NOT retry on:
      - FileNotFoundError (missing binary)
      - Authentication failures
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except subprocess.TimeoutExpired as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (backoff ** attempt)
                        logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}, retrying in {delay:.1f}s: {e}")
                        time.sleep(delay)
                except (ConnectionError, ConnectionRefusedError, OSError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (backoff ** attempt)
                        logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries}, retrying in {delay:.1f}s: {e}")
                        time.sleep(delay)
                except Exception as e:
                    msg = str(e).lower()
                    if any(kw in msg for kw in ['rate', 'timeout', '429', '503', 'temporarily']):
                        last_error = e
                        if attempt < max_retries - 1:
                            delay = base_delay * (backoff ** attempt)
                            logger.warning(f"Retryable error on attempt {attempt + 1}/{max_retries}, retrying in {delay:.1f}s: {e}")
                            time.sleep(delay)
                    else:
                        raise  # Non-retryable error
            # All retries exhausted
            if last_error:
                raise last_error
        return wrapper
    return decorator


# ── SSH Safety Helper ─────────────────────────────────────────────

def _safe_ssh_args(host: str, user: str, key_file: str) -> Optional[tuple]:
    """Validate and build safe SSH arguments.

    Returns (safe_host, safe_user, expanded_key) or None if invalid.
    """
    if not validate_hostname(host):
        logger.error(f"Invalid SSH host: {host!r}")
        return None
    if not validate_ssh_user(user):
        logger.error(f"Invalid SSH user: {user!r}")
        return None
    expanded_key = os.path.expanduser(key_file)
    if not os.path.exists(expanded_key):
        logger.warning(f"SSH key not found: {expanded_key}")
    return (host, user, expanded_key)


# ── Platform Type Classification ──────────────────────────────────

# Platforms with real API/SSH push — can auto-confirm sessions
AUTO_PLATFORMS = {"kaggle", "oracle_cloud", "gcp"}

# Platforms that are notebook-based — always require manual upload + /confirm
MANUAL_PLATFORMS = {
    "google_colab", "paperspace", "sagemaker", "lightning_ai",
    "codesphere", "intel_devcloud", "deepnote", "nvidia_vgpu",
}

# HuggingFace Spaces — has API push but NOT suitable for long training
# (ZeroGPU is for inference/demos, not long-running training)
HF_PLATFORMS = {"huggingface"}


def is_auto_platform(platform_key: str) -> bool:
    """Check if a platform supports fully automated push + start."""
    return platform_key in AUTO_PLATFORMS


def is_manual_platform(platform_key: str) -> bool:
    """Check if a platform requires manual upload + /confirm."""
    return platform_key in MANUAL_PLATFORMS or platform_key in HF_PLATFORMS


def platform_type_label(platform_key: str) -> str:
    """Get a human-readable label for the platform automation type."""
    if platform_key in AUTO_PLATFORMS:
        return "AUTO"
    if platform_key in HF_PLATFORMS:
        return "MANUAL (deployment only)"
    return "MANUAL"


# ── Base Handler ───────────────────────────────────────────────────

class PlatformHandler(ABC):
    """Base class for all platform handlers."""

    key: str = ""
    name: str = ""

    @abstractmethod
    def push_code(self, account, script_path: str, checkpoint_dir: str = "./checkpoints") -> dict:
        pass

    @abstractmethod
    def start_session(self, account, entry_script: str = "train.py") -> dict:
        pass

    @abstractmethod
    def check_status(self, account, entry_script: str = "train.py") -> dict:
        pass

    @abstractmethod
    def stop_session(self, account, entry_script: str = "train.py") -> dict:
        pass

    def is_available(self, account) -> dict:
        return {"ok": True, "message": "Unknown", "available": None}

    def get_gpu_info(self, account) -> dict:
        return {"ok": True, "gpu": "Unknown"}


# ── Kaggle Handler ─────────────────────────────────────────────────

class KaggleHandler(PlatformHandler):
    """Kaggle Notebooks API handler.

    Auth: account.credentials = {"kaggle_username": "...", "kaggle_key": "..."}
    Or set KAGGLE_USERNAME and KAGGLE_KEY env vars.

    This is the most fully-automated handler: push, status check, and
    checkpoint upload all work via the Kaggle API.
    """

    key = "kaggle"
    name = "Kaggle Notebooks"

    def _get_client(self, account):
        """Get authenticated Kaggle API client."""
        try:
            username = _cred(account, "kaggle_username", "KAGGLE_USERNAME")
            key = _cred(account, "kaggle_key", "KAGGLE_KEY")

            if username:
                os.environ["KAGGLE_USERNAME"] = username
            if key:
                os.environ["KAGGLE_KEY"] = key

            has_creds = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
            if not has_creds:
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

    def _upload_checkpoints_as_dataset(self, account, checkpoint_dir: str) -> dict:
        """Upload checkpoint folder as a Kaggle dataset for resume support.

        This enables cross-session resume: when a new kernel starts,
        it can pull the dataset to get the latest checkpoints.
        """
        ckpt_path = Path(checkpoint_dir)
        if not ckpt_path.exists() or not any(ckpt_path.iterdir()):
            return {"ok": True, "message": "No checkpoints to upload"}

        username = _cred(account, "kaggle_username", "KAGGLE_USERNAME")
        if not username:
            return {"ok": False, "message": "KAGGLE_USERNAME not set for dataset upload"}

        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '-', account.name)
        dataset_slug = f"{username}/{safe_name}-checkpoints"

        # Create dataset metadata
        meta = {
            "title": f"{safe_name}-checkpoints",
            "id": dataset_slug,
            "licenses": [{"name": "CC0-1.0"}],
        }
        meta_path = ckpt_path / "dataset-metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        try:
            result = subprocess.run(
                ["kaggle", "datasets", "create", "-p", str(ckpt_path), "--dir-mode", "zip"],
                capture_output=True, text=True, timeout=60,
                env={**os.environ, "KAGGLE_USERNAME": _cred(account, "kaggle_username", "KAGGLE_USERNAME"),
                     "KAGGLE_KEY": _cred(account, "kaggle_key", "KAGGLE_KEY")},
            )
            if result.returncode == 0:
                return {"ok": True, "message": f"Checkpoints uploaded as dataset: {dataset_slug}"}
            # Dataset may already exist — try to update
            result2 = subprocess.run(
                ["kaggle", "datasets", "version", "-p", str(ckpt_path), "-m", "Updated checkpoints",
                 "--dir-mode", "zip"],
                capture_output=True, text=True, timeout=60,
                env={**os.environ, "KAGGLE_USERNAME": _cred(account, "kaggle_username", "KAGGLE_USERNAME"),
                     "KAGGLE_KEY": _cred(account, "kaggle_key", "KAGGLE_KEY")},
            )
            if result2.returncode == 0:
                return {"ok": True, "message": f"Checkpoints dataset updated: {dataset_slug}"}
            return {"ok": False, "message": f"Dataset upload failed: {result2.stderr.strip()}"}
        except FileNotFoundError:
            return {"ok": False, "message": "kaggle CLI not found for dataset upload"}
        except Exception as e:
            return {"ok": False, "message": f"Dataset upload error: {e}"}

    @with_retry(max_retries=2, base_delay=2.0)
    def push_code(self, account, script_path: str, checkpoint_dir: str = "./checkpoints") -> dict:
        api = self._get_client(account)
        if not api:
            return {"ok": False, "message": "Kaggle auth failed. Set credentials via /add or KAGGLE_USERNAME + KAGGLE_KEY env vars."}

        script = Path(script_path)
        if not script.exists():
            return {"ok": False, "message": f"Script not found: {script_path}"}

        username = _cred(account, "kaggle_username", "KAGGLE_USERNAME")
        if not username:
            return {"ok": False, "message": "KAGGLE_USERNAME not set — cannot determine kernel owner"}

        # Sanitize account name for kernel slug
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '-', account.name)

        # Include checkpoint dataset as a data source if it exists
        dataset_sources = []
        ckpt_path = Path(checkpoint_dir)
        if ckpt_path.exists() and any(ckpt_path.iterdir()):
            dataset_sources.append(f"{username}/{safe_name}-checkpoints")

        kernel_meta = {
            "id": f"{username}/{safe_name}-training",
            "title": f"{safe_name}-training",
            "code_file": str(script),
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "competition_sources": [],
            "dataset_sources": dataset_sources,
            "kernel_sources": [],
        }

        meta_path = script.parent / "kernel-metadata.json"
        with open(meta_path, "w") as f:
            json.dump(kernel_meta, f, indent=2)

        try:
            result = subprocess.run(
                ["kaggle", "kernels", "push", "-p", str(script.parent)],
                capture_output=True, text=True, timeout=60,
                env={**os.environ, "KAGGLE_USERNAME": _cred(account, "kaggle_username", "KAGGLE_USERNAME"),
                     "KAGGLE_KEY": _cred(account, "kaggle_key", "KAGGLE_KEY")},
            )
            if result.returncode == 0:
                # Upload checkpoints as dataset for resume support
                ckpt_result = self._upload_checkpoints_as_dataset(account, checkpoint_dir)
                ckpt_msg = ""
                if ckpt_result.get("ok") and "No checkpoints" not in ckpt_result.get("message", ""):
                    ckpt_msg = f" | Checkpoints: {ckpt_result['message']}"
                return {"ok": True, "message": f"Kernel pushed: {result.stdout.strip()}{ckpt_msg}"}
            else:
                return {"ok": False, "message": f"Push failed: {result.stderr.strip()}"}
        except FileNotFoundError:
            return {"ok": False, "message": "kaggle CLI not found. Install: pip install kaggle"}
        except Exception as e:
            return {"ok": False, "message": f"Push error: {e}"}

    def start_session(self, account, entry_script: str = "train.py") -> dict:
        """Start a Kaggle kernel by pushing code.

        Args:
            account: The Kaggle account to use
            entry_script: Path to the training script (NOT hardcoded anymore)
        """
        return self.push_code(account, entry_script)

    @with_retry(max_retries=2, base_delay=2.0)
    def check_status(self, account, entry_script: str = "train.py") -> dict:
        api = self._get_client(account)
        if not api:
            return {"ok": False, "message": "Auth failed", "status": "unknown"}

        username = _cred(account, "kaggle_username", "KAGGLE_USERNAME")
        if not username:
            return {"ok": False, "message": "KAGGLE_USERNAME not set", "status": "unknown"}

        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '-', account.name)
        slug = f"{safe_name}-training"
        try:
            result = subprocess.run(
                ["kaggle", "kernels", "status", f"{username}/{slug}"],
                capture_output=True, text=True, timeout=30,
            )
            status_text = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
            # Map Kaggle status strings to our status vocabulary
            status_lower = status_text.lower()
            if "complete" in status_lower:
                mapped = "complete"
            elif "running" in status_lower or "executing" in status_lower:
                mapped = "running"
            elif "error" in status_lower or "fail" in status_lower:
                mapped = "error"
            elif "cancel" in status_lower:
                mapped = "stopped"
            else:
                mapped = status_text
            return {"ok": True, "status": mapped, "message": status_text}
        except Exception as e:
            return {"ok": False, "message": str(e), "status": "error"}

    def stop_session(self, account, entry_script: str = "train.py") -> dict:
        return {"ok": True, "message": "Kaggle kernels auto-stop after session limit"}

    def is_available(self, account) -> dict:
        api = self._get_client(account)
        if api:
            return {"ok": True, "available": True, "message": "Kaggle API authenticated"}
        return {"ok": False, "available": False, "message": "Kaggle auth failed — set credentials via /add"}


# ── HuggingFace Handler ────────────────────────────────────────────

class HuggingFaceHandler(PlatformHandler):
    """HuggingFace Spaces handler.

    WARNING: HuggingFace Spaces (ZeroGPU) is designed for hosting ML demos
    and inference endpoints, NOT for long-running training jobs. The ZeroGPU
    quota is per-request (typically seconds to minutes), not per-session.
    The session_limit_hours of 4h in the platform config is misleading.

    Use this handler ONLY for:
    - Deploying a trained model as a demo
    - Running quick inference tests

    For actual training, use Kaggle or Oracle Cloud SSH instead.
    """

    key = "huggingface"
    name = "Hugging Face Spaces"

    def _get_client(self, account):
        try:
            from huggingface_hub import HfApi
            token = _cred(account, "hf_token", "HF_TOKEN")
            api = HfApi(token=token)
            return api
        except ImportError:
            return None
        except Exception as e:
            logger.warning(f"HF auth failed for {account.name}: {e}")
            return None

    @with_retry(max_retries=3, base_delay=2.0)
    def push_code(self, account, script_path: str, checkpoint_dir: str = "./checkpoints") -> dict:
        api = self._get_client(account)
        if not api:
            return {"ok": False, "message": "HF auth failed. Set HF_TOKEN via /add or env var."}

        script = Path(script_path)
        if not script.exists():
            return {"ok": False, "message": f"Script not found: {script_path}"}

        app_code = self._generate_gradio_app(script_path, checkpoint_dir)
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '-', account.name)
        repo_name = f"{safe_name}-trainer"
        username = None
        try:
            who = api.whoami()
            username = who.get("name", "user")
        except Exception:
            pass

        repo_id = f"{username}/{repo_name}" if username else repo_name
        token = _cred(account, "hf_token", "HF_TOKEN")

        try:
            try:
                from huggingface_hub import create_repo
                create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", token=token, exist_ok=True)
            except Exception:
                pass

            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(app_code)
                app_path = f.name

            api.upload_file(path_or_fileobj=app_path, path_in_repo="app.py", repo_id=repo_id, repo_type="space")
            api.upload_file(path_or_fileobj=str(script), path_in_repo=script.name, repo_id=repo_id, repo_type="space")

            os.unlink(app_path)

            return {
                "ok": True,
                "message": f"Pushed to Space: {repo_id}",
                "repo_id": repo_id,
                "warning": "ZeroGPU Spaces are for inference/demos, not long-running training. "
                           "Use Kaggle or Oracle SSH for actual training.",
            }
        except Exception as e:
            return {"ok": False, "message": f"Push error: {e}"}

    def _generate_gradio_app(self, script_path: str, checkpoint_dir: str) -> str:
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

    def start_session(self, account, entry_script: str = "train.py") -> dict:
        return self.push_code(account, entry_script)

    @with_retry(max_retries=2, base_delay=2.0)
    def check_status(self, account, entry_script: str = "train.py") -> dict:
        api = self._get_client(account)
        if not api:
            return {"ok": False, "message": "Auth failed", "status": "unknown"}

        try:
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '-', account.name)
            info = api.space_info(repo_id=f"{api.whoami().get('name', 'user')}/{safe_name}-trainer")
            runtime = info.runtime if hasattr(info, 'runtime') else None
            stage = runtime.stage if runtime else "unknown"
            return {"ok": True, "status": stage, "message": f"Space status: {stage}"}
        except Exception as e:
            return {"ok": False, "message": str(e), "status": "error"}

    @with_retry(max_retries=2, base_delay=1.0)
    def stop_session(self, account, entry_script: str = "train.py") -> dict:
        api = self._get_client(account)
        if not api:
            return {"ok": False, "message": "Auth failed"}
        try:
            from huggingface_hub import pause_space
            username = api.whoami().get("name", "user")
            token = _cred(account, "hf_token", "HF_TOKEN")
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '-', account.name)
            pause_space(f"{username}/{safe_name}-trainer", token=token)
            return {"ok": True, "message": "Space paused (GPU freed)"}
        except Exception as e:
            return {"ok": False, "message": f"Stop error: {e}"}

    def is_available(self, account) -> dict:
        api = self._get_client(account)
        if api:
            return {"ok": True, "available": True, "message": "HF API authenticated"}
        return {"ok": False, "available": False, "message": "HF auth failed — set credentials via /add"}


# ── Google Colab Handler ───────────────────────────────────────────

class GoogleColabHandler(PlatformHandler):
    """Google Colab handler.

    Colab has NO public API for creating/managing runtimes.
    Generates .ipynb notebooks for manual upload.

    IMPORTANT: This is a MANUAL platform. After /start, user must:
    1. Open the generated notebook in Colab
    2. Enable GPU (Runtime > Change runtime type > T4 GPU)
    3. Run all cells
    4. Run /confirm in the TUI to start the countdown timer

    Credentials: account.credentials = {"email": "..."}
    (Colab doesn't use API keys — email is for identification/rotation tracking)
    """

    key = "google_colab"
    name = "Google Colab"

    def _scan_for_secrets(self, content: str) -> list[str]:
        """Scan script content for common secret patterns.

        Returns a list of warning messages (empty = no secrets found).
        """
        patterns = {
            "API key": re.compile(r"api_key\s*=\s*['\"][a-zA-Z0-9_\-]{20,}['\"]", re.IGNORECASE),
            "Token/Bearer/Auth": re.compile(r"(token|bearer|auth)\s*[:=]\s*['\"][a-zA-Z0-9_\-]{20,}['\"]", re.IGNORECASE),
            "Password/Secret": re.compile(r"(password|passwd|secret)\s*[:=]\s*['\"].{8,}['\"]", re.IGNORECASE),
            "AWS Access Key": re.compile(r"AKIA[0-9A-Z]{16}"),
            "Private Key": re.compile(r"-----BEGIN (RSA |EC )?PRIVATE KEY-----"),
            "Env Secret (HF_TOKEN, OPENAI_API_KEY, etc.)": re.compile(
                r"(HF_TOKEN|OPENAI_API_KEY|KAGGLE_KEY|AWS_SECRET|GCP_KEY)\s*=",
                re.IGNORECASE,
            ),
        }
        warnings = []
        for label, pattern in patterns.items():
            if pattern.search(content):
                warnings.append(f"Potential {label} detected")
        return warnings

    def push_code(self, account, script_path: str, checkpoint_dir: str = "./checkpoints") -> dict:
        script = Path(script_path)
        if not script.exists():
            return {"ok": False, "message": f"Script not found: {script_path}"}

        script_content = script.read_text()
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '-', account.name)

        # Scan for secrets before embedding the script into the notebook
        secret_warnings = self._scan_for_secrets(script_content)
        secrets_warning = None
        if secret_warnings:
            logger.warning(
                "Secrets detected in %s: %s — script will NOT be embedded",
                script_path, "; ".join(secret_warnings),
            )
            secrets_warning = "⚠ Training script not embedded (potential secrets detected). Upload script manually."
            # Replace the script content with a safe loader stub so the notebook
            # references the script by filename instead of embedding source.
            script_name = script.name
            script_content = (
                f"# Script not embedded due to detected secrets.\n"
                f"# Upload {script_name} manually and run:\n"
                f"!python {script_name} --checkpoint-dir ./checkpoints"
            )

        notebook = self._generate_notebook(script_content, checkpoint_dir, safe_name, secrets_warning=secrets_warning)
        nb_path = Path(f"colab_{safe_name}_training.ipynb")
        nb_path.write_text(json.dumps(notebook, indent=2))

        try:
            result = subprocess.run(
                ["colab-cli", "upload", str(nb_path)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return {"ok": True, "message": f"Uploaded to Colab: {result.stdout.strip()}"}
        except FileNotFoundError:
            pass

        result = {
            "ok": True,
            "message": f"Notebook generated: {nb_path} — open in Colab and run",
            "notebook_path": str(nb_path),
            "manual": True,
            "url": "https://colab.research.google.com/",
        }
        if secrets_warning:
            result["secrets_warning"] = secrets_warning
        return result

    def _generate_notebook(self, script_content: str, checkpoint_dir: str, account_name: str, *, secrets_warning: str | None = None) -> dict:
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
        ]

        if secrets_warning:
            # Add a markdown warning cell and a safe !python command cell
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [secrets_warning],
            })
            cells.append({
                "cell_type": "code",
                "metadata": {"id": "training"},
                "source": [
                    "# Training Script (run via command — source not embedded)\n",
                    script_content,
                ],
                "execution_count": None,
                "outputs": [],
            })
        else:
            # Embed the full training script source
            cells.append({
                "cell_type": "code",
                "metadata": {"id": "training"},
                "source": [
                    "# Training Script\n",
                    script_content,
                ],
                "execution_count": None,
                "outputs": [],
            })

        return {
            "nbformat": 4, "nbformat_minor": 0,
            "metadata": {
                "colab": {"provenance": [], "gpuType": "T4"},
                "kernelspec": {"name": "python3", "display_name": "Python 3"},
                "accelerator": "GPU",
            },
            "cells": cells,
        }

    def start_session(self, account, entry_script: str = "train.py") -> dict:
        return self.push_code(account, entry_script)

    def check_status(self, account, entry_script: str = "train.py") -> dict:
        return {"ok": True, "status": "unknown", "message": "Colab has no status API — check browser"}

    def stop_session(self, account, entry_script: str = "train.py") -> dict:
        return {"ok": True, "message": "Stop Colab manually in browser (Runtime > Disconnect)"}

    def is_available(self, account) -> dict:
        return {"ok": True, "available": True, "message": "Colab notebooks can be created anytime"}


# ── Oracle Cloud Handler ───────────────────────────────────────────

class OracleCloudHandler(PlatformHandler):
    """Oracle Cloud Free Tier handler.

    Uses SSH to manage always-free Ampere A1 compute instances.
    This is a fully automated (AUTO) platform — push, start, status,
    and stop all work via SSH.

    Auth: account.credentials = {"oci_vm_host": "129.x.x.x", "oci_vm_user": "opc", "oci_ssh_key": "~/.ssh/id_rsa"}
    Or set OCI_VM_HOST, OCI_VM_USER, OCI_SSH_KEY env vars.
    """

    key = "oracle_cloud"
    name = "Oracle Cloud Free Tier"

    def _ssh_config(self, account) -> tuple:
        host = _cred(account, "oci_vm_host", "OCI_VM_HOST")
        user = _cred(account, "oci_vm_user", "OCI_VM_USER", "opc")
        key_file = _cred(account, "oci_ssh_key", "OCI_SSH_KEY", "~/.ssh/id_rsa")
        return host, user, key_file

    def _validated_ssh(self, account) -> Optional[tuple]:
        """Validate SSH args and return (host, user, expanded_key) or None."""
        host, user, key_file = self._ssh_config(account)
        if not host:
            return None
        safe = _safe_ssh_args(host, user, key_file)
        if not safe:
            return None
        return safe

    @with_retry(max_retries=2, base_delay=2.0)
    def push_code(self, account, script_path: str, checkpoint_dir: str = "./checkpoints") -> dict:
        host, user, key_file = self._ssh_config(account)
        if not host:
            return {"ok": True, "message": "Set OCI VM credentials via /add", "manual": True}

        safe = _safe_ssh_args(host, user, key_file)
        if not safe:
            return {"ok": False, "message": f"Invalid SSH host or username. Host: {host!r}, User: {user!r}"}

        safe_host, safe_user, expanded_key = safe
        script = Path(script_path)
        if not script.exists():
            return {"ok": False, "message": f"Script not found: {script_path}"}

        try:
            # Push script — preserve original filename on remote so start_session can find it
            remote_script_name = script.name  # e.g. "finetune.py", not always "train.py"
            result = subprocess.run(
                ["scp", "-i", expanded_key, str(script), f"{safe_user}@{safe_host}:~/{remote_script_name}"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return {"ok": False, "message": f"SCP failed: {result.stderr}"}

            # Push checkpoints if they exist (for resume support)
            ckpt_path = Path(checkpoint_dir)
            if ckpt_path.exists() and any(ckpt_path.iterdir()):
                ckpt_result = subprocess.run(
                    ["scp", "-i", expanded_key, "-r", str(ckpt_path), f"{safe_user}@{safe_host}:~/checkpoints"],
                    capture_output=True, text=True, timeout=30,
                )
                ckpt_msg = " | Checkpoints pushed" if ckpt_result.returncode == 0 else " | Checkpoint push failed"
            else:
                ckpt_msg = ""

            return {"ok": True, "message": f"Pushed to {safe_host}{ckpt_msg}"}
        except Exception as e:
            return {"ok": False, "message": f"Push error: {e}"}

    @with_retry(max_retries=2, base_delay=2.0)
    def start_session(self, account, entry_script: str = "train.py") -> dict:
        host, user, key_file = self._ssh_config(account)
        if not host:
            return {"ok": True, "message": "Set OCI VM credentials via /add", "manual": True}

        safe = _safe_ssh_args(host, user, key_file)
        if not safe:
            return {"ok": False, "message": f"Invalid SSH host or username. Host: {host!r}, User: {user!r}"}

        safe_host, safe_user, expanded_key = safe
        # Use the basename of entry_script so it matches the SCP'd file
        script_name = Path(entry_script).name
        quoted_name = shlex.quote(script_name)
        try:
            result = subprocess.run(
                ["ssh", "-i", expanded_key, f"{safe_user}@{safe_host}",
                 f"nohup python ~/{quoted_name} > ~/training.log 2>&1 & echo $! > ~/training.pid"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return {"ok": True, "message": f"Training started on {safe_host}"}
            return {"ok": False, "message": f"SSH failed: {result.stderr}"}
        except Exception as e:
            return {"ok": False, "message": f"Start error: {e}"}

    def check_status(self, account, entry_script: str = "train.py") -> dict:
        safe = self._validated_ssh(account)
        if not safe:
            return {"ok": True, "status": "unknown", "message": "No valid SSH config"}
        safe_host, safe_user, expanded_key = safe
        try:
            result = subprocess.run(
                ["ssh", "-i", expanded_key, f"{safe_user}@{safe_host}",
                 "if [ -f ~/training.pid ] && kill -0 $(cat ~/training.pid) 2>/dev/null; then echo 'running'; else echo 'not running'; fi"],
                capture_output=True, text=True, timeout=10,
            )
            running = "running" in result.stdout and "not running" not in result.stdout
            return {"ok": True, "status": "running" if running else "stopped", "message": result.stdout.strip()}
        except Exception as e:
            return {"ok": False, "message": str(e), "status": "error"}

    def stop_session(self, account, entry_script: str = "train.py") -> dict:
        safe = self._validated_ssh(account)
        if not safe:
            return {"ok": True, "message": "No valid SSH config — stop manually"}
        safe_host, safe_user, expanded_key = safe
        try:
            subprocess.run(
                ["ssh", "-i", expanded_key, f"{safe_user}@{safe_host}",
                 "kill $(cat ~/training.pid 2>/dev/null) 2>/dev/null; rm -f ~/training.pid"],
                capture_output=True, text=True, timeout=10,
            )
            return {"ok": True, "message": "Training process killed"}
        except Exception as e:
            return {"ok": False, "message": str(e)}


# ── GCP Handler ────────────────────────────────────────────────────

class GCPHandler(PlatformHandler):
    """Google Cloud Platform handler.

    Uses SSH to manage GCP VMs with free GPU credits.
    This is a fully automated (AUTO) platform.

    Auth: account.credentials = {"gcp_vm_host": "35.x.x.x", "gcp_vm_user": "ubuntu", "gcp_ssh_key": "~/.ssh/id_rsa"}
    Or set GCP_VM_HOST, GCP_VM_USER, GCP_SSH_KEY env vars.
    """

    key = "gcp"
    name = "Google Cloud Platform"

    def _ssh_config(self, account) -> tuple:
        host = _cred(account, "gcp_vm_host", "GCP_VM_HOST")
        user = _cred(account, "gcp_vm_user", "GCP_VM_USER", "ubuntu")
        key_file = _cred(account, "gcp_ssh_key", "GCP_SSH_KEY", "~/.ssh/id_rsa")
        return host, user, key_file

    def _validated_ssh(self, account) -> Optional[tuple]:
        host, user, key_file = self._ssh_config(account)
        if not host:
            return None
        safe = _safe_ssh_args(host, user, key_file)
        if not safe:
            return None
        return safe

    @with_retry(max_retries=2, base_delay=2.0)
    def push_code(self, account, script_path: str, checkpoint_dir: str = "./checkpoints") -> dict:
        host, user, key_file = self._ssh_config(account)
        if not host:
            return {"ok": True, "message": "Set GCP VM credentials via /add", "manual": True}

        safe = _safe_ssh_args(host, user, key_file)
        if not safe:
            return {"ok": False, "message": f"Invalid SSH host or username. Host: {host!r}, User: {user!r}"}

        safe_host, safe_user, expanded_key = safe
        script = Path(script_path)
        if not script.exists():
            return {"ok": False, "message": f"Script not found: {script_path}"}
        try:
            # Push script — preserve original filename on remote so start_session can find it
            remote_script_name = script.name  # e.g. "finetune.py", not always "train.py"
            result = subprocess.run(
                ["scp", "-i", expanded_key, str(script), f"{safe_user}@{safe_host}:~/{remote_script_name}"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return {"ok": False, "message": f"SCP failed: {result.stderr}"}

            # Push checkpoints for resume support
            ckpt_path = Path(checkpoint_dir)
            if ckpt_path.exists() and any(ckpt_path.iterdir()):
                ckpt_result = subprocess.run(
                    ["scp", "-i", expanded_key, "-r", str(ckpt_path), f"{safe_user}@{safe_host}:~/checkpoints"],
                    capture_output=True, text=True, timeout=30,
                )
                ckpt_msg = " | Checkpoints pushed" if ckpt_result.returncode == 0 else " | Checkpoint push failed"
            else:
                ckpt_msg = ""

            return {"ok": True, "message": f"Pushed to {safe_host}{ckpt_msg}"}
        except Exception as e:
            return {"ok": False, "message": str(e)}

    @with_retry(max_retries=2, base_delay=2.0)
    def start_session(self, account, entry_script: str = "train.py") -> dict:
        host, user, key_file = self._ssh_config(account)
        if not host:
            return {"ok": True, "message": "Set GCP VM credentials via /add", "manual": True}

        safe = _safe_ssh_args(host, user, key_file)
        if not safe:
            return {"ok": False, "message": f"Invalid SSH host or username. Host: {host!r}, User: {user!r}"}

        safe_host, safe_user, expanded_key = safe
        # Use the basename of entry_script so it matches the SCP'd file
        script_name = Path(entry_script).name
        quoted_name = shlex.quote(script_name)
        try:
            result = subprocess.run(
                ["ssh", "-i", expanded_key, f"{safe_user}@{safe_host}",
                 f"nohup python ~/{quoted_name} > ~/training.log 2>&1 & echo $! > ~/training.pid"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return {"ok": True, "message": f"Training started on {safe_host}"}
            return {"ok": False, "message": f"SSH failed: {result.stderr}"}
        except Exception as e:
            return {"ok": False, "message": str(e)}

    def check_status(self, account, entry_script: str = "train.py") -> dict:
        safe = self._validated_ssh(account)
        if not safe:
            return {"ok": True, "status": "unknown", "message": "No valid SSH config"}
        safe_host, safe_user, expanded_key = safe
        try:
            result = subprocess.run(
                ["ssh", "-i", expanded_key, f"{safe_user}@{safe_host}",
                 "if [ -f ~/training.pid ] && kill -0 $(cat ~/training.pid) 2>/dev/null; then echo 'running'; else echo 'not running'; fi"],
                capture_output=True, text=True, timeout=10,
            )
            running = "running" in result.stdout and "not running" not in result.stdout
            return {"ok": True, "status": "running" if running else "stopped", "message": result.stdout.strip()}
        except Exception as e:
            return {"ok": False, "message": str(e), "status": "error"}

    def stop_session(self, account, entry_script: str = "train.py") -> dict:
        safe = self._validated_ssh(account)
        if not safe:
            return {"ok": True, "message": "No valid SSH config — stop manually"}
        safe_host, safe_user, expanded_key = safe
        try:
            subprocess.run(
                ["ssh", "-i", expanded_key, f"{safe_user}@{safe_host}",
                 "kill $(cat ~/training.pid 2>/dev/null) 2>/dev/null; rm -f ~/training.pid"],
                capture_output=True, text=True, timeout=10,
            )
            return {"ok": True, "message": "Training process killed"}
        except Exception as e:
            return {"ok": False, "message": str(e)}


# ── Notebook Generator Handler (fallback for notebook platforms) ───

class NotebookHandler(PlatformHandler):
    """Handler for notebook-based platforms that don't have push APIs.

    Generates .ipynb files for manual upload.
    Works for: SageMaker, Paperspace, Deepnote, Lightning AI, Codesphere, Intel DevCloud, NVIDIA vGPU

    IMPORTANT: All these are MANUAL platforms. After /start, user must:
    1. Upload the generated notebook to the platform
    2. Run the notebook
    3. Run /confirm in the TUI to start the countdown timer
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
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '-', account.name)
        notebook = self._generate_notebook(script_content, checkpoint_dir, safe_name)
        nb_path = Path(f"{self.key}_{safe_name}_training.ipynb")
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

    def start_session(self, account, entry_script: str = "train.py") -> dict:
        return self.push_code(account, entry_script)

    def check_status(self, account, entry_script: str = "train.py") -> dict:
        return {"ok": True, "status": "unknown", "message": f"{self.name} has no status API — check browser"}

    def stop_session(self, account, entry_script: str = "train.py") -> dict:
        return {"ok": True, "message": f"Stop {self.name} manually in browser"}


# ── Handler Registry ───────────────────────────────────────────────

HANDLERS: dict[str, PlatformHandler] = {
    "google_colab": GoogleColabHandler(),
    "kaggle": KaggleHandler(),
    "huggingface": HuggingFaceHandler(),
    "oracle_cloud": OracleCloudHandler(),
    "gcp": GCPHandler(),
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
