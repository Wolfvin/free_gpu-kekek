"""Platform definitions and registry for free GPU platforms."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PlatformStatus(Enum):
    AVAILABLE = "available"
    IN_USE = "in_use"
    COOLDOWN = "cooldown"
    EXHAUSTED = "exhausted"
    DISABLED = "disabled"
    WEEKLY_LIMIT = "weekly_limit"
    TRIAL_EXPIRED = "trial_expired"


class GPURType(Enum):
    T4 = "T4"
    A100 = "A100"
    P100 = "Tesla P100"
    M4000 = "Quadro M4000"
    ZERO_GPU = "ZeroGPU"
    AMPERE_A1 = "Ampere A1"
    INTEL_MAX = "Intel Max/Flex"
    SHARED = "Shared GPU"
    VGPU = "vGPU"
    VARIES = "GPU (varies)"
    LIMITED = "Limited GPU"
    CREDITS = "GPU (via credits)"


@dataclass
class AccountConfig:
    """A single account on a platform."""
    name: str
    token: Optional[str] = None
    api_key: Optional[str] = None
    credentials: dict = field(default_factory=dict)  # Platform-specific creds
    # Runtime state
    status: PlatformStatus = PlatformStatus.AVAILABLE
    sessions_used: int = 0
    total_hours_used: float = 0.0
    weekly_hours_used: float = 0.0
    current_session_start: Optional[float] = None
    cooldown_until: Optional[float] = None


@dataclass
class PlatformConfig:
    """Configuration for a GPU platform."""
    key: str
    name: str
    url: str
    gpu_type: str
    session_limit_hours: float
    cooldown_minutes: float = 5.0
    weekly_limit_hours: Optional[float] = None
    monthly_limit_hours: Optional[float] = None
    trial_days: Optional[int] = None
    enabled: bool = True
    accounts: list[AccountConfig] = field(default_factory=list)
    best_for: str = ""

    @property
    def total_accounts(self) -> int:
        return len(self.accounts)

    @property
    def available_accounts(self) -> list[AccountConfig]:
        return [a for a in self.accounts if a.status == PlatformStatus.AVAILABLE]

    @property
    def active_account(self) -> Optional[AccountConfig]:
        for a in self.accounts:
            if a.status == PlatformStatus.IN_USE:
                return a
        return None

    @property
    def max_continuous_hours(self) -> float:
        """Total hours possible by stacking all accounts sequentially."""
        return self.session_limit_hours * len(self.accounts)

    @property
    def status(self) -> PlatformStatus:
        if not self.enabled:
            return PlatformStatus.DISABLED
        if any(a.status == PlatformStatus.IN_USE for a in self.accounts):
            return PlatformStatus.IN_USE
        if self.available_accounts:
            return PlatformStatus.AVAILABLE
        if all(a.status == PlatformStatus.WEEKLY_LIMIT for a in self.accounts):
            return PlatformStatus.WEEKLY_LIMIT
        if all(a.status == PlatformStatus.TRIAL_EXPIRED for a in self.accounts):
            return PlatformStatus.TRIAL_EXPIRED
        return PlatformStatus.EXHAUSTED


# ── Credential Schema per Platform ─────────────────────────────────
# Each entry: key -> { label, hint, secret (mask input), required }

CREDENTIAL_SCHEMAS: dict[str, list[dict]] = {
    "google_colab": [
        {
            "key": "email",
            "label": "Google Account Email",
            "hint": "your-email@gmail.com",
            "secret": False,
            "required": True,
        },
    ],
    "kaggle": [
        {
            "key": "kaggle_username",
            "label": "Kaggle Username",
            "hint": "https://www.kaggle.com/settings → API → Create New Token",
            "secret": False,
            "required": True,
        },
        {
            "key": "kaggle_key",
            "label": "Kaggle API Key",
            "hint": "From kaggle.json → key field",
            "secret": True,
            "required": True,
        },
    ],
    "huggingface": [
        {
            "key": "hf_token",
            "label": "HuggingFace Token",
            "hint": "https://huggingface.co/settings/tokens",
            "secret": True,
            "required": True,
        },
    ],
    "paperspace": [
        {
            "key": "paperspace_api_key",
            "label": "Paperspace API Key",
            "hint": "https://gradient.paperspace.com/profile/api-keys",
            "secret": True,
            "required": True,
        },
    ],
    "sagemaker": [
        {
            "key": "sagemaker_token",
            "label": "SageMaker Studio Lab Token",
            "hint": "Login token from Studio Lab",
            "secret": True,
            "required": True,
        },
    ],
    "lightning_ai": [
        {
            "key": "lightning_token",
            "label": "Lightning AI API Key",
            "hint": "https://lightning.ai/settings",
            "secret": True,
            "required": True,
        },
    ],
    "codesphere": [
        {
            "key": "codesphere_token",
            "label": "Codesphere API Token",
            "hint": "https://codesphere.com/settings",
            "secret": True,
            "required": True,
        },
    ],
    "oracle_cloud": [
        {
            "key": "oci_vm_host",
            "label": "VM IP Address",
            "hint": "e.g. 129.x.x.x",
            "secret": False,
            "required": True,
        },
        {
            "key": "oci_vm_user",
            "label": "SSH Username",
            "hint": "usually 'opc' for Oracle Linux, 'ubuntu' for Ubuntu",
            "secret": False,
            "required": True,
        },
        {
            "key": "oci_ssh_key",
            "label": "SSH Private Key Path",
            "hint": "~/.ssh/id_rsa",
            "secret": False,
            "required": True,
        },
    ],
    "gcp": [
        {
            "key": "gcp_vm_host",
            "label": "VM External IP",
            "hint": "e.g. 35.x.x.x",
            "secret": False,
            "required": True,
        },
        {
            "key": "gcp_vm_user",
            "label": "SSH Username",
            "hint": "usually your GCP username or 'ubuntu'",
            "secret": False,
            "required": True,
        },
        {
            "key": "gcp_ssh_key",
            "label": "SSH Private Key Path",
            "hint": "~/.ssh/id_rsa or ~/.ssh/google_compute_engine",
            "secret": False,
            "required": True,
        },
    ],
    "intel_devcloud": [
        {
            "key": "intel_token",
            "label": "Intel DevCloud Token",
            "hint": "https://devcloud.intel.com/ — get from profile",
            "secret": True,
            "required": True,
        },
    ],
    "deepnote": [
        {
            "key": "deepnote_token",
            "label": "Deepnote API Token",
            "hint": "https://deepnote.com/settings",
            "secret": True,
            "required": True,
        },
    ],
    "nvidia_vgpu": [
        {
            "key": "nvidia_license_key",
            "label": "NVIDIA vGPU License Key",
            "hint": "From evaluation email",
            "secret": True,
            "required": True,
        },
    ],
}


# ── Platform Registry ──────────────────────────────────────────────

PLATFORM_DEFS: dict[str, dict] = {
    "google_colab": {
        "name": "Google Colab",
        "url": "https://colab.research.google.com/",
        "gpu_type": "T4",
        "session_limit_hours": 12,
        "cooldown_minutes": 5,
        "best_for": "Quick experiments, notebooks, learning ML/DL",
    },
    "kaggle": {
        "name": "Kaggle Notebooks",
        "url": "https://www.kaggle.com/",
        "gpu_type": "Tesla P100",
        "session_limit_hours": 9,
        "weekly_limit_hours": 30,
        "cooldown_minutes": 5,
        "best_for": "Competitions, stable notebook experience",
    },
    "huggingface": {
        "name": "Hugging Face Spaces",
        "url": "https://huggingface.co/spaces",
        "gpu_type": "ZeroGPU",
        "session_limit_hours": 4,
        "cooldown_minutes": 10,
        "best_for": "Hosting ML demos, Gradio/Streamlit apps",
    },
    "paperspace": {
        "name": "Paperspace Gradient",
        "url": "https://gradient.paperspace.com/",
        "gpu_type": "Quadro M4000",
        "session_limit_hours": 6,
        "cooldown_minutes": 10,
        "best_for": "Jupyter notebook-style GPU work",
    },
    "sagemaker": {
        "name": "Amazon SageMaker Studio Lab",
        "url": "https://studiolab.sagemaker.aws/",
        "gpu_type": "T4",
        "session_limit_hours": 4,
        "cooldown_minutes": 10,
        "best_for": "Persistent ML projects",
    },
    "lightning_ai": {
        "name": "Lightning AI",
        "url": "https://lightning.ai/",
        "gpu_type": "GPU (varies)",
        "session_limit_hours": 4,
        "monthly_limit_hours": 80,
        "cooldown_minutes": 10,
        "best_for": "PyTorch Lightning workflows",
    },
    "codesphere": {
        "name": "Codesphere",
        "url": "https://codesphere.com/",
        "gpu_type": "Shared GPU",
        "session_limit_hours": 4,
        "cooldown_minutes": 15,
        "best_for": "Deploying AI models quickly",
    },
    "oracle_cloud": {
        "name": "Oracle Cloud Free Tier",
        "url": "https://cloud.oracle.com/free",
        "gpu_type": "Ampere A1",
        "session_limit_hours": 999,
        "cooldown_minutes": 0,
        "best_for": "Running LLMs, long-running tasks",
    },
    "gcp": {
        "name": "Google Cloud Platform",
        "url": "https://cloud.google.com/free",
        "gpu_type": "GPU (via credits)",
        "session_limit_hours": 8,
        "cooldown_minutes": 0,
        "best_for": "GCP ecosystem, short-term projects",
    },
    "intel_devcloud": {
        "name": "Intel Developer Cloud",
        "url": "https://devcloud.intel.com/",
        "gpu_type": "Intel Max/Flex",
        "session_limit_hours": 4,
        "cooldown_minutes": 15,
        "best_for": "Intel GPU development, oneAPI projects",
    },
    "deepnote": {
        "name": "Deepnote",
        "url": "https://deepnote.com/",
        "gpu_type": "Limited GPU",
        "session_limit_hours": 4,
        "cooldown_minutes": 15,
        "best_for": "Collaborative data science",
    },
    "nvidia_vgpu": {
        "name": "NVIDIA vGPU Trial",
        "url": "https://www.nvidia.com/en-us/data-center/resources/vgpu-evaluation/",
        "gpu_type": "vGPU (trial)",
        "session_limit_hours": 8,
        "trial_days": 90,
        "cooldown_minutes": 0,
        "best_for": "Enterprise GPU evaluation",
    },
}


def build_platform(key: str, cfg: dict, config_dir: str = ".") -> PlatformConfig:
    """Build a PlatformConfig from YAML config + platform definition.

    Credentials are decrypted from storage (keyring/Fernet) before use.
    """
    from vault import decrypt_credentials
    defn = PLATFORM_DEFS[key]
    accounts = []
    for ac in cfg.get("accounts", []):
        # Extract and decrypt credentials from stored form
        stored_creds = ac.get("credentials", {})
        if stored_creds:
            creds = decrypt_credentials(key, ac["name"], stored_creds, config_dir)
        else:
            creds = {}
        # Also support legacy token/api_key fields
        if ac.get("token") and "token" not in creds:
            creds["token"] = ac["token"]
        if ac.get("api_key") and "api_key" not in creds:
            creds["api_key"] = ac["api_key"]
        accounts.append(AccountConfig(
            name=ac["name"],
            token=ac.get("token"),
            api_key=ac.get("api_key"),
            credentials=creds,
        ))
    return PlatformConfig(
        key=key,
        name=defn["name"],
        url=defn["url"],
        gpu_type=cfg.get("gpu_type", defn["gpu_type"]),
        session_limit_hours=cfg.get("session_limit_hours", defn["session_limit_hours"]),
        cooldown_minutes=cfg.get("cooldown_minutes", defn["cooldown_minutes"]),
        weekly_limit_hours=cfg.get("weekly_limit_hours", defn.get("weekly_limit_hours")),
        monthly_limit_hours=cfg.get("monthly_limit_hours", defn.get("monthly_limit_hours")),
        trial_days=cfg.get("trial_days", defn.get("trial_days")),
        enabled=cfg.get("enabled", True),
        accounts=accounts,
        best_for=defn["best_for"],
    )
