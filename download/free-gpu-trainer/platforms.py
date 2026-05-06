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


def build_platform(key: str, cfg: dict) -> PlatformConfig:
    """Build a PlatformConfig from YAML config + platform definition."""
    defn = PLATFORM_DEFS[key]
    accounts = []
    for ac in cfg.get("accounts", []):
        accounts.append(AccountConfig(
            name=ac["name"],
            token=ac.get("token"),
            api_key=ac.get("api_key"),
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
