"""Provider adapter registry.

Maintains a mapping of provider_key -> ProviderAdapter instances.
Adapters are registered on import and looked up by the scheduler.
"""

import logging
from typing import Optional
from providers.base import ProviderAdapter

logger = logging.getLogger("fgt.providers.registry")

# Global registry
_registry: dict[str, ProviderAdapter] = {}


def register(adapter: ProviderAdapter):
    """Register a provider adapter."""
    _registry[adapter.provider_key] = adapter
    logger.debug(f"Registered provider adapter: {adapter.provider_key}")


def get_adapter(provider_key: str) -> Optional[ProviderAdapter]:
    """Get a provider adapter by key."""
    return _registry.get(provider_key)


def list_adapters() -> dict[str, ProviderAdapter]:
    """Get all registered adapters."""
    return dict(_registry)


def list_by_class(provider_class: str) -> list[ProviderAdapter]:
    """Get adapters filtered by provider class (A, B, or C)."""
    return [a for a in _registry.values() if a.provider_class == provider_class]


def list_auto_adapters() -> list[ProviderAdapter]:
    """Get adapters that support fully automated execution."""
    return [a for a in _registry.values() if a.supports_auto]


def list_by_automation_level(level: str) -> list[ProviderAdapter]:
    """Get adapters filtered by automation level."""
    return [a for a in _registry.values() if a.automation_level == level]


# ── Auto-register all adapters on import ──────────────────────────

def _auto_register():
    """Import and register all built-in adapters."""
    try:
        from providers.google_colab import GoogleColabAdapter
        register(GoogleColabAdapter())
    except Exception as e:
        logger.debug(f"Failed to register GoogleColabAdapter: {e}")

    try:
        from providers.kaggle import KaggleAdapter
        register(KaggleAdapter())
    except Exception as e:
        logger.debug(f"Failed to register KaggleAdapter: {e}")

    try:
        from providers.huggingface import HuggingFaceAdapter
        register(HuggingFaceAdapter())
    except Exception as e:
        logger.debug(f"Failed to register HuggingFaceAdapter: {e}")

    try:
        from providers.oracle_cloud import OracleCloudAdapter
        register(OracleCloudAdapter())
    except Exception as e:
        logger.debug(f"Failed to register OracleCloudAdapter: {e}")

    try:
        from providers.gcp import GCPAdapter
        register(GCPAdapter())
    except Exception as e:
        logger.debug(f"Failed to register GCPAdapter: {e}")

    try:
        from providers.paperspace import PaperspaceAdapter
        register(PaperspaceAdapter())
    except Exception as e:
        logger.debug(f"Failed to register PaperspaceAdapter: {e}")

    try:
        from providers.sagemaker import SageMakerAdapter
        register(SageMakerAdapter())
    except Exception as e:
        logger.debug(f"Failed to register SageMakerAdapter: {e}")

    try:
        from providers.lightning_ai import LightningAIAdapter
        register(LightningAIAdapter())
    except Exception as e:
        logger.debug(f"Failed to register LightningAIAdapter: {e}")

    try:
        from providers.codesphere import CodesphereAdapter
        register(CodesphereAdapter())
    except Exception as e:
        logger.debug(f"Failed to register CodesphereAdapter: {e}")

    try:
        from providers.intel_devcloud import IntelDevcloudAdapter
        register(IntelDevcloudAdapter())
    except Exception as e:
        logger.debug(f"Failed to register IntelDevcloudAdapter: {e}")

    try:
        from providers.deepnote import DeepnoteAdapter
        register(DeepnoteAdapter())
    except Exception as e:
        logger.debug(f"Failed to register DeepnoteAdapter: {e}")

    try:
        from providers.nvidia_vgpu import NvidiaVgpuAdapter
        register(NvidiaVgpuAdapter())
    except Exception as e:
        logger.debug(f"Failed to register NvidiaVgpuAdapter: {e}")

    logger.info(f"Registered {len(_registry)} provider adapters")


# Auto-register on first import
_auto_register()
