"""Provider adapters for FamilyGPU Orchestrator.

All providers implement a unified interface defined in base.py.
Each adapter handles the specifics of interacting with its platform.
"""

from providers.base import ProviderAdapter, ProviderResult
from providers.registry import get_adapter, list_adapters

__all__ = [
    "ProviderAdapter",
    "ProviderResult",
    "get_adapter",
    "list_adapters",
]
