from typing import Dict, Type

from .base import BaseProvider
from .fal_ai import FalAIProvider
from .replicate import ReplicateProvider
from .sambanova import SambanovaProvider
from .together import TogetherProvider


PROVIDERS: Dict[str, Type[BaseProvider]] = {
    "fal-ai": FalAIProvider,
    "together": TogetherProvider,
    "sambanova": SambanovaProvider,
    "replicate": ReplicateProvider,
}


def get_provider(name: str) -> BaseProvider:
    """Get provider instance by name."""
    if name not in PROVIDERS:
        raise ValueError(f"provider: {name} not supported, available providers: {list(PROVIDERS.keys())}")
    return PROVIDERS[name]()
