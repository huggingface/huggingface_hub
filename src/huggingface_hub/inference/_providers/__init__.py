from typing import Any, Dict, Optional, Protocol, Union

from .fal_ai import text_to_image as falai_text_to_image
from .hf_inference import conversational as hf_inference_conversational
from .hf_inference import text_to_image as hf_inference_text_to_image
from .replicate import text_to_image as replicate_text_to_image
from .sambanova import conversational as sambanova_conversational
from .together import conversational as together_conversational
from .together import text_generation as together_text_generation
from .together import text_to_image as together_text_to_image


class TaskProviderHelper(Protocol):
    """Protocol defining the interface for task-specific provider helpers."""

    def build_url(self, model: Optional[str] = None) -> str: ...
    def map_model(self, model: Optional[str] = None) -> str: ...
    def prepare_headers(self, headers: Dict, *, token: Optional[str] = None) -> Dict: ...
    def prepare_payload(
        self, inputs: Any, parameters: Dict[str, Any], model: Optional[str] = None
    ) -> Dict[str, Any]: ...
    def get_response(self, response: Union[bytes, Dict]) -> Any: ...


PROVIDERS: Dict[str, Dict[str, TaskProviderHelper]] = {
    "replicate": {
        "text-to-image": replicate_text_to_image,  # type: ignore
    },
    "fal-ai": {
        "text-to-image": falai_text_to_image,  # type: ignore
        # TODO: add automatic-speech-recognition
    },
    "sambanova": {
        "conversational": sambanova_conversational,  # type: ignore
    },
    "together": {
        "text-to-image": together_text_to_image,  # type: ignore
        "conversational": together_conversational,  # type: ignore
        "text-generation": together_text_generation,  # type: ignore
    },
    "hf-inference": {  # TODO: add other tasks
        "text-to-image": hf_inference_text_to_image,  # type: ignore
        "conversational": hf_inference_conversational,  # type: ignore
    },
}


def get_provider_helper(provider: str, task: str) -> TaskProviderHelper:
    """Get provider helper instance by name and task.

    Args:
        provider (str): Name of the provider
        task (str): Name of the task

    Returns:
        TaskProviderHelper: Helper instance for the specified provider and task

    Raises:
        ValueError: If provider or task is not supported
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Provider '{provider}' not supported. Available providers: {list(PROVIDERS.keys())}")
    if task not in PROVIDERS[provider]:
        raise ValueError(
            f"Task '{task}' not supported for provider '{provider}'. "
            f"Available tasks: {list(PROVIDERS[provider].keys())}"
        )
    return PROVIDERS[provider][task]
