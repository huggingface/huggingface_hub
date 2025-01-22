# mypy: disable-error-code="dict-item"
from typing import Any, Dict, Optional, Protocol, Union

from . import fal_ai, replicate, sambanova, together
from .hf_inference import HFInferenceBinaryInputTask, HFInferenceConversational, HFInferenceTask


class TaskProviderHelper(Protocol):
    """Protocol defining the interface for task-specific provider helpers."""

    def build_url(self, model: Optional[str] = None) -> str: ...
    def map_model(self, model: Optional[str] = None) -> str: ...
    def prepare_headers(self, headers: Dict, *, token: Optional[str] = None) -> Dict: ...
    def prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]: ...
    def get_response(self, response: Union[bytes, Dict]) -> Any: ...


PROVIDERS: Dict[str, Dict[str, TaskProviderHelper]] = {
    "replicate": {
        "text-to-image": replicate.text_to_image,
    },
    "fal-ai": {
        "text-to-image": fal_ai.text_to_image,
        "automatic-speech-recognition": fal_ai.automatic_speech_recognition,
    },
    "sambanova": {
        "conversational": sambanova.conversational,
    },
    "together": {
        "text-to-image": together.text_to_image,
        "conversational": together.conversational,
        "text-generation": together.text_generation,
    },
    "hf-inference": {
        "text-to-image": HFInferenceTask("text-to-image"),
        "conversational": HFInferenceConversational(),
        "text-classification": HFInferenceTask("text-classification"),
        "question-answering": HFInferenceTask("question-answering"),
        "audio-classification": HFInferenceBinaryInputTask("audio-classification"),
        "automatic-speech-recognition": HFInferenceBinaryInputTask("automatic-speech-recognition"),
        "fill-mask": HFInferenceTask("fill-mask"),
        "feature-extraction": HFInferenceTask("feature-extraction"),
        "image-classification": HFInferenceBinaryInputTask("image-classification"),
        "image-segmentation": HFInferenceBinaryInputTask("image-segmentation"),
        "document-question-answering": HFInferenceTask("document-question-answering"),
        "image-to-text": HFInferenceTask("image-to-text"),
        "object-detection": HFInferenceBinaryInputTask("object-detection"),
        "audio-to-audio": HFInferenceTask("audio-to-audio"),
        "zero-shot-image-classification": HFInferenceBinaryInputTask("zero-shot-image-classification"),
        "zero-shot-classification": HFInferenceTask("zero-shot-classification"),
        "image-to-image": HFInferenceBinaryInputTask("image-to-image"),
        "sentence-similarity": HFInferenceTask("sentence-similarity"),
        "table-question-answering": HFInferenceTask("table-question-answering"),
        "tabular-classification": HFInferenceTask("tabular-classification"),
        "text-to-speech": HFInferenceTask("text-to-speech"),
        "token-classification": HFInferenceTask("token-classification"),
        "translation": HFInferenceTask("translation"),
        "summarization": HFInferenceTask("summarization"),
        "visual-question-answering": HFInferenceBinaryInputTask("visual-question-answering"),
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
