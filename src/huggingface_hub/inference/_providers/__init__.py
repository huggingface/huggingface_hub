from typing import Any, Dict, Optional, Protocol, Union

from . import fal_ai, hf_inference, replicate, sambanova, together


class TaskProviderHelper(Protocol):
    """Protocol defining the interface for task-specific provider helpers."""

    def build_url(model: Optional[str] = None) -> str: ...
    def map_model(model: Optional[str] = None) -> str: ...
    def prepare_headers(headers: Dict, *, token: Optional[str] = None) -> Dict: ...
    def prepare_payload(
        inputs: Any, parameters: Dict[str, Any], model: Optional[str] = None, expect_binary: bool = False
    ) -> Dict[str, Any]: ...
    def get_response(response: Union[bytes, Dict]) -> Any: ...


PROVIDERS: Dict[str, Dict[str, TaskProviderHelper]] = {
    "replicate": {
        "text-to-image": replicate.text_to_image,  # type: ignore
    },
    "fal-ai": {
        "text-to-image": fal_ai.text_to_image,  # type: ignore
        # TODO: add automatic-speech-recognition
    },
    "sambanova": {
        "conversational": sambanova.conversational,  # type: ignore
    },
    "together": {
        "text-to-image": together.text_to_image,  # type: ignore
        "conversational": together.conversational,  # type: ignore
        "text-generation": together.text_generation,  # type: ignore
    },
    "hf-inference": {
        "text-to-image": hf_inference.text_to_image,
        "conversational": hf_inference.conversational,
        "text-classification": hf_inference.text_classification,
        "question-answering": hf_inference.question_answering,
        "audio-classification": hf_inference.audio_classification,
        "automatic-speech-recognition": hf_inference.automatic_speech_recognition,
        "fill-mask": hf_inference.fill_mask,
        "feature-extraction": hf_inference.feature_extraction,
        "image-classification": hf_inference.image_classification,
        "image-segmentation": hf_inference.image_segmentation,
        "document-question-answering": hf_inference.document_question_answering,
        "image-to-text": hf_inference.image_to_text,
        "object-detection": hf_inference.object_detection,
        "audio-to-audio": hf_inference.audio_to_audio,
        "zero-shot-image-classification": hf_inference.zero_shot_image_classification,
        "zero-shot-classification": hf_inference.zero_shot_classification,
        "image-to-image": hf_inference.image_to_image,
        "sentence-similarity": hf_inference.sentence_similarity,
        "table-question-answering": hf_inference.table_question_answering,
        "tabular-classification": hf_inference.tabular_classification,
        "text-to-speech": hf_inference.text_to_speech,
        "token-classification": hf_inference.token_classification,
        "translation": hf_inference.translation,
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
