from typing import Any, Dict, List, Optional

from huggingface_hub.constants import ENDPOINT
from huggingface_hub.utils import (
    build_hf_headers,
    get_session,
    hf_raise_for_status,
)


# Will be globally fetched only once (see '_fetch_recommended_models')
_RECOMMENDED_MODELS: Optional[Dict[str, Optional[str]]] = None


def _first_or_none(items: List[Any]) -> Optional[Any]:
    try:
        return items[0] or None
    except IndexError:
        return None


def _fetch_recommended_models() -> Dict[str, Optional[str]]:
    global _RECOMMENDED_MODELS
    if _RECOMMENDED_MODELS is None:
        response = get_session().get(f"{ENDPOINT}/api/tasks", headers=build_hf_headers())
        hf_raise_for_status(response)
        _RECOMMENDED_MODELS = {
            task: _first_or_none(details["widgetModels"]) for task, details in response.json().items()
        }
    return _RECOMMENDED_MODELS


def get_recommended_model(task: str) -> str:
    """
    Get the model Hugging Face recommends for the input task.

    Args:
        task (`str`):
            The Hugging Face task to get which model Hugging Face recommends.
            All available tasks can be found [here](https://huggingface.co/tasks).

    Returns:
        `str`: Name of the model recommended for the input task.

    Raises:
        `ValueError`: If Hugging Face has no recommendation for the input task.
    """
    model = _fetch_recommended_models().get(task)
    if model is None:
        raise ValueError(
            f"Task {task} has no recommended model. Please specify a model"
            " explicitly. Visit https://huggingface.co/tasks for more info."
        )
    return model
