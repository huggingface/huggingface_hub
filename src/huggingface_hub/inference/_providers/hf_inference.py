import logging
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

from huggingface_hub.constants import ENDPOINT
from huggingface_hub.inference._common import _b64_encode, _open_as_binary
from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status


# TYPES
UrlT = str
PathT = Union[str, Path]
BinaryT = Union[bytes, BinaryIO]
ContentT = Union[BinaryT, PathT, UrlT]

# Use to set a Accept: image/png header
TASKS_EXPECTING_IMAGES = {"text-to-image", "image-to-image"}

logger = logging.getLogger(__name__)


## RECOMMENDED MODELS

# Will be globally fetched only once (see '_fetch_recommended_models')
_RECOMMENDED_MODELS: Optional[Dict[str, Optional[str]]] = None

BASE_URL = "https://api-inference.huggingface.co"


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


class HFInferenceTask:
    """Base class for HF Inference API tasks."""

    def __init__(self, task: str):
        self.task = task

    def build_url(self, model: Optional[str] = None) -> str:
        if model is None:
            model = get_recommended_model(self.task)
        return f"{BASE_URL}/models/{model}"

    def map_model(self, model: str) -> str:
        return model

    def prepare_headers(self, headers: Dict, *, token: Optional[str] = None) -> Dict:
        return headers

    def prepare_payload(self, inputs: Any, parameters: Dict[str, Any], model: Optional[str]) -> Dict[str, Any]:
        if isinstance(inputs, (bytes, Path)):
            raise ValueError(f"Unexpected binary inputs. Got {inputs}")  # type: ignore

        return {
            "json": {
                inputs: inputs,
                parameters: {k: v for k, v in parameters.items() if v is not None},
            }
        }

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        return response


class HFInferenceBinaryInputTask(HFInferenceTask):
    def prepare_payload(self, inputs: Any, parameters: Dict[str, Any], model: Optional[str]) -> Dict[str, Any]:
        parameters = {k: v for k, v in parameters.items() if v is not None}
        has_parameters = len(parameters) > 0

        # Raise if not a binary object or a local path or a URL.
        if not isinstance(inputs, (bytes, Path)) and not isinstance(inputs, str):
            raise ValueError(f"Expected binary inputs or a local path or a URL. Got {inputs}")

        # Send inputs as raw content when no parameters are provided
        if not has_parameters:
            with _open_as_binary(inputs) as data:
                data_as_bytes = data if isinstance(data, bytes) else data.read()
                return {"data": data_as_bytes}

        # Otherwise encode as b64
        return {"json": {"inputs": _b64_encode(inputs), "parameters": parameters}}


class HFInferenceConversational(HFInferenceTask):
    def __init__(self):
        super().__init__("conversational")

    def build_url(self, model: Optional[str] = None) -> str:
        if model is None:
            model = get_recommended_model("text-generation")
        return f"{BASE_URL}/models/{model}/v1/chat/completions"

    def prepare_payload(self, inputs: Any, parameters: Dict[str, Any], model: Optional[str]) -> Dict[str, Any]:
        parameters = {key: value for key, value in parameters.items() if value is not None}
        return {"model": model, "messages": inputs, **parameters}
