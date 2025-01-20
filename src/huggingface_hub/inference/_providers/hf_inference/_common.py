import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Dict,
    List,
    Optional,
    Union,
)

from huggingface_hub.constants import ENDPOINT
from huggingface_hub.inference._common import _b64_encode
from huggingface_hub.utils import (
    build_hf_headers,
    get_session,
    hf_raise_for_status,
)


if TYPE_CHECKING:
    pass

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


class BaseInferenceTask:
    """Base class for HF Inference API tasks."""

    BASE_URL = "https://api-inference.huggingface.co"
    TASK_NAME: str = ""  # To be defined by subclasses

    @classmethod
    def build_url(cls, model: Optional[str] = None) -> str:
        if model is None:
            model = get_recommended_model(cls.TASK_NAME)
        return f"{cls.BASE_URL}/models/{model}"

    @staticmethod
    def map_model(model: str) -> str:
        return model

    @staticmethod
    def prepare_headers(headers: Dict, *, token: Optional[str] = None) -> Dict:
        return headers

    @classmethod
    def prepare_payload(
        cls,
        inputs: Any,
        parameters: Dict[str, Any],
        model: Optional[str] = None,
        *,
        expect_binary: bool = False,
    ) -> Dict[str, Any]:
        """
        Prepare the payload for an API request, handling various input types and parameters.
        `expect_binary` is set to `True` when the inputs are a binary object or a local path or URL. This is the case for image and audio inputs.
        """
        if parameters is None:
            parameters = {}
        parameters = {k: v for k, v in parameters.items() if v is not None}
        has_parameters = len(parameters) > 0

        is_binary = isinstance(inputs, (bytes, Path))
        # If expect_binary is True, inputs must be a binary object or a local path or a URL.
        if expect_binary and not is_binary and not isinstance(inputs, str):
            raise ValueError(f"Expected binary inputs or a local path or a URL. Got {inputs}")  # type: ignore
        # Send inputs as raw content when no parameters are provided
        if expect_binary and not has_parameters:
            return {"data": inputs}
        # If expect_binary is False, inputs must not be a binary object.
        if not expect_binary and is_binary:
            raise ValueError(f"Unexpected binary inputs. Got {inputs}")  # type: ignore

        json: Dict[str, Any] = {}
        # If inputs is a bytes-like object, encode it to base64
        if expect_binary:
            json["inputs"] = _b64_encode(inputs)  # type: ignore
        # Otherwise (string, dict, list) send it as is
        else:
            json["inputs"] = inputs
        # Add parameters to the json payload if any
        if has_parameters:
            json["parameters"] = parameters
        return {"json": json}

    @staticmethod
    def get_response(response: Union[bytes, Dict]) -> Any:
        return response
