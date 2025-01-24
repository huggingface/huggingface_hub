from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from huggingface_hub.constants import ENDPOINT
from huggingface_hub.inference._common import RequestParameters, TaskProviderHelper, _b64_encode, _open_as_binary
from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status


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


class HFInferenceTask(TaskProviderHelper):
    """Base class for HF Inference API tasks."""

    def __init__(self, task: str):
        self.task = task

    def prepare_request(
        self,
        *,
        inputs: Any,
        parameters: Dict[str, Any],
        headers: Dict,
        model: Optional[str],
        api_key: Optional[str],
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> RequestParameters:
        if extra_payload is None:
            extra_payload = {}
        mapped_model = self.map_model(model)
        url = self.build_url(mapped_model)
        data, json = self._prepare_payload(inputs, parameters=parameters, model=model, extra_payload=extra_payload)
        headers = self.prepare_headers(headers=headers, api_key=api_key)

        return RequestParameters(
            url=url,
            task=self.task,
            model=mapped_model,
            json=json,
            data=data,
            headers=headers,
        )

    def map_model(self, model: Optional[str]) -> str:
        return model if model is not None else get_recommended_model(self.task)

    def build_url(self, model: str) -> str:
        # hf-inference provider can handle URLs (e.g. Inference Endpoints or TGI deployment)
        if model.startswith(("http://", "https://")):
            return model

        return (
            # Feature-extraction and sentence-similarity are the only cases where we handle models with several tasks.
            f"{BASE_URL}/pipeline/{self.task}/{model}"
            if self.task in ("feature-extraction", "sentence-similarity")
            # Otherwise, we use the default endpoint
            else f"{BASE_URL}/models/{model}"
        )

    def prepare_headers(self, headers: Dict, *, api_key: Optional[Union[bool, str]] = None) -> Dict:
        return {**build_hf_headers(token=api_key), **headers}

    def _prepare_payload(
        self, inputs: Any, parameters: Dict[str, Any], model: Optional[str], extra_payload: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        if isinstance(inputs, bytes):
            raise ValueError(f"Unexpected binary input for task {self.task}.")
        if isinstance(inputs, Path):
            raise ValueError(f"Unexpected path input for task {self.task} (got {inputs})")
        return None, {
            "inputs": inputs,
            "parameters": {k: v for k, v in parameters.items() if v is not None},
            **extra_payload,
        }

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        return response


class HFInferenceBinaryInputTask(HFInferenceTask):
    def _prepare_payload(
        self, inputs: Any, parameters: Dict[str, Any], model: Optional[str], extra_payload: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        parameters = {k: v for k, v in parameters.items() if v is not None}
        has_parameters = len(parameters) > 0 or len(extra_payload) > 0

        # Raise if not a binary object or a local path or a URL.
        if not isinstance(inputs, (bytes, Path)) and not isinstance(inputs, str):
            raise ValueError(f"Expected binary inputs or a local path or a URL. Got {inputs}")

        # Send inputs as raw content when no parameters are provided
        if not has_parameters:
            with _open_as_binary(inputs) as data:
                data_as_bytes = data if isinstance(data, bytes) else data.read()
                return data_as_bytes, None

        # Otherwise encode as b64
        return None, {"inputs": _b64_encode(inputs), "parameters": parameters, **extra_payload}


class HFInferenceConversational(HFInferenceTask):
    def __init__(self):
        super().__init__("text-generation")

    def prepare_request(
        self,
        *,
        inputs: Any,
        parameters: Dict[str, Any],
        headers: Dict,
        model: Optional[str],
        api_key: Optional[str],
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> RequestParameters:
        model = self.map_model(model)
        payload_model = parameters.get("model") or model

        if payload_model is None or payload_model.startswith(("http://", "https://")):
            payload_model = "tgi"  # use a random string if not provided

        json = {
            **{key: value for key, value in parameters.items() if value is not None},
            "model": payload_model,
            "messages": inputs,
            **(extra_payload or {}),
        }
        headers = self.prepare_headers(headers=headers, api_key=api_key)

        return RequestParameters(
            url=self.build_url(model),
            task=self.task,
            model=model,
            json=json,
            data=None,
            headers=headers,
        )

    def build_url(self, model: str) -> str:
        base_url = model if model.startswith(("http://", "https://")) else f"{BASE_URL}/models/{model}"
        return _build_chat_completion_url(base_url)


def _build_chat_completion_url(model_url: str) -> str:
    # Strip trailing /
    model_url = model_url.rstrip("/")

    # Append /chat/completions if not already present
    if model_url.endswith("/v1"):
        model_url += "/chat/completions"

    # Append /v1/chat/completions if not already present
    if not model_url.endswith("/chat/completions"):
        model_url += "/v1/chat/completions"

    return model_url
