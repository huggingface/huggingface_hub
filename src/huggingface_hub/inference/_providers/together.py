import base64
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from huggingface_hub.inference._common import RequestParameters, TaskProviderHelper, _as_dict
from huggingface_hub.inference._providers._common import filter_none, get_base_url, get_mapped_model
from huggingface_hub.utils import build_hf_headers, get_token, logging


logger = logging.get_logger(__name__)


BASE_URL = "https://api.together.xyz"

PER_TASK_ROUTES = {
    "conversational": "v1/chat/completions",
    "text-generation": "v1/completions",
    "text-to-image": "v1/images/generations",
}


class TogetherTask(TaskProviderHelper, ABC):
    """Base class for Together API tasks."""

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
        if api_key is None:
            api_key = get_token()
        if api_key is None:
            raise ValueError(
                "You must provide an api_key to work with Together API or log in with `huggingface-cli login`."
            )
        headers = {**build_hf_headers(token=api_key), **headers}

        # Route to the proxy if the api_key is a HF TOKEN
        base_url = get_base_url("together", BASE_URL, api_key)
        mapped_model = mapped_model = get_mapped_model("fal-ai", model, self.task)

        if "model" in parameters:
            parameters["model"] = mapped_model
        payload = self._prepare_payload(inputs, parameters=parameters)

        return RequestParameters(
            url=f"{base_url}/{PER_TASK_ROUTES[self.task]}",
            task=self.task,
            model=mapped_model,
            json=payload,
            data=None,
            headers=headers,
        )

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        return response

    @abstractmethod
    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]: ...


class TogetherTextGenerationTask(TogetherTask):
    # Handle both "text-generation" and "conversational"
    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": inputs, **filter_none(parameters)}


class TogetherTextToImageTask(TogetherTask):
    def __init__(self):
        super().__init__("text-to-image")

    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        parameters = filter_none(parameters)
        if "num_inference_steps" in parameters:
            parameters["steps"] = parameters.pop("num_inference_steps")
        if "guidance_scale" in parameters:
            parameters["guidance"] = parameters.pop("guidance_scale")

        return {"prompt": inputs, "response_format": "base64", **parameters}

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        response_dict = _as_dict(response)
        return base64.b64decode(response_dict["data"][0]["b64_json"])
