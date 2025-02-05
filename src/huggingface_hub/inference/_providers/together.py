import base64
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from huggingface_hub import constants
from huggingface_hub.inference._common import RequestParameters, TaskProviderHelper, _as_dict, _get_provider_model_id
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
        conversational: bool = False,
    ) -> RequestParameters:
        if api_key is None:
            api_key = get_token()
        if api_key is None:
            raise ValueError(
                "You must provide an api_key to work with Together API or log in with `huggingface-cli login`."
            )
        headers = {**build_hf_headers(token=api_key), **headers}

        # Route to the proxy if the api_key is a HF TOKEN
        if api_key.startswith("hf_"):
            base_url = constants.INFERENCE_PROXY_TEMPLATE.format(provider="together")
            logger.info("Calling Together provider through Hugging Face proxy.")
        else:
            base_url = BASE_URL
            logger.info("Calling Together provider directly.")
        mapped_model = self.map_model(model=model, task=self.task, conversational=conversational)
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

    def map_model(
        self,
        model: Optional[str],
        task: str,
        conversational: bool = False,
    ) -> str:
        """Default implementation for mapping model HF model IDs to provider model IDs."""
        if model is None:
            raise ValueError("Please provide a HF model ID supported by Together.")
        return _get_provider_model_id(model, "together", task, conversational)

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        return response

    @abstractmethod
    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]: ...


class TogetherTextGenerationTask(TogetherTask):
    # Handle both "text-generation" and "conversational"
    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": inputs, **{k: v for k, v in parameters.items() if v is not None}}


class TogetherTextToImageTask(TogetherTask):
    def __init__(self):
        super().__init__("text-to-image")

    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        parameters = {k: v for k, v in parameters.items() if v is not None}
        if "num_inference_steps" in parameters:
            parameters["steps"] = parameters.pop("num_inference_steps")
        if "guidance_scale" in parameters:
            parameters["guidance"] = parameters.pop("guidance_scale")

        payload = {
            "prompt": inputs,
            "response_format": "base64",
            **parameters,
        }
        return payload

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        response_dict = _as_dict(response)
        return base64.b64decode(response_dict["data"][0]["b64_json"])
