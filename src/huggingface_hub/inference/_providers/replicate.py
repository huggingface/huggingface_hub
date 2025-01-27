import json
from typing import Any, Dict, Optional, Union

from huggingface_hub import constants
from huggingface_hub.inference._common import RequestParameters, TaskProviderHelper
from huggingface_hub.utils import build_hf_headers, get_session, logging


logger = logging.get_logger(__name__)


BASE_URL = "https://api.replicate.com"

SUPPORTED_MODELS = {
    "text-to-image": {
        "black-forest-labs/FLUX.1-schnell": "black-forest-labs/flux-schnell",
        "ByteDance/SDXL-Lightning": "bytedance/sdxl-lightning-4step:5599ed30703defd1d160a25a63321b4dec97101d98b4674bcc56e41f62f35637",
    },
}


def _build_url(base_url: str, model: str) -> str:
    if ":" in model:
        return f"{base_url}/v1/predictions"
    return f"{base_url}/v1/models/{model}/predictions"


class ReplicateTextToImageTask(TaskProviderHelper):
    def __init__(self):
        # TODO: adapt in a base class when supporting multiple tasks
        self.task = "text-to-image"

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
            raise ValueError("You must provide an api_key to work with Replicate API.")

        # Route to the proxy if the api_key is a HF TOKEN
        if api_key.startswith("hf_"):
            base_url = constants.INFERENCE_PROXY_TEMPLATE.format(provider="replicate")
            logger.info("Calling Replicate provider through Hugging Face proxy.")
        else:
            base_url = BASE_URL
            logger.info("Calling Replicate provider directly.")
        mapped_model = self._map_model(model)
        url = _build_url(base_url, mapped_model)

        headers = {
            **build_hf_headers(token=api_key),
            **headers,
            "Prefer": "wait",
        }

        payload = self._prepare_payload(inputs, parameters=parameters, model=mapped_model)

        return RequestParameters(
            url=url,
            task=self.task,
            model=mapped_model,
            json=payload,
            data=None,
            headers=headers,
        )

    def _map_model(self, model: Optional[str]) -> str:
        if model is None:
            raise ValueError("Please provide a model available on Replicate.")
        if self.task not in SUPPORTED_MODELS:
            raise ValueError(f"Task {self.task} not supported with Replicate.")
        mapped_model = SUPPORTED_MODELS[self.task].get(model)
        if mapped_model is None:
            raise ValueError(f"Model {model} is not supported with Replicate for task {self.task}.")
        return mapped_model

    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any], model: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "input": {
                "prompt": inputs,
                **{k: v for k, v in parameters.items() if v is not None},
                "model": model,
            }
        }
        if ":" in model:
            version = model.split(":", 1)[1]
            payload["version"] = version
        return payload

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        if isinstance(response, bytes):
            response_dict = json.loads(response)
        else:
            response_dict = response
        image_url = response_dict["output"][0]
        return get_session().get(image_url).content
