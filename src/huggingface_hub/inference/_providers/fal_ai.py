import base64
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from huggingface_hub.constants import INFERENCE_PROXY_TEMPLATE
from huggingface_hub.inference._common import RequestParameters, TaskProviderHelper
from huggingface_hub.utils import build_hf_headers, get_session, logging


logger = logging.get_logger(__name__)


BASE_URL = "https://fal.run"

SUPPORTED_MODELS = {
    "automatic-speech-recognition": {
        "openai/whisper-large-v3": "fal-ai/whisper",
    },
    "text-to-image": {
        "black-forest-labs/FLUX.1-schnell": "fal-ai/flux/schnell",
        "black-forest-labs/FLUX.1-dev": "fal-ai/flux/dev",
    },
}


class FalAITask(TaskProviderHelper, ABC):
    """Base class for FalAI API tasks."""

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
            raise ValueError("You must provide an api_key to work with fal.ai API.")

        mapped_model = self._map_model(model)
        headers = {
            **build_hf_headers(token=api_key),
            **headers,
        }

        # Route to the proxy if the api_key is a HF TOKEN
        if api_key.startswith("hf_"):
            base_url = INFERENCE_PROXY_TEMPLATE.format(provider="fal-ai")
            logger.info(
                "Routing the call through Hugging Face's infrastructure using your HF token, "
                "and the usage will be billed directly to your Hugging Face account"
            )
        else:
            base_url = BASE_URL
            headers["authorization"] = f"Key {api_key}"
            logger.info("Interacting directly with fal.ai's service using the provided API key")

        payload = self._prepare_payload(inputs, parameters=parameters)

        return RequestParameters(
            url=f"{base_url}/{mapped_model}",
            task=self.task,
            model=mapped_model,
            json=payload,
            data=None,
            headers=headers,
        )

    def _map_model(self, model: Optional[str]) -> str:
        if model is None:
            raise ValueError("Please provide a model available on FalAI.")
        if self.task not in SUPPORTED_MODELS:
            raise ValueError(f"Task {self.task} not supported with FalAI.")
        mapped_model = SUPPORTED_MODELS[self.task].get(model)
        if mapped_model is None:
            raise ValueError(f"Model {model} is not supported with FalAI for task {self.task}.")
        return mapped_model

    @abstractmethod
    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]: ...


class FalAIAutomaticSpeechRecognitionTask(FalAITask):
    def __init__(self):
        super().__init__("automatic-speech-recognition")

    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(inputs, str) and inputs.startswith(("http://", "https://")):
            # If input is a URL, pass it directly
            audio_url = inputs
        else:
            # If input is a file path, read it first
            if isinstance(inputs, str):
                with open(inputs, "rb") as f:
                    inputs = f.read()

            audio_b64 = base64.b64encode(inputs).decode()
            content_type = "audio/mpeg"
            audio_url = f"data:{content_type};base64,{audio_b64}"

        return {
            "audio_url": audio_url,
            **{k: v for k, v in parameters.items() if v is not None},
        }

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        text = _as_dict(response)["text"]
        if not isinstance(text, str):
            raise ValueError(f"Unexpected output format from FalAI API. Expected string, got {type(text)}.")
        return text


class FalAITextToImageTask(FalAITask):
    def __init__(self):
        super().__init__("text-to-image")

    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        parameters = {k: v for k, v in parameters.items() if v is not None}
        if "image_size" not in parameters and "width" in parameters and "height" in parameters:
            parameters["image_size"] = {
                "width": parameters.pop("width"),
                "height": parameters.pop("height"),
            }
        return {"prompt": inputs, **parameters}

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        url = _as_dict(response)["images"][0]["url"]
        return get_session().get(url).content


def _as_dict(response: Union[bytes, Dict]) -> Dict:
    return json.loads(response) if isinstance(response, bytes) else response
