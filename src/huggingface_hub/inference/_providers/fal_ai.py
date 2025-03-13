import base64
import time
from abc import ABC
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse

from huggingface_hub.inference._common import RequestParameters, _as_dict
from huggingface_hub.inference._providers._common import TaskProviderHelper, filter_none
from huggingface_hub.utils import get_session, hf_raise_for_status
from huggingface_hub.utils.logging import get_logger


logger = get_logger(__name__)

# Arbitrary polling interval
_POLLING_INTERVAL = 2.0


class FalAITask(TaskProviderHelper, ABC):
    def __init__(self, task: str):
        super().__init__(provider="fal-ai", base_url="https://fal.run", task=task)

    def _prepare_headers(self, headers: Dict, api_key: str) -> Dict:
        headers = super()._prepare_headers(headers, api_key)
        if not api_key.startswith("hf_"):
            headers["authorization"] = f"Key {api_key}"
        return headers

    def _prepare_route(self, mapped_model: str, api_key: Optional[str] = None) -> str:
        return f"/{mapped_model}"


class FalAIAutomaticSpeechRecognitionTask(FalAITask):
    def __init__(self):
        super().__init__("automatic-speech-recognition")

    def _prepare_payload_as_dict(self, inputs: Any, parameters: Dict, mapped_model: str) -> Optional[Dict]:
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

        return {"audio_url": audio_url, **filter_none(parameters)}

    def get_response(self, response: Union[bytes, Dict], request_params: Optional[RequestParameters] = None) -> Any:
        text = _as_dict(response)["text"]
        if not isinstance(text, str):
            raise ValueError(f"Unexpected output format from FalAI API. Expected string, got {type(text)}.")
        return text


class FalAITextToImageTask(FalAITask):
    def __init__(self):
        super().__init__("text-to-image")

    def _prepare_payload_as_dict(self, inputs: Any, parameters: Dict, mapped_model: str) -> Optional[Dict]:
        parameters = filter_none(parameters)
        if "width" in parameters and "height" in parameters:
            parameters["image_size"] = {
                "width": parameters.pop("width"),
                "height": parameters.pop("height"),
            }
        return {"prompt": inputs, **parameters}

    def get_response(self, response: Union[bytes, Dict], request_params: Optional[RequestParameters] = None) -> Any:
        url = _as_dict(response)["images"][0]["url"]
        return get_session().get(url).content


class FalAITextToSpeechTask(FalAITask):
    def __init__(self):
        super().__init__("text-to-speech")

    def _prepare_payload_as_dict(self, inputs: Any, parameters: Dict, mapped_model: str) -> Optional[Dict]:
        return {"lyrics": inputs, **filter_none(parameters)}

    def get_response(self, response: Union[bytes, Dict], request_params: Optional[RequestParameters] = None) -> Any:
        url = _as_dict(response)["audio"]["url"]
        return get_session().get(url).content


class FalAITextToVideoTask(FalAITask):
    def __init__(self):
        super().__init__("text-to-video")

    def _prepare_route(self, mapped_model: str, api_key: Optional[str] = None) -> str:
        if api_key and api_key.startswith("hf_"):
            # Use the queue subdomain for HF routing
            return f"/{mapped_model}?_subdomain=queue"
        return f"/{mapped_model}"

    def _prepare_payload_as_dict(self, inputs: Any, parameters: Dict, mapped_model: str) -> Optional[Dict]:
        return {"prompt": inputs, **filter_none(parameters)}

    def get_response(
        self,
        response: Union[bytes, Dict],
        request_params: Optional[RequestParameters] = None,
    ) -> Any:
        response_dict = _as_dict(response)

        request_id = response_dict.get("request_id")
        if not request_id:
            raise ValueError("No request ID found in the response")
        if request_params is None:
            raise ValueError("Request parameters should be provided to get text-to-video responses with Fal AI.")
        # extract the base url and query params

        parsed = urlparse(request_params.url)
        base_url = request_params.url.split("?")[0]  # or parsed.scheme + "://" + parsed.netloc + parsed.path ?
        query = "?" + parsed.query if parsed.query else ""

        status_url = f"{base_url}/requests/{request_id}/status{query}"
        result_url = f"{base_url}/requests/{request_id}{query}"

        status = response_dict.get("status")
        logger.info("Generating the video.. this can take several minutes.")
        while status != "COMPLETED":
            time.sleep(_POLLING_INTERVAL)
            response = get_session().get(status_url, headers=request_params.headers)
            hf_raise_for_status(response)
            response_dict = _as_dict(response.json())
            status = response_dict.get("status")

        response = get_session().get(result_url, headers=request_params.headers).json()
        url = _as_dict(response)["video"]["url"]
        return get_session().get(url).content
