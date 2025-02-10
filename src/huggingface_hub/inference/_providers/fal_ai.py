import base64
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from huggingface_hub.inference._common import RequestParameters, TaskProviderHelper, _as_dict
from huggingface_hub.inference._providers._common import filter_none, get_base_url, get_mapped_model
from huggingface_hub.utils import build_hf_headers, get_session, get_token, logging


logger = logging.get_logger(__name__)


BASE_URL = "https://fal.run"


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
            api_key = get_token()
        if api_key is None:
            raise ValueError(
                "You must provide an api_key to work with fal.ai API or log in with `huggingface-cli login`."
            )
        mapped_model = get_mapped_model("fal-ai", model, self.task)
        headers = {**build_hf_headers(token=api_key), **headers}

        # Route to the proxy if the api_key is a HF TOKEN
        base_url = get_base_url("fal-ai", BASE_URL, api_key)
        if not api_key.startswith("hf_"):
            headers["authorization"] = f"Key {api_key}"

        payload = self._prepare_payload(inputs, parameters=parameters)

        return RequestParameters(
            url=f"{base_url}/{mapped_model}",
            task=self.task,
            model=mapped_model,
            json=payload,
            data=None,
            headers=headers,
        )

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

        return {"audio_url": audio_url, **filter_none(parameters)}

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        text = _as_dict(response)["text"]
        if not isinstance(text, str):
            raise ValueError(f"Unexpected output format from FalAI API. Expected string, got {type(text)}.")
        return text


class FalAITextToImageTask(FalAITask):
    def __init__(self):
        super().__init__("text-to-image")

    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        parameters = filter_none(parameters)
        if "width" in parameters and "height" in parameters:
            parameters["image_size"] = {
                "width": parameters.pop("width"),
                "height": parameters.pop("height"),
            }
        return {"prompt": inputs, **parameters}

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        url = _as_dict(response)["images"][0]["url"]
        return get_session().get(url).content


class FalAITextToSpeechTask(FalAITask):
    def __init__(self):
        super().__init__("text-to-speech")

    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return {"lyrics": inputs, **filter_none(parameters)}

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        url = _as_dict(response)["audio"]["url"]
        return get_session().get(url).content


class FalAITextToVideoTask(FalAITask):
    def __init__(self):
        super().__init__("text-to-video")

    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return {"prompt": inputs, **filter_none(parameters)}

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        url = _as_dict(response)["video"]["url"]
        return get_session().get(url).content
