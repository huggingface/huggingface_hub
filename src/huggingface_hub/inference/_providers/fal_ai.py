import base64
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from huggingface_hub import constants
from huggingface_hub.inference._common import RequestParameters, TaskProviderHelper, _as_dict
from huggingface_hub.utils import build_hf_headers, get_session, get_token, logging


logger = logging.get_logger(__name__)


BASE_URL = "https://fal.run"

SUPPORTED_MODELS = {
    "automatic-speech-recognition": {
        "openai/whisper-large-v3": "fal-ai/whisper",
    },
    "text-to-image": {
        "black-forest-labs/FLUX.1-schnell": "fal-ai/flux/schnell",
        "black-forest-labs/FLUX.1-dev": "fal-ai/flux/dev",
        "playgroundai/playground-v2.5-1024px-aesthetic": "fal-ai/playground-v25",
        "ByteDance/SDXL-Lightning": "fal-ai/lightning-models",
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": "fal-ai/pixart-sigma",
        "stabilityai/stable-diffusion-3-medium": "fal-ai/stable-diffusion-v3-medium",
        "Warlord-K/Sana-1024": "fal-ai/sana",
        "fal/AuraFlow-v0.2": "fal-ai/aura-flow",
        "stabilityai/stable-diffusion-3.5-large": "fal-ai/stable-diffusion-v35-large",
        "Kwai-Kolors/Kolors": "fal-ai/kolors",
    },
    "text-to-video": {
        "genmo/mochi-1-preview": "fal-ai/mochi-v1",
        "tencent/HunyuanVideo": "fal-ai/hunyuan-video",
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
            api_key = get_token()
        if api_key is None:
            raise ValueError(
                "You must provide an api_key to work with fal.ai API or log in with `huggingface-cli login`."
            )

        mapped_model = self._map_model(model)
        headers = {
            **build_hf_headers(token=api_key),
            **headers,
        }

        # Route to the proxy if the api_key is a HF TOKEN
        if api_key.startswith("hf_"):
            base_url = constants.INFERENCE_PROXY_TEMPLATE.format(provider="fal-ai")
            logger.info("Calling fal.ai provider through Hugging Face proxy.")
        else:
            base_url = BASE_URL
            headers["authorization"] = f"Key {api_key}"
            logger.info("Calling fal.ai provider directly.")

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


class FalAITextToVideoTask(FalAITask):
    def __init__(self):
        super().__init__("text-to-video")

    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        parameters = {k: v for k, v in parameters.items() if v is not None}
        return {"prompt": inputs, **parameters}

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        url = _as_dict(response)["video"]["url"]
        return get_session().get(url).content
