from typing import Any, Dict, Optional, Union

from huggingface_hub import constants
from huggingface_hub.inference._common import RequestParameters, TaskProviderHelper, _as_dict
from huggingface_hub.utils import build_hf_headers, get_session, get_token, logging


logger = logging.get_logger(__name__)


BASE_URL = "https://api.replicate.com"

SUPPORTED_MODELS = {
    "text-to-image": {
        "black-forest-labs/FLUX.1-dev": "black-forest-labs/flux-dev",
        "black-forest-labs/FLUX.1-schnell": "black-forest-labs/flux-schnell",
        "ByteDance/Hyper-SD": "bytedance/hyper-flux-16step:382cf8959fb0f0d665b26e7e80b8d6dc3faaef1510f14ce017e8c732bb3d1eb7",
        "ByteDance/SDXL-Lightning": "bytedance/sdxl-lightning-4step:5599ed30703defd1d160a25a63321b4dec97101d98b4674bcc56e41f62f35637",
        "playgroundai/playground-v2.5-1024px-aesthetic": "playgroundai/playground-v2.5-1024px-aesthetic:a45f82a1382bed5c7aeb861dac7c7d191b0fdf74d8d57c4a0e6ed7d4d0bf7d24",
        "stabilityai/stable-diffusion-3.5-large-turbo": "stability-ai/stable-diffusion-3.5-large-turbo",
        "stabilityai/stable-diffusion-3.5-large": "stability-ai/stable-diffusion-3.5-large",
        "stabilityai/stable-diffusion-3.5-medium": "stability-ai/stable-diffusion-3.5-medium",
        "stabilityai/stable-diffusion-xl-base-1.0": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
    },
    "text-to-speech": {
        "OuteAI/OuteTTS-0.3-500M": "jbilcke/oute-tts:39a59319327b27327fa3095149c5a746e7f2aee18c75055c3368237a6503cd26",
    },
    "text-to-video": {
        "genmo/mochi-1-preview": "genmoai/mochi-1:1944af04d098ef69bed7f9d335d102e652203f268ec4aaa2d836f6217217e460",
    },
}


def _build_url(base_url: str, model: str) -> str:
    if ":" in model:
        return f"{base_url}/v1/predictions"
    return f"{base_url}/v1/models/{model}/predictions"


class ReplicateTask(TaskProviderHelper):
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
                "You must provide an api_key to work with Replicate API or log in with `huggingface-cli login`."
            )

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

    def _prepare_payload(
        self,
        inputs: Any,
        parameters: Dict[str, Any],
        model: str,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "input": {
                "prompt": inputs,
                **{k: v for k, v in parameters.items() if v is not None},
            }
        }
        if ":" in model:
            version = model.split(":", 1)[1]
            payload["version"] = version
        return payload

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        response_dict = _as_dict(response)
        if response_dict.get("output") is None:
            raise TimeoutError(
                f"Inference request timed out after 60 seconds. No output generated for model {response_dict.get('model')}"
                "The model might be in cold state or starting up. Please try again later."
            )
        output_url = (
            response_dict["output"] if isinstance(response_dict["output"], str) else response_dict["output"][0]
        )
        return get_session().get(output_url).content


class ReplicateTextToSpeechTask(ReplicateTask):
    def __init__(self):
        super().__init__("text-to-speech")

    def _prepare_payload(
        self,
        inputs: Any,
        parameters: Dict[str, Any],
        model: str,
    ) -> Dict[str, Any]:
        # The following payload might work only for a subset of text-to-speech Replicate models.
        payload: Dict[str, Any] = {
            "input": {
                "inputs": inputs,
                **{k: v for k, v in parameters.items() if v is not None},
            },
        }
        if ":" in model:
            version = model.split(":", 1)[1]
            payload["version"] = version
        return payload
