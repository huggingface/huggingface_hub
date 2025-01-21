import json
from typing import Any, Dict, Optional, Union

from huggingface_hub.utils import get_session


BASE_URL = "https://api.replicate.com"

SUPPORTED_MODELS = {
    "black-forest-labs/FLUX.1-schnell": "black-forest-labs/flux-schnell",
    "ByteDance/SDXL-Lightning": "bytedance/sdxl-lightning-4step:5599ed30703defd1d160a25a63321b4dec97101d98b4674bcc56e41f62f35637",
}


def build_url(model: Optional[str] = None) -> str:
    if model is not None and ":" in model:
        return f"{BASE_URL}/v1/predictions"
    return f"{BASE_URL}/v1/models/{model}/predictions"


def map_model(model: str) -> str:
    mapped_model = SUPPORTED_MODELS.get(model)
    if mapped_model is None:
        raise ValueError(f"Model {model} is not supported for Replicate text-to-image task")
    return mapped_model


def prepare_headers(headers: Dict, *, token: Optional[str] = None) -> Dict:
    headers["Prefer"] = "wait"
    return headers


def prepare_payload(inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "json": {
            "input": {"prompt": inputs, **{k: v for k, v in parameters.items() if v is not None}},
        }
    }
    model = parameters.get("model")
    if model is not None and ":" in model:
        version = model.split(":", 1)[1]
        payload["json"]["version"] = version  # type: ignore
    return payload


def get_response(response: Union[bytes, Dict]) -> Any:
    if isinstance(response, bytes):
        response_dict = json.loads(response)
    else:
        response_dict = response
    image_url = response_dict["output"][0]
    return get_session().get(image_url).content
