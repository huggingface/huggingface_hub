import base64
import json
from typing import Any, Dict, Optional, Union


BASE_URL = "https://api.together.xyz"

SUPPORTED_MODELS = {
    "black-forest-labs/FLUX.1-Canny-dev": "black-forest-labs/FLUX.1-canny",
    "black-forest-labs/FLUX.1-Depth-dev": "black-forest-labs/FLUX.1-depth",
    "black-forest-labs/FLUX.1-dev": "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-Redux-dev": "black-forest-labs/FLUX.1-redux",
    "black-forest-labs/FLUX.1-schnell": "black-forest-labs/FLUX.1-pro",
    "stabilityai/stable-diffusion-xl-base-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
}


def build_url(model: Optional[str] = None) -> str:
    return f"{BASE_URL}/v1/images/generations"


def map_model(model: str) -> str:
    mapped_model = SUPPORTED_MODELS.get(model)
    if mapped_model is None:
        raise ValueError(f"Model {model} is not supported for Together text-to-image task")
    return mapped_model


def prepare_headers(headers: Dict, *, token: Optional[str] = None) -> Dict:
    return headers


def prepare_payload(
    input: str,
    parameters: Dict[str, Any],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    parameters = {key: value for key, value in parameters.items() if value is not None}
    payload = {"json": {"prompt": input, "model": model, "response_format": "base64", **parameters}}
    return payload


def get_response(response: Union[bytes, Dict]) -> Any:
    if isinstance(response, bytes):
        response_dict = json.loads(response)
    else:
        response_dict = response
    return base64.b64decode(response_dict["data"][0]["b64_json"])
