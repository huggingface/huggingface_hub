import json
from typing import Any, Dict, Optional, Union

from huggingface_hub.utils import get_session


BASE_URL = "https://fal.run"

SUPPORTED_MODELS = {
    "black-forest-labs/FLUX.1-schnell": "fal-ai/flux/schnell",
    "black-forest-labs/FLUX.1-dev": "fal-ai/flux/dev",
}


def build_url(model: Optional[str] = None) -> str:
    return f"{BASE_URL}/{model}"


def map_model(model: str) -> str:
    mapped_model = SUPPORTED_MODELS.get(model)
    if mapped_model is None:
        raise ValueError(f"Model {model} is not supported for Fal.AI text-to-image task")
    return mapped_model


def prepare_headers(headers: Dict, **kwargs) -> Dict:
    headers["Authorization"] = f"Key {kwargs['token']}"
    return headers


def prepare_payload(inputs: Any, parameters: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    parameters = {k: v for k, v in parameters.items() if v is not None}
    if len(parameters) > 0:
        return {"json": {"prompt": inputs, **parameters}}
    return {"json": {"prompt": inputs}}


def get_response(response: Union[bytes, Dict]) -> Any:
    if isinstance(response, bytes):
        response_dict = json.loads(response)  # type: ignore
    else:
        response_dict = response
    url = response_dict["images"][0]["url"]
    return get_session().get(url).content
