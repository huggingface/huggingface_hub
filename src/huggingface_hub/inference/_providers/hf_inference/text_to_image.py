from typing import Any, Dict, Optional, Union

from ._common import get_recommended_model


BASE_URL = "https://api-inference.huggingface.co"


def build_url(model: Optional[str] = None) -> str:
    if model is None:
        model = get_recommended_model("text-to-image")
    url = f"{BASE_URL}/models/{model}"
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url += "/chat/completions"
    elif not url.endswith("/chat/completions"):
        url += "/v1/chat/completions"
    return url


def map_model(model: str) -> str:
    return model


def prepare_headers(headers: Dict, **kwargs) -> Dict:
    return headers


def prepare_payload(inputs: Any, parameters: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    json = {
        "inputs": inputs,
        "model": model,
        **parameters,
    }
    return {"json": json}


def get_response(response: Union[bytes, Dict]) -> Any:
    return response
