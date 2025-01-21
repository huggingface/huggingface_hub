from typing import Any, Dict, Optional, Union


BASE_URL = "https://api.together.xyz"

SUPPORTED_MODELS = {
    "meta-llama/Meta-Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    "mistralai/Mixtral-8x7B-v0.1": "mistralai/Mixtral-8x7B-v0.1",
}


def build_url(model: Optional[str] = None) -> str:
    return f"{BASE_URL}/v1/completions"


def map_model(model: str) -> str:
    mapped_model = SUPPORTED_MODELS.get(model)
    if mapped_model is None:
        raise ValueError(f"Model {model} is not supported for Together text-generation task")
    return mapped_model


def prepare_headers(headers: Dict, *, token: Optional[str] = None) -> Dict:
    return headers


def prepare_payload(inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
    if "model" not in parameters:
        raise ValueError("Model is required for Together text-generation task")
    payload = {"messages": inputs, **{k: v for k, v in parameters.items() if v is not None}}
    return payload


def get_response(response: Union[bytes, Dict]) -> Any:
    return response
