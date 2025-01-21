from typing import Any, Dict, Optional, Union


BASE_URL = "https://api.sambanova.ai"

SUPPORTED_MODELS = {
    "Qwen/Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B-Instruct",
    "Qwen/QwQ-32B-Preview": "QwQ-32B-Preview",
    "meta-llama/Llama-3.3-70B-Instruct": "Meta-Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.2-1B": "Meta-Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B": "Meta-Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct": "Llama-3.2-90B-Vision-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct": "Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-405B-Instruct": "Meta-Llama-3.1-405B-Instruct",
    "meta-llama/Llama-Guard-3-8B": "Meta-Llama-Guard-3-8B",
}


def build_url(model: Optional[str] = None) -> str:
    return f"{BASE_URL}/v1/chat/completions"


def map_model(model: str) -> str:
    mapped_model = SUPPORTED_MODELS.get(model)
    if mapped_model is None:
        raise ValueError(f"Model {model} is not supported for Sambanova conversational task")
    return mapped_model


def prepare_headers(headers: Dict, *, token: Optional[str] = None) -> Dict:
    return headers


def prepare_payload(inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
    if "model" not in parameters:
        raise ValueError("Model is required for Sambanova conversational task")
    payload = {"messages": inputs, **{k: v for k, v in parameters.items() if v is not None}}
    return payload


def get_response(response: Union[bytes, Dict]) -> Any:
    return response
