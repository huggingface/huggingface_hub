from typing import Any, Dict, Optional, Union


BASE_URL = "https://api.together.xyz"

SUPPORTED_MODELS = {
    "databricks/dbrx-instruct": "databricks/dbrx-instruct",
    "deepseek-ai/deepseek-llm-67b-chat": "deepseek-ai/deepseek-llm-67b-chat",
    "google/gemma-2-9b-it": "google/gemma-2-9b-it",
    "google/gemma-2b-it": "google/gemma-2-27b-it",
    "llava-hf/llava-v1.6-mistral-7b-hf": "llava-hf/llava-v1.6-mistral-7b-hf",
    "meta-llama/Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-hf": "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "meta-llama/Llama-Vision-Free",
    "meta-llama/Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "meta-llama/Llama-3.2-90B-Vision-Instruct": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3-70B-Instruct": "meta-llama/Llama-3-70b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct": "togethercomputer/Llama-3-8b-chat-hf-int4",
    "meta-llama/Meta-Llama-3.1-405B-Instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
    "microsoft/WizardLM-2-8x22B": "microsoft/WizardLM-2-8x22B",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x22B-Instruct-v0.1": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "Qwen/Qwen2-72B-Instruct": "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/QwQ-32B-Preview": "Qwen/QwQ-32B-Preview",
    "scb10x/llama-3-typhoon-v1.5-8b-instruct": "scb10x/scb10x-llama3-typhoon-v1-5-8b-instruct",
    "scb10x/llama-3-typhoon-v1.5x-70b-instruct-awq": "scb10x/scb10x-llama3-typhoon-v1-5x-4f316",
}


def build_url(model: Optional[str] = None) -> str:
    return f"{BASE_URL}/v1/chat/completions"


def map_model(model: str) -> str:
    mapped_model = SUPPORTED_MODELS.get(model)
    if mapped_model is None:
        raise ValueError(f"Model {model} is not supported for Together conversational task")
    return mapped_model


def prepare_headers(headers: Dict, **kwargs) -> Dict:
    return headers


def prepare_payload(inputs: Any, parameters: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    payload = {
        "messages": inputs,
        "model": model,
        **parameters,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    return payload


def get_response(response: Union[bytes, Dict]) -> Any:
    return response
