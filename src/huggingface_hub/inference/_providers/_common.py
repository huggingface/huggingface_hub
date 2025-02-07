import logging
from functools import lru_cache
from typing import Any, Dict, Optional

from huggingface_hub import constants


logger = logging.getLogger(__name__)


# Dev purposes only.
# If you want to try to run inference for a new model locally before it's registered on huggingface.co
# for a given Inference Provider, you can add it to the following dictionary.
HARDCODED_MODEL_ID_MAPPING: Dict[str, Dict[str, str]] = {
    # "HF model ID" => "Model ID on Inference Provider's side"
    #
    # Example:
    # "Qwen/Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B-Instruct",
    "fal-ai": {},
    "fireworks-ai": {},
    "hf-inference": {},
    "replicate": {},
    "sambanova": {},
    "together": {},
}


def get_base_url(provider: str, base_url: str, api_key: str) -> str:
    # Route to the proxy if the api_key is a HF TOKEN
    if api_key.startswith("hf_"):
        logger.info(f"Calling '{provider}' provider through Hugging Face router.")
        return constants.INFERENCE_PROXY_TEMPLATE.format(provider=provider)
    else:
        logger.info("Calling '{provider}' provider directly.")
        return base_url


def get_mapped_model(provider: str, model: Optional[str], task: str) -> str:
    if model is None:
        raise ValueError(f"Please provide an HF model ID supported by {provider}.")

    # hardcoded mapping for local testing
    if HARDCODED_MODEL_ID_MAPPING.get(provider, {}).get(model):
        return HARDCODED_MODEL_ID_MAPPING[provider][model]

    provider_mapping = _fetch_inference_provider_mapping(model).get(provider)
    if provider_mapping is None:
        raise ValueError(f"Model {model} is not supported by provider {provider}.")

    if provider_mapping.task != task:
        raise ValueError(
            f"Model {model} is not supported for task {task} and provider {provider}. "
            f"Supported task: {provider_mapping.task}."
        )

    if provider_mapping.status == "staging":
        logger.warning(f"Model {model} is in staging mode for provider {provider}. Meant for test purposes only.")
    return provider_mapping.provider_id


def filter_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


@lru_cache(maxsize=None)
def _fetch_inference_provider_mapping(model: str) -> Dict:
    """
    Fetch provider mappings for a model from the Hub.
    """
    from huggingface_hub.hf_api import model_info

    info = model_info(model, expand=["inferenceProviderMapping"])
    provider_mapping = info.inference_provider_mapping
    if provider_mapping is None:
        raise ValueError(f"No provider mapping found for model {model}")
    return provider_mapping
