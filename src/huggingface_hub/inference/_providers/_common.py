from functools import lru_cache
from typing import Any, overload

from huggingface_hub import constants
from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import MimeBytes, RequestParameters
from huggingface_hub.inference._generated.types.chat_completion import ChatCompletionInputMessage
from huggingface_hub.utils import build_hf_headers, get_token, logging


logger = logging.get_logger(__name__)


# Dev purposes only.
# If you want to try to run inference for a new model locally before it's registered on huggingface.co
# for a given Inference Provider, you can add it to the following dictionary.
HARDCODED_MODEL_INFERENCE_MAPPING: dict[str, dict[str, InferenceProviderMapping]] = {
    # "HF model ID" => InferenceProviderMapping object initialized with "Model ID on Inference Provider's side"
    #
    # Example:
    # "Qwen/Qwen2.5-Coder-32B-Instruct": InferenceProviderMapping(hf_model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    #                                    provider_id="Qwen2.5-Coder-32B-Instruct",
    #                                    task="conversational",
    #                                    status="live")
    "cerebras": {},
    "cohere": {},
    "deepinfra": {},
    "fal-ai": {},
    "fireworks-ai": {},
    "groq": {},
    "hf-inference": {},
    "neuronpool": {},
    "nscale": {},
    "ovhcloud": {},
    "replicate": {},
    "scaleway": {},
    "together": {},
    "wavespeed": {},
    "zai-org": {},
}
