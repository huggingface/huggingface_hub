import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from huggingface_hub.inference._common import _b64_to_bytes

from .base import BaseProvider


@dataclass
class TogetherProvider(BaseProvider):
    BASE_URL = "https://api.together.xyz"
    MODEL_IDS_MAPPING: Dict[str, str] = field(
        default_factory=lambda: {
            # text-to-image
            "black-forest-labs/FLUX.1-Canny-dev": "black-forest-labs/FLUX.1-canny",
            "black-forest-labs/FLUX.1-Depth-dev": "black-forest-labs/FLUX.1-depth",
            "black-forest-labs/FLUX.1-dev": "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-Redux-dev": "black-forest-labs/FLUX.1-redux",
            "black-forest-labs/FLUX.1-schnell": "black-forest-labs/FLUX.1-pro",
            "stabilityai/stable-diffusion-xl-base-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
            # chat completion
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
            # text-generation
            "meta-llama/Meta-Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
            "mistralai/Mixtral-8x7B-v0.1": "mistralai/Mixtral-8x7B-v0.1",
        }
    )

    def set_custom_headers(self, headers: Dict, **kwargs) -> Dict:
        return headers

    def build_url(
        self,
        task: Optional[str] = None,
        chat_completion: bool = False,
        model: Optional[str] = None,
    ) -> str:
        if task == "text-to-image":
            return f"{self.BASE_URL}/v1/images/generations"
        if task == "text-generation":
            if chat_completion:
                return f"{self.BASE_URL}/v1/chat/completions"
            return f"{self.BASE_URL}/v1/completions"
        return f"{self.BASE_URL}/v1/chat/completions"

    def prepare_custom_payload(
        self,
        prompt: str,
        model: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Schema for text-to-image: https://docs.together.ai/reference/post_images-generations
        Schema for text-generation and chat-completion: https://docs.together.ai/reference/chat-completions-1
        """
        payload = {"json": {"prompt": prompt, **kwargs}}
        if task == "text-to-image":
            payload["json"].update(
                {
                    "model": model,
                    "response_format": "base64",
                }
            )
        return payload

    def get_response(self, response: Union[bytes, Dict], task: Optional[str] = None) -> Any:
        """
        Fetch Together's response.
        Schema: https://docs.together.ai/reference/post_images-generations
        """
        if isinstance(response, bytes):
            response_dict = json.loads(response)  # type: ignore
        else:
            response_dict = response
        if task == "text-to-image":
            # We currently support only single image generation per prompt
            return _b64_to_bytes(response_dict["data"][0]["b64_json"])
        return response_dict
