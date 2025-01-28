import base64
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from huggingface_hub import constants
from huggingface_hub.inference._common import RequestParameters, TaskProviderHelper, _as_dict
from huggingface_hub.utils import build_hf_headers, get_token, logging


logger = logging.get_logger(__name__)


BASE_URL = "https://api.together.xyz"

SUPPORTED_MODELS = {
    "conversational": {
        "databricks/dbrx-instruct": "databricks/dbrx-instruct",
        "deepseek-ai/DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/deepseek-llm-67b-chat": "deepseek-ai/deepseek-llm-67b-chat",
        "google/gemma-2-9b-it": "google/gemma-2-9b-it",
        "google/gemma-2b-it": "google/gemma-2-27b-it",
        "meta-llama/Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-3.2-11B-Vision-Instruct": "meta-llama/Llama-Vision-Free",
        "meta-llama/Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "meta-llama/Llama-3.2-90B-Vision-Instruct": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "meta-llama/Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3-70B-Instruct": "meta-llama/Llama-3-70b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-405B-Instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
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
    },
    "text-generation": {
        "meta-llama/Llama-2-70b-hf": "meta-llama/Llama-2-70b-hf",
        "meta-llama/Meta-Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
        "mistralai/Mixtral-8x7B-v0.1": "mistralai/Mixtral-8x7B-v0.1",
    },
    "text-to-image": {
        "black-forest-labs/FLUX.1-Canny-dev": "black-forest-labs/FLUX.1-canny",
        "black-forest-labs/FLUX.1-Depth-dev": "black-forest-labs/FLUX.1-depth",
        "black-forest-labs/FLUX.1-dev": "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-Redux-dev": "black-forest-labs/FLUX.1-redux",
        "black-forest-labs/FLUX.1-schnell": "black-forest-labs/FLUX.1-pro",
        "stabilityai/stable-diffusion-xl-base-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    },
}


PER_TASK_ROUTES = {
    "conversational": "v1/chat/completions",
    "text-generation": "v1/completions",
    "text-to-image": "v1/images/generations",
}


class TogetherTask(TaskProviderHelper, ABC):
    """Base class for Together API tasks."""

    def __init__(self, task: str):
        self.task = task

    def prepare_request(
        self,
        *,
        inputs: Any,
        parameters: Dict[str, Any],
        headers: Dict,
        model: Optional[str],
        api_key: Optional[str],
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> RequestParameters:
        if api_key is None:
            api_key = get_token()
        if api_key is None:
            raise ValueError(
                "You must provide an api_key to work with Together API or log in with `huggingface-cli login`."
            )
        headers = {**build_hf_headers(token=api_key), **headers}

        # Route to the proxy if the api_key is a HF TOKEN
        if api_key.startswith("hf_"):
            base_url = constants.INFERENCE_PROXY_TEMPLATE.format(provider="together")
            logger.info("Calling Together provider through Hugging Face proxy.")
        else:
            base_url = BASE_URL
            logger.info("Calling Together provider directly.")
        mapped_model = self._map_model(model)
        if "model" in parameters:
            parameters["model"] = mapped_model
        payload = self._prepare_payload(inputs, parameters=parameters)

        return RequestParameters(
            url=f"{base_url}/{PER_TASK_ROUTES[self.task]}",
            task=self.task,
            model=mapped_model,
            json=payload,
            data=None,
            headers=headers,
        )

    def _map_model(self, model: Optional[str]) -> str:
        if model is None:
            raise ValueError("Please provide a model available on Together.")
        if self.task not in SUPPORTED_MODELS:
            raise ValueError(f"Task {self.task} not supported with Together.")
        mapped_model = SUPPORTED_MODELS[self.task].get(model)
        if mapped_model is None:
            raise ValueError(f"Model {model} is not supported with Together for task {self.task}.")
        return mapped_model

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        return response

    @abstractmethod
    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]: ...


class TogetherTextGenerationTask(TogetherTask):
    # Handle both "text-generation" and "conversational"
    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": inputs, **{k: v for k, v in parameters.items() if v is not None}}


class TogetherTextToImageTask(TogetherTask):
    def __init__(self):
        super().__init__("text-to-image")

    def _prepare_payload(self, inputs: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "prompt": inputs,
            "response_format": "base64",
            **{k: v for k, v in parameters.items() if v is not None},
        }
        return payload

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        response_dict = _as_dict(response)
        return base64.b64decode(response_dict["data"][0]["b64_json"])
