from typing import Any, Dict, Optional, Union

from huggingface_hub.inference._common import RequestParameters, TaskProviderHelper
from huggingface_hub.utils import build_hf_headers


BASE_URL = "https://api.sambanova.ai"

SUPPORTED_MODELS = {
    "conversational": {
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
    },
}


class SambanovaConversationalTask(TaskProviderHelper):
    def __init__(self):
        # TODO: adapt in a base class when supporting multiple tasks
        self.task = "conversational"

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
        mapped_model = self._map_model(model)

        if api_key is None:
            raise ValueError("You must provide an api_key to work with Sambanova API.")

        headers = {**build_hf_headers(token=api_key), **headers}

        payload = {
            "messages": inputs,
            **{k: v for k, v in parameters.items() if v is not None},
            "model": mapped_model,
        }

        return RequestParameters(
            url=f"{BASE_URL}/v1/chat/completions",
            task=self.task,
            model=mapped_model,
            json=payload,
            data=None,
            headers=headers,
        )

    def _map_model(self, model: Optional[str]) -> str:
        if model is None:
            raise ValueError("Please provide a model available on Sambanova.")
        if self.task not in SUPPORTED_MODELS:
            raise ValueError(f"Task {self.task} not supported with Sambanova.")
        mapped_model = SUPPORTED_MODELS[self.task].get(model)
        if mapped_model is None:
            raise ValueError(f"Model {model} is not supported with Sambanova for task {self.task}.")
        return mapped_model

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        return response
