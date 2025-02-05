from typing import Any, Dict, Optional, Union

from huggingface_hub import constants
from huggingface_hub.inference._common import RequestParameters, TaskProviderHelper, _get_provider_model_id
from huggingface_hub.utils import build_hf_headers, get_token, logging


logger = logging.get_logger(__name__)


BASE_URL = "https://api.sambanova.ai"


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
        conversational: bool = False,
    ) -> RequestParameters:
        if api_key is None:
            api_key = get_token()
        if api_key is None:
            raise ValueError(
                "You must provide an api_key to work with Sambanova API or log in with `huggingface-cli login`."
            )

        # Route to the proxy if the api_key is a HF TOKEN
        if api_key.startswith("hf_"):
            base_url = constants.INFERENCE_PROXY_TEMPLATE.format(provider="sambanova")
            logger.info("Calling Sambanova provider through Hugging Face proxy.")
        else:
            base_url = BASE_URL
            logger.info("Calling Sambanova provider directly.")
        headers = {**build_hf_headers(token=api_key), **headers}

        mapped_model = self.map_model(model=model, task=self.task, conversational=conversational)
        payload = {
            "messages": inputs,
            **{k: v for k, v in parameters.items() if v is not None},
            "model": mapped_model,
        }

        return RequestParameters(
            url=f"{base_url}/v1/chat/completions",
            task=self.task,
            model=mapped_model,
            json=payload,
            data=None,
            headers=headers,
        )

    def map_model(
        self,
        model: Optional[str],
        task: str,
        conversational: bool = False,
    ) -> str:
        """Default implementation for mapping model HF model IDs to provider model IDs."""
        if model is None:
            raise ValueError("Please provide a HF model ID supported by Sambanova.")
        return _get_provider_model_id(model, "sambanova", task, conversational)

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        return response
