from typing import Any, Dict, Optional, Union

from huggingface_hub import constants
from huggingface_hub.inference._common import RequestParameters, TaskProviderHelper, _as_dict, _get_provider_model_id
from huggingface_hub.utils import build_hf_headers, get_session, get_token, logging


logger = logging.get_logger(__name__)


BASE_URL = "https://api.replicate.com"


def _build_url(base_url: str, model: str) -> str:
    if ":" in model:
        return f"{base_url}/v1/predictions"
    return f"{base_url}/v1/models/{model}/predictions"


class ReplicateTask(TaskProviderHelper):
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
        conversational: bool = False,
    ) -> RequestParameters:
        if api_key is None:
            api_key = get_token()
        if api_key is None:
            raise ValueError(
                "You must provide an api_key to work with Replicate API or log in with `huggingface-cli login`."
            )

        # Route to the proxy if the api_key is a HF TOKEN
        if api_key.startswith("hf_"):
            base_url = constants.INFERENCE_PROXY_TEMPLATE.format(provider="replicate")
            logger.info("Calling Replicate provider through Hugging Face proxy.")
        else:
            base_url = BASE_URL
            logger.info("Calling Replicate provider directly.")

        mapped_model = self.map_model(model=model, task=self.task, conversational=conversational)
        url = _build_url(base_url, mapped_model)

        headers = {
            **build_hf_headers(token=api_key),
            **headers,
            "Prefer": "wait",
        }

        payload = self._prepare_payload(inputs, parameters=parameters, model=mapped_model)

        return RequestParameters(
            url=url,
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
            raise ValueError("Please provide a HF model ID supported by Replicate.")
        return _get_provider_model_id(model, "replicate", task, conversational)

    def _prepare_payload(
        self,
        inputs: Any,
        parameters: Dict[str, Any],
        model: str,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "input": {
                "prompt": inputs,
                **{k: v for k, v in parameters.items() if v is not None},
            }
        }
        if ":" in model:
            version = model.split(":", 1)[1]
            payload["version"] = version
        return payload

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        response_dict = _as_dict(response)
        if response_dict.get("output") is None:
            raise TimeoutError(
                f"Inference request timed out after 60 seconds. No output generated for model {response_dict.get('model')}"
                "The model might be in cold state or starting up. Please try again later."
            )
        output_url = (
            response_dict["output"] if isinstance(response_dict["output"], str) else response_dict["output"][0]
        )
        return get_session().get(output_url).content


class ReplicateTextToSpeechTask(ReplicateTask):
    def __init__(self):
        super().__init__("text-to-speech")

    def _prepare_payload(
        self,
        inputs: Any,
        parameters: Dict[str, Any],
        model: str,
    ) -> Dict[str, Any]:
        # The following payload might work only for a subset of text-to-speech Replicate models.
        payload: Dict[str, Any] = {
            "input": {
                "text": inputs,
                **{k: v for k, v in parameters.items() if v is not None},
            },
        }
        if ":" in model:
            version = model.split(":", 1)[1]
            payload["version"] = version
        return payload
