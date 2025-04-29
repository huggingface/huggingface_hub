from typing import Any, Dict, Optional, Union

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import RequestParameters, _as_dict
from huggingface_hub.inference._providers._common import BaseConversationalTask, TaskProviderHelper, filter_none


class SambanovaConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="sambanova", base_url="https://api.sambanova.ai")


class SambanovaFeatureExtractionTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider="sambanova", base_url="https://api.sambanova.ai", task="feature-extraction")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/embeddings"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: Dict, provider_mapping_info: InferenceProviderMapping
    ) -> Optional[Dict]:
        parameters = filter_none(parameters)
        return {"input": inputs, "model": provider_mapping_info.provider_id, **parameters}

    def get_response(self, response: Union[bytes, Dict], request_params: Optional[RequestParameters] = None) -> Any:
        embeddings = _as_dict(response)["data"]
        return [embedding["embedding"] for embedding in embeddings]
