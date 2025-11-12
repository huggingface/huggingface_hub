from typing import Any, Optional, Union

from huggingface_hub.inference._common import RequestParameters, _as_dict
from huggingface_hub.inference._providers._common import BaseConversationalTask, BaseTextGenerationTask


_PROVIDER = "ovhcloud"
_BASE_URL = "https://oai.endpoints.kepler.ai.cloud.ovh.net"


class OVHcloudAIEndpointsConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/chat/completions"


class OVHcloudAIEndpointsTextGenerationTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/chat/completions"

    def get_response(self, response: Union[bytes, dict], request_params: Optional[RequestParameters] = None) -> Any:
        output = _as_dict(response)["choices"][0]
        return {
            "generated_text": output["text"],
            "details": {
                "finish_reason": output.get("finish_reason"),
                "seed": output.get("seed"),
            },
        }
