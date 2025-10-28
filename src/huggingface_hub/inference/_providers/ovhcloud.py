import base64
from abc import ABC
from typing import Any, Dict, Optional, Union

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import RequestParameters, _as_dict
from huggingface_hub.inference._providers._common import (
    TaskProviderHelper,
    filter_none,
)

_PROVIDER = "ovhcloud"
_BASE_URL = "https://oai.endpoints.kepler.ai.cloud.ovh.net"

class OVHcloudAIEndpointsTask(TaskProviderHelper, ABC):
    def __init__(self, task: str):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task=task)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        if self.task == "text-to-image":
            return "/v1/images/generations"
        elif self.task == "conversational":
            return "/v1/chat/completions"
        elif self.task == "feature-extraction":
            return "/v1/embeddings"
        elif self.task == "automatic-speech-recognition":
            return "/v1/audio/transcriptions"
        raise ValueError(f"Unsupported task '{self.task}' for OVHcloud AI Endpoints.")
    
    def _prepare_payload_as_dict(
        self, messages: Any, parameters: Dict, provider_mapping_info: InferenceProviderMapping
    ) -> Optional[Dict]:
        return {"messages": messages, "model": provider_mapping_info.provider_id, **filter_none(parameters)}


class OVHcloudAIEndpointsConversationalTask(OVHcloudAIEndpointsTask):
    def __init__(self):
        super().__init__("conversational")

    def _prepare_payload_as_dict(
        self, messages: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> Optional[dict]:
        return super()._prepare_payload_as_dict(messages, parameters, provider_mapping_info)
        

class OVHcloudAIEndpointsTextToImageTask(OVHcloudAIEndpointsTask):
    def __init__(self):
        super().__init__("text-to-image")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> Optional[dict]:
        mapped_model = provider_mapping_info.provider_id
        return {"prompt": inputs, "model": mapped_model, **filter_none(parameters)}

    def get_response(self, response: Union[bytes, dict], request_params: Optional[RequestParameters] = None) -> Any:
        response_dict = _as_dict(response)
        return base64.b64decode(response_dict["data"][0]["b64_json"])
    
class OVHcloudAIEndpointsFeatureExtractionTask(OVHcloudAIEndpointsTask):
    def __init__(self):
        super().__init__("feature-extraction")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: Dict, provider_mapping_info: InferenceProviderMapping
    ) -> Optional[Dict]:
        return {"input": inputs, "model": provider_mapping_info.provider_id, **filter_none(parameters)}
    
    def get_response(self, response: Union[bytes, dict], request_params: Optional[RequestParameters] = None) -> Any:
        embeddings = _as_dict(response)["data"]
        return [embedding["embedding"] for embedding in embeddings]
    
class OVHcloudAIEndpointsAutomaticSpeechRecognitionTask(OVHcloudAIEndpointsTask):
    def __init__(self):
        super().__init__("automatic-speech-recognition")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> Optional[dict]:
        return {"file": inputs, "model": provider_mapping_info.provider_id, **filter_none(parameters)}

    def get_response(self, response: Union[bytes, dict], request_params: Optional[RequestParameters] = None) -> Any:
        response_dict = _as_dict(response)
        return response_dict["text"]
