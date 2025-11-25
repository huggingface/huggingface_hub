from ._common import TaskProviderHelper
from typing import Any, Dict, Optional, Union

class MokzuTextToVideoTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider="mokzu", base_url="https://api.mokzu.com/v1", task="text-to-video")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return f"{self.base_url}/{self.task}"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> Optional[dict]:
        return {"prompt": inputs, **filter_none(parameters)}

    def get_response(self, response: Union[Dict, bytes], request_params: Optional[Dict] = None) -> Any::
        return response["video_url"] if isinstance(response, dict) else {"video_url": ""}

class MokzuImageToVideoTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider="mokzu", base_url="https://api.mokzu.com/v1", task="image-to-video")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return f"{self.base_url}/{self.task}"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> Optional[dict]:
        encoded = base64.b64encode(inputs).decode("utf-8")
        return {"file": encoded, **filter_none(parameters)}

    def get_response(self, response: Union[Dict, bytes], request_params: Optional[Dict] = None) -> Any::
        return response["video_url"] if isinstance(response, dict) else {"video_url": ""}
