import base64
from typing import Any, Optional, Union

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import RequestParameters, _as_dict, _as_url
from huggingface_hub.inference._providers._common import TaskProviderHelper, filter_none


class MokzuImageToVideoTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider="mokzu", base_url="https://api.mokzu.com", task="image-to-video")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/image-to-video"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> Optional[dict]:
        # Inputs can be bytes (image data) or dict with image and prompt
        if isinstance(inputs, bytes):
            encoded = base64.b64encode(inputs).decode("utf-8")
            payload = {"image": encoded, **filter_none(parameters)}
        elif isinstance(inputs, dict):
            # For dict input, expect 'image' (bytes or base64) and optional 'prompt'
            image_data = inputs.get("image", "")
            if isinstance(image_data, bytes):
                image_data = base64.b64encode(image_data).decode("utf-8")
            payload = {
                "image": image_data,
                "prompt": inputs.get("prompt", ""),
                **filter_none(parameters)
            }
        else:
            # Assume string (base64 or URL)
            payload = {"image": inputs, **filter_none(parameters)}
        
        # Ensure prompt exists
        if "prompt" not in payload:
            payload["prompt"] = parameters.get("prompt", "")
        
        return payload

    def get_response(self, response: Union[bytes, dict], request_params: Optional[RequestParameters] = None) -> Any:
        response_dict = _as_dict(response)
        video_url = response_dict.get("video_url", "")
        if video_url:
            return _as_url(video_url, default_mime_type="video/mp4")
        raise ValueError("No video_url in response")
