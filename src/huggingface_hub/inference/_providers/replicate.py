import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from huggingface_hub.utils import get_session

from .base import BaseProvider


@dataclass
class ReplicateProvider(BaseProvider):
    BASE_URL = "https://api.replicate.com"
    SUPPORTED_MODELS: Dict[str, str] = field(
        default_factory=lambda: {
            "text-to-image": {
                "black-forest-labs/FLUX.1-schnell": "black-forest-labs/flux-schnell",
                "ByteDance/SDXL-Lightning": "bytedance/sdxl-lightning-4step:5599ed30703defd1d160a25a63321b4dec97101d98b4674bcc56e41f62f35637",
            },
        }
    )

    def set_custom_headers(self, headers: Dict, **kwargs) -> Dict:
        """
        `Prefer` header is used to wait for the response to be ready.
        see: https://replicate.com/docs/reference/http#predictions.create-headers
        """
        headers["Prefer"] = "wait"
        return headers

    def build_url(
        self,
        task: Optional[str] = None,
        chat_completion: bool = False,
        model: Optional[str] = None,
    ) -> str:
        if model is not None and ":" in model:
            return f"{self.BASE_URL}/v1/predictions"
        return f"{self.BASE_URL}/v1/models/{model}/predictions"

    def prepare_custom_payload(
        self,
        prompt: str,
        model: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Most models in replicate expect inputs in {"input": {"prompt": "..."}} format.
        Schema: https://replicate.com/docs/reference/http#predictions.create-request-body
        """
        payload = {"json": {"input": {"prompt": prompt, **kwargs}}}
        if task == "text-to-image":
            if model and ":" in model:
                version = model.split(":", 1)[1]
                payload["json"]["version"] = version
        return payload

    def get_response(self, response: Union[bytes, Dict], task: Optional[str] = None) -> Any:
        """
        Fetch Replicate's response.
        Schema: https://replicate.com/docs/reference/http#predictions.get
        """
        if isinstance(response, bytes):
            response_dict = json.loads(response)  # type: ignore
        else:
            response_dict = response
        if task == "text-to-image":
            # We currently support only single image generation per prompt
            image_url = response_dict["output"][0]
            return get_session().get(image_url).content
        return response_dict
