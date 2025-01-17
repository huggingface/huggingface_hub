import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from huggingface_hub.utils import get_session

from .base import BaseProvider


@dataclass
class FalAIProvider(BaseProvider):
    BASE_URL = "https://fal.run"
    MODEL_IDS_MAPPING: Dict[str, str] = field(
        default_factory=lambda: {
            # text-to-image
            "black-forest-labs/FLUX.1-schnell": "fal-ai/flux/schnell",
            "black-forest-labs/FLUX.1-dev": "fal-ai/flux/dev",
            # automatic-speech-recognition
            "openai/whisper-large-v3": "fal-ai/whisper",
        }
    )

    def set_custom_headers(self, headers: Dict, **kwargs) -> Dict:
        """."""
        headers["Authorization"] = f"Key {kwargs['token']}"
        return headers

    def build_url(
        self,
        task: Optional[str] = None,
        chat_completion: bool = False,
        model: Optional[str] = None,
    ) -> str:
        return f"{self.BASE_URL}/{model}"

    def prepare_custom_payload(
        self,
        prompt: str,
        model: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        fal.ai models expect inputs in {"prompt": "..."}} format.
        Schema for fal-ai/flux/dev: https://fal.ai/models/fal-ai/flux-lora/api#schema-input
        """
        # TODO: Return payload based on the task
        return {"json": {"prompt": prompt, **kwargs}}

    def get_response(self, response: Union[bytes, Dict], task: Optional[str] = None) -> Any:
        """
        Fetch fal.ai's response
        Schema for fal-ai/flux/dev: https://fal.ai/models/fal-ai/flux-lora/api#schema-output
        """
        if isinstance(response, bytes):
            response_dict = json.loads(response)  # type: ignore
        else:
            response_dict = response
        if task == "text-to-image":
            # We currently support only single image generation per prompt
            url = response_dict["images"][0]["url"]
            return get_session().get(url).content
        return response_dict
