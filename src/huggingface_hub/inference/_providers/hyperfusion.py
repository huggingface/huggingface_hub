from typing import Optional

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._providers._common import BaseConversationalTask


_PROVIDER = "hyperfusion"
_BASE_URL = "https://api.hyperfusion.io"


class HyperfusionConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_api_key(self, api_key: Optional[str]) -> str:
        if api_key is None:
            raise ValueError(
                "You must provide an api_key to work with Hyperfusion API."
            )
        return api_key

    def _prepare_mapping_info(self, model: Optional[str]) -> InferenceProviderMapping:
        if model is None:
            raise ValueError("Please provide an Hyperfusion model ID, e.g. `llm-en`.")
        return InferenceProviderMapping(
            providerId=model, task="conversational", status="live", hf_model_id=model
        )
