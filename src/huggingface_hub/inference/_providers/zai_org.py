from typing import Any, Dict

from huggingface_hub.inference._providers._common import BaseConversationalTask


class ZaiConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="zai-org", base_url="https://api.z.ai")

    def _prepare_headers(self, headers: Dict, api_key: str) -> Dict[str, Any]:
        headers = super()._prepare_headers(headers, api_key)
        headers["Accept-Language"] = "en-US,en"
        headers["x-source-channel"] = "hugging_face"
        return headers

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/api/paas/v4/chat/completions"
