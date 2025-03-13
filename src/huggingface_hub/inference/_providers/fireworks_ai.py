from typing import Optional

from ._common import BaseConversationalTask


class FireworksAIConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="fireworks-ai", base_url="https://api.fireworks.ai")

    def _prepare_route(self, mapped_model: str, api_key: Optional[str] = None) -> str:
        return "/inference/v1/chat/completions"
