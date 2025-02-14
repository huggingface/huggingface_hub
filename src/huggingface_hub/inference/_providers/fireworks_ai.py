from typing import Any, Dict, Optional

from ._common import TaskProviderHelper, filter_none


class FireworksAIConversationalTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider="fireworks-ai", base_url="https://api.fireworks.ai/inference", task="conversational")

    def _prepare_route(self, mapped_model: str) -> str:
        return "/v1/chat/completions"

    def _prepare_payload_as_dict(self, inputs: Any, parameters: Dict, mapped_model: str) -> Optional[Dict]:
        return {"messages": inputs, **filter_none(parameters), "model": mapped_model}
