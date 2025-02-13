from typing import Any, Dict, Optional

from huggingface_hub.inference._providers._common import TaskProviderHelper, filter_none


class SambanovaConversationalTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider="sambanova", base_url="https://api.sambanova.ai", task="conversational")

    def _prepare_route(self, mapped_model: str) -> str:
        return "/v1/chat/completions"

    def _prepare_payload(
        self, inputs: Any, parameters: Dict, mapped_model: str, extra_payload: Optional[Dict] = None
    ) -> Optional[Dict]:
        return {"messages": inputs, **filter_none(parameters), "model": mapped_model, **(extra_payload or {})}
