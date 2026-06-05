from ._common import BaseConversationalTask


class TelnyxConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="telnyx", base_url="https://api.telnyx.com/v2/ai/openai")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/chat/completions"
