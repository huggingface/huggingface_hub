from ._common import BaseConversationalTask


class LLMTechConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="llmtech", base_url="https://api.llmtech.eu")
