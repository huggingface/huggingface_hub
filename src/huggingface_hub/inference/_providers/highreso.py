from ._common import BaseConversationalTask


class HighresoConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="highreso", base_url="https://api.highreso.ai/hf")
