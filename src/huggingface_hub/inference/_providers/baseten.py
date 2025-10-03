from ._common import BaseConversationalTask


class BasetenConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="baseten", base_url="https://inference.baseten.co")

