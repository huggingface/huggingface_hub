from ._common import BaseConversationalTask


class IteraComputeConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="iteracompute", base_url="https://api.iteracompute.com/hf")
