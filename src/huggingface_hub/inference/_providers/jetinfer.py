from ._common import BaseConversationalTask


class JetInferConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="jetinfer", base_url="https://api.jetinfer.com")
