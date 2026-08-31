from ._common import BaseConversationalTask


class AiqreConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="aiqre", base_url="https://api.aiqre.com")
