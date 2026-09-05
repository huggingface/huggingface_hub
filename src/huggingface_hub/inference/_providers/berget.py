from ._common import BaseConversationalTask


class BergetConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="berget", base_url="https://api.berget.ai")
