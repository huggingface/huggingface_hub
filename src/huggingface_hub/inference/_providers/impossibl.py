from ._common import BaseConversationalTask


class ImpossiblConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="impossibl", base_url="https://api.impossibl.com")
