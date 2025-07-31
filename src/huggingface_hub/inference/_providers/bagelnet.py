from ._common import BaseConversationalTask


class BagelNetConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="bagelnet", base_url="https://api.bagel.net") 