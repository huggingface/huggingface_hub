from ._common import BaseConversationalTask


class InfraxskyConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="infraxsky", base_url="https://kencloudapi.duckdns.org/hf")
