from ._common import BaseConversationalTask


class InfraxskyConversationalTask(BaseConversationalTask):
    # OpenAI-compatible API, reference: https://kencloudapi.duckdns.org/docs
    def __init__(self):
        super().__init__(provider="infraxsky", base_url="https://kencloudapi.duckdns.org/hf")
