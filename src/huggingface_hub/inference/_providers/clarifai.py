from ._common import BaseConversationalTask


_PROVIDER = "clarifai"
_BASE_URL = "https://api.clarifai.com/v2/ext/openai"


class ClarifaiConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)
