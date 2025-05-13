from ._common import BaseConversationalTask, BaseTextGenerationTask


_PROVIDER = "featherless-ai"
_BASE_URL = "https://api.featherless.ai"


class FeatherlessTextGenerationTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)


class FeatherlessConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)
