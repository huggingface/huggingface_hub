from huggingface_hub.inference._providers._common import (
    BaseConversationalTask,
    BaseTextGenerationTask,
)


_PROVIDER = "swarmind"
_BASE_URL = "https://api.swarmind.ai"

class SwarmindConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/lai/private/v1/chat/completions"

class SwarmindTextGenerationTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/lai/private/v1/chat/completions"
