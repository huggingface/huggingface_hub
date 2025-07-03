from abc import ABC

from huggingface_hub.inference._providers._common import (
    BaseConversationalTask,
    BaseTextGenerationTask,
    TaskProviderHelper,
)


_PROVIDER = "swarmind"
_BASE_URL = "https://api.swarmind.ai/lai/private"


class SwarmindTask(TaskProviderHelper, ABC):
    """Base class for Swarmind API tasks."""

    def __init__(self, task: str):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task=task)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        if self.task == "conversational":
            return "/v1/chat/completions"
        elif self.task == "text-generation":
            return "/v1/completions"
        raise ValueError(f"Unsupported task '{self.task}' for Swarmind API.")


class SwarmindTextGenerationTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

class SwarmindConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

