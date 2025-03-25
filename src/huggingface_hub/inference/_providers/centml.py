from typing import Optional

from huggingface_hub.inference._providers._common import (
    BaseConversationalTask,
    BaseTextGenerationTask,
)


class CentmlConversationalTask(BaseConversationalTask):
    """
    Provider helper for centml conversational (chat completions) tasks.
    This helper builds requests in the OpenAI API format.
    """

    def __init__(self):
        # Set the provider name to "centml" and use the centml serverless endpoint URL.
        super().__init__(provider="centml", base_url="https://api.centml.com/openai")

    def _prepare_api_key(self, api_key: Optional[str]) -> str:
        if api_key is None:
            raise ValueError(
                "An API key must be provided to use the centml provider.")
        return api_key

    def _prepare_mapped_model(self, model: Optional[str]) -> str:
        if model is None:
            raise ValueError("Please provide a centml model ID.")
        return model


class CentmlTextGenerationTask(BaseTextGenerationTask):
    """
    Provider helper for centml text generation (completions) tasks.
    This helper builds requests in the OpenAI API format.
    """

    def __init__(self):
        super().__init__(provider="centml", base_url="https://api.centml.com/openai")

    def _prepare_api_key(self, api_key: Optional[str]) -> str:
        if api_key is None:
            raise ValueError(
                "An API key must be provided to use the centml provider.")
        return api_key

    def _prepare_mapped_model(self, model: Optional[str]) -> str:
        if model is None:
            raise ValueError("Please provide a centml model ID.")
        return model

