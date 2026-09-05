from ._common import BaseConversationalTask


class BergetConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="berget", base_url="https://api.berget.ai")

    def _prepare_api_key(self, api_key: str | None) -> str:
        if api_key is None:
            raise ValueError("You must provide an api_key to work with Berget API.")
        if api_key.startswith("hf_"):
            raise ValueError(
                "Berget provider is not available through Hugging Face routing yet, please use your own Berget API key."
            )
        return api_key
