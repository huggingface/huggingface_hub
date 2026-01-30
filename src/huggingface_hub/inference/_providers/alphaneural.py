from typing import Any, Optional, Union

from huggingface_hub.inference._common import RequestParameters, _as_dict

from ._common import BaseConversationalTask, BaseTextGenerationTask


_PROVIDER = "alphaneural"
_BASE_URL = "https://proxy.alfnrl.io"


class AlphaneuralConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)


class AlphaneuralTextGenerationTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def get_response(self, response: Union[bytes, dict], request_params: Optional[RequestParameters] = None) -> Any:
        """Convert OpenAI-format response to HuggingFace format."""
        output = _as_dict(response)["choices"][0]
        return {
            "generated_text": output["text"],
            "details": {
                "finish_reason": output.get("finish_reason"),
                "seed": output.get("seed"),
            },
        }
