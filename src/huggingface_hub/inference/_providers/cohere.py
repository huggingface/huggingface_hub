from typing import Any, Dict, Union

from huggingface_hub.inference._common import _as_dict
from huggingface_hub.inference._providers._common import (
    BaseConversationalTask,
)


_PROVIDER = "cohere"
_BASE_URL = "https://api.cohere.com"


class CohereConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_route(self, mapped_model: str) -> str:
        if self.task == "conversational":
            return "/v2/chat"
        raise ValueError(f"Unsupported task '{self.task}' for Cohere API.")

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        cohere_dict = _as_dict(response)
        # Build response compatible with ChatCompletionOutput
        response_dict = {
            "choices": [
                {
                    "finish_reason": cohere_dict.get("finish_reason", "").lower(),
                    "index": 0,
                    "message": {
                        "role": cohere_dict.get("message", {}).get("role", "assistant"),
                        "content": self._extract_content(cohere_dict.get("message", {}).get("content", [])),
                    },
                    "logprobs": None,
                }
            ],
            "created": int(cohere_dict.get("created", 0)) if cohere_dict.get("created") else 0,
            "id": cohere_dict.get("id", ""),
            "model": cohere_dict.get("model", ""),
            "system_fingerprint": cohere_dict.get("system_fingerprint", ""),
            "usage": {
                "completion_tokens": cohere_dict.get("usage", {}).get("tokens", {}).get("output_tokens", 0),
                "prompt_tokens": cohere_dict.get("usage", {}).get("tokens", {}).get("input_tokens", 0),
                "total_tokens": (
                    cohere_dict.get("usage", {}).get("tokens", {}).get("input_tokens", 0)
                    + cohere_dict.get("usage", {}).get("tokens", {}).get("output_tokens", 0)
                ),
            },
        }

        return response_dict

    def _extract_content(self, content_list):
        """Extract text content from Cohere's content list format."""
        if not content_list:
            return ""

        # If content is a list of content blocks (e.g., text blocks)
        if isinstance(content_list, list):
            text_parts = []
            for item in content_list:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
            return "".join(text_parts)

        # If content is already a string
        return content_list
