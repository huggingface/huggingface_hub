from typing import Any, Dict, Optional

from huggingface_hub.hf_api import InferenceProviderMapping
from ._common import BaseConversationalTask


class GroqConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="groq", base_url="https://api.groq.com")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/openai/v1/chat/completions"

    def _prepare_payload_as_dict(
        self,
        inputs: Any,
        parameters: Dict,
        provider_mapping_info: InferenceProviderMapping,
    ) -> Optional[Dict]:
        payload = super()._prepare_payload_as_dict(
            inputs, parameters, provider_mapping_info
        )

        # Fix tool messages for Groq compatibility
        if payload is not None and "messages" in payload:
            messages = payload.get("messages")
            if messages is not None:
                for message in messages:
                    is_dict = isinstance(message, dict)
                    is_tool_role = message.get("role") == "tool"
                    if is_dict and is_tool_role:
                        # Remove tool_calls field from tool messages
                        message.pop("tool_calls", None)

        return payload if payload is not None else {}
