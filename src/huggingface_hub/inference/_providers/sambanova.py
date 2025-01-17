from dataclasses import dataclass, field
from typing import Dict, Optional

from .base import BaseProvider


@dataclass
class SambanovaProvider(BaseProvider):
    BASE_URL = "https://api.sambanova.ai"
    MODEL_IDS_MAPPING: Dict[str, str] = field(
        default_factory=lambda: {
            # chat-completion
            "Qwen/Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B-Instruct",
            "Qwen/QwQ-32B-Preview": "QwQ-32B-Preview",
            "meta-llama/Llama-3.3-70B-Instruct": "Meta-Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.2-1B": "Meta-Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B": "Meta-Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.2-11B-Vision-Instruct": "Llama-3.2-11B-Vision-Instruct",
            "meta-llama/Llama-3.2-90B-Vision-Instruct": "Llama-3.2-90B-Vision-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct": "Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct": "Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Llama-3.1-405B-Instruct": "Meta-Llama-3.1-405B-Instruct",
            "meta-llama/Llama-Guard-3-8B": "Meta-Llama-Guard-3-8B",
        }
    )

    def set_custom_headers(self, headers: Dict, **kwargs) -> Dict:
        return headers

    def build_url(
        self,
        task: Optional[str] = None,
        chat_completion: bool = False,
        model: Optional[str] = None,
    ) -> str:
        if task == "text-generation" and chat_completion:
            return f"{self.BASE_URL}/v1/chat/completions"
        return self.BASE_URL
