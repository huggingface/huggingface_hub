from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass
class BaseProvider:
    """Base class defining the interface for inference providers."""

    BASE_URL: str = field(init=False)
    MODEL_IDS_MAPPING: Dict[str, str] = field(default_factory=dict, init=False)

    def build_url(
        self,
        task: Optional[str] = None,
        chat_completion: bool = False,
        model: Optional[str] = None,
    ) -> str:
        """Build the URL for the provider"""
        raise NotImplementedError

    def set_custom_headers(self, headers: Dict, **kwargs) -> Dict:
        """Set custom headers for the provider"""
        raise NotImplementedError

    def prepare_custom_payload(
        self,
        prompt: str,
        model: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """Prepare custom payload for the provider"""
        raise NotImplementedError

    def get_response(self, response: Union[bytes, Dict], task: Optional[str] = None) -> Any:
        """Fetch the response from the provider"""
        raise NotImplementedError
