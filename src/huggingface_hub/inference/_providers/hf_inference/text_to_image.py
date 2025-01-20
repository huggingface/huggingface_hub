from typing import Any, Dict, Optional

from ._common import BaseInferenceTask


class TextToImage(BaseInferenceTask):
    TASK_NAME = "text-to-image"

    @classmethod
    def prepare_payload(
        cls,
        inputs: Any,
        parameters: Dict[str, Any],
        model: Optional[str] = None,
        *,
        expect_binary: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": inputs,
            **parameters,
        }
        return {key: value for key, value in payload.items() if value is not None}


build_url = TextToImage.build_url
map_model = TextToImage.map_model
prepare_headers = TextToImage.prepare_headers
prepare_payload = TextToImage.prepare_payload
get_response = TextToImage.get_response
