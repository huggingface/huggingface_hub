from typing import Any, Dict, Optional

from ._common import BaseInferenceTask, get_recommended_model


class Conversational(BaseInferenceTask):
    TASK_NAME = "text-generation"

    @classmethod
    def build_url(cls, model: Optional[str] = None) -> str:
        if model is None:
            model = get_recommended_model(cls.TASK_NAME)
        url = f"{cls.BASE_URL}/models/{model}"
        url = url.rstrip("/")
        if url.endswith("/v1"):
            url += "/chat/completions"
        elif not url.endswith("/chat/completions"):
            url += "/v1/chat/completions"
        return url

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


build_url = Conversational.build_url
map_model = Conversational.map_model
prepare_headers = Conversational.prepare_headers
prepare_payload = Conversational.prepare_payload
get_response = Conversational.get_response
