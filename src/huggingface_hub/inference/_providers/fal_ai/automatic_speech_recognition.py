import base64
import json
from typing import Any, Dict, Optional, Union


BASE_URL = "https://fal.run"

SUPPORTED_MODELS = {
    "openai/whisper-large-v3": "fal-ai/whisper",
}


def build_url(model: Optional[str] = None) -> str:
    return f"{BASE_URL}/{model}"


def map_model(model: str) -> str:
    mapped_model = SUPPORTED_MODELS.get(model)
    if mapped_model is None:
        raise ValueError(f"Model {model} is not supported for Fal.AI automatic-speech-recognition task")
    return mapped_model


def prepare_headers(headers: Dict, *, token: Optional[str] = None) -> Dict:
    return {
        **headers,
        "authorization": f"Key {token}",
    }


def prepare_payload(
    inputs: Any,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    if isinstance(inputs, str) and (inputs.startswith("http://") or inputs.startswith("https://")):
        # If input is a URL, pass it directly
        audio_url = inputs
    else:
        # If input is a file path, read it first
        if isinstance(inputs, str):
            with open(inputs, "rb") as f:
                inputs = f.read()

        audio_b64 = base64.b64encode(inputs).decode()
        content_type = "audio/mpeg"
        audio_url = f"data:{content_type};base64,{audio_b64}"

    return {
        "json": {
            "audio_url": audio_url,
            **{k: v for k, v in parameters.items() if v is not None},
        }
    }


def get_response(response: Union[bytes, Dict]) -> Any:
    if isinstance(response, bytes):
        response_dict = json.loads(response)
    else:
        response_dict = response
    if not isinstance(response_dict["text"], str):
        raise ValueError("Unexpected output format from API. Expected string.")
    return response_dict["text"]
