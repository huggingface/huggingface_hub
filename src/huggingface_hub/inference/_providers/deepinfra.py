import mimetypes
import uuid
from typing import Any

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import MimeBytes, RequestParameters, _as_dict, _open_as_mime_bytes

from ._common import BaseConversationalTask, BaseTextGenerationTask, TaskProviderHelper, filter_none


_PROVIDER = "deepinfra"
_BASE_URL = "https://api.deepinfra.com"


def _encode_multipart(audio: MimeBytes, fields: dict[str, Any]) -> tuple[bytes, str]:
    """Encode an audio file plus text fields as a ``multipart/form-data`` body.

    Returns the raw body bytes and the matching ``Content-Type`` header value
    (including the generated boundary).
    """
    boundary = uuid.uuid4().hex
    filename = "audio" + (mimetypes.guess_extension(audio.mime_type or "") or "")
    lines: list[bytes] = [
        f"--{boundary}".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode(),
        f"Content-Type: {audio.mime_type or 'application/octet-stream'}".encode(),
        b"",
        bytes(audio),
    ]
    for key, value in fields.items():
        lines += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="{key}"'.encode(),
            b"",
            str(value).encode(),
        ]
    lines += [f"--{boundary}--".encode(), b""]
    return b"\r\n".join(lines), f"multipart/form-data; boundary={boundary}"


class DeepInfraTextGenerationTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/openai/completions"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        params = filter_none(parameters.copy())
        params["max_tokens"] = params.pop("max_new_tokens", None)

        return {"prompt": inputs, **params, "model": provider_mapping_info.provider_id}

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        output = _as_dict(response)["choices"][0]
        return {
            "generated_text": output["text"],
            "details": {
                "finish_reason": output.get("finish_reason"),
                "seed": output.get("seed"),
            },
        }


class DeepInfraConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/openai/chat/completions"


class DeepInfraAutomaticSpeechRecognitionTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task="automatic-speech-recognition")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/openai/audio/transcriptions"

    def _prepare_payload_as_bytes(
        self,
        inputs: Any,
        parameters: dict,
        provider_mapping_info: InferenceProviderMapping,
        extra_payload: dict | None,
    ) -> MimeBytes | None:
        # DeepInfra exposes an OpenAI-compatible transcription endpoint, which expects
        # the audio and parameters as a multipart/form-data body (not JSON).
        audio = _open_as_mime_bytes(inputs)
        # `model` is applied last so caller-supplied parameters/extra_payload cannot
        # override the mapped provider model (matches the other DeepInfra helpers).
        fields: dict[str, Any] = {
            **filter_none(parameters),
            **filter_none(extra_payload or {}),
            "model": provider_mapping_info.provider_id,
        }
        body, content_type = _encode_multipart(audio, fields)
        return MimeBytes(body, mime_type=content_type)

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        text = _as_dict(response)["text"]
        if not isinstance(text, str):
            raise ValueError(f"Unexpected output format from DeepInfra API. Expected string, got {type(text)}.")
        return {"text": text}
