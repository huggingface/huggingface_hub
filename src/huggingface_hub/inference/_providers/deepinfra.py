import json
import mimetypes
import time
import uuid
from typing import Any

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import MimeBytes, RequestParameters, _as_dict, _open_as_mime_bytes
from huggingface_hub.utils import get_session, hf_raise_for_status

from ._common import BaseConversationalTask, BaseTextGenerationTask, TaskProviderHelper, filter_none


_PROVIDER = "deepinfra"
_BASE_URL = "https://api.deepinfra.com"


def _form_field_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):  # bool before int: bool is an int subclass
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value)


def _encode_multipart(audio: MimeBytes, fields: dict[str, Any]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    # Fall back to .wav when the MIME type is unknown: transcription servers sniff the format from the filename.
    filename = "audio" + (mimetypes.guess_extension(audio.mime_type or "") or ".wav")
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
            _form_field_value(value).encode(),
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
        # OpenAI-compatible transcription endpoint expects a multipart/form-data body, not JSON.
        audio = _open_as_mime_bytes(inputs)
        # `model` is applied last so parameters cannot override the mapped provider model.
        fields: dict[str, Any] = {
            **filter_none(parameters),
            **filter_none(extra_payload or {}),
            "model": provider_mapping_info.provider_id,
        }
        body, content_type = _encode_multipart(audio, fields)
        return MimeBytes(body, mime_type=content_type)

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        output = _as_dict(response)
        text = output["text"]
        if not isinstance(text, str):
            raise ValueError(f"Unexpected output format from DeepInfra API. Expected string, got {type(text)}.")
        result: dict[str, Any] = {"text": text}
        segments = output.get("segments")
        if isinstance(segments, list):
            result["chunks"] = [
                {"text": segment.get("text"), "timestamp": [segment.get("start"), segment.get("end")]}
                for segment in segments
                if isinstance(segment, dict)
            ]
        return result


class DeepInfraTextToSpeechTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task="text-to-speech")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/openai/audio/speech"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        # `voice` is model-specific and optional; we pass it through and let the API surface a
        # clear error when a model requires one. `model` is applied last so parameters cannot
        # override the mapped provider model.
        return {
            "input": inputs,
            **filter_none(parameters),
            "model": provider_mapping_info.provider_id,
        }

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        if isinstance(response, bytes):
            return response
        raise ValueError(f"Expected raw audio bytes for text-to-speech, got {type(response).__name__}.")


class DeepInfraFeatureExtractionTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task="feature-extraction")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/openai/embeddings"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        # `model` is applied last so parameters cannot override the mapped provider model.
        return {
            "input": inputs,
            **filter_none(parameters),
            "model": provider_mapping_info.provider_id,
        }

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        return [item["embedding"] for item in _as_dict(response)["data"]]


class DeepInfraTextToVideoTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task="text-to-video")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/openai/videos"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        # DeepInfra video generation is asynchronous: this submits the job and get_response
        # polls it. `prompt`/`model` are applied after caller parameters so neither can be overridden.
        return {
            **filter_none(parameters),
            "prompt": inputs,
            "model": provider_mapping_info.provider_id,
        }

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        if request_params is None:
            raise ValueError("A `request_params` object is required to poll DeepInfra text-to-video jobs.")
        job = _as_dict(response)
        job_id = job.get("id")
        if not job_id:
            raise ValueError(f"Unexpected response from DeepInfra text-to-video API: {response!r}")
        session = get_session()
        status_url = f"{request_params.url.rstrip('/')}/{job_id}"
        status = job.get("status")
        while status not in ("succeeded", "failed"):
            time.sleep(2)
            poll = session.get(status_url, headers=request_params.headers)
            hf_raise_for_status(poll)
            job = poll.json()
            status = job.get("status")
        if status == "failed":
            raise ValueError(f"DeepInfra text-to-video job failed: {job.get('error')}")
        data = job.get("data")
        if not (isinstance(data, list) and data and isinstance(data[0], dict) and data[0].get("url")):
            raise ValueError(f"Unexpected response from DeepInfra text-to-video API: {job!r}")
        video = session.get(data[0]["url"])
        hf_raise_for_status(video)
        return video.content
