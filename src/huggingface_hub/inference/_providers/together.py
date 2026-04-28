import base64
import time
from abc import ABC
from typing import Any

from urllib3.filepost import encode_multipart_formdata

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import (
    MimeBytes,
    RequestParameters,
    _as_dict,
    _as_url,
    _open_as_mime_bytes,
)
from huggingface_hub.inference._providers._common import (
    BaseConversationalTask,
    BaseTextGenerationTask,
    TaskProviderHelper,
    filter_none,
)
from huggingface_hub.utils import get_session, hf_raise_for_status, logging


logger = logging.get_logger(__name__)

_PROVIDER = "together"
_BASE_URL = "https://api.together.xyz"

# Polling interval for async video generation (in seconds).
_VIDEO_POLLING_INTERVAL = 2.0


class TogetherTask(TaskProviderHelper, ABC):
    """Base class for Together API tasks."""

    def __init__(self, task: str):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task=task)

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        match self.task:
            case "text-to-image" | "image-to-image":
                return "/v1/images/generations"
            case "text-to-speech":
                return "/v1/audio/speech"
            case "automatic-speech-recognition":
                return "/v1/audio/transcriptions"
            case "feature-extraction":
                return "/v1/embeddings"
            case "text-to-video" | "image-text-to-video":
                # Video creation lives under /v2 (see https://docs.together.ai/reference/create-videos).
                return "/v2/videos"
        raise ValueError(f"Unsupported task '{self.task}' for Together API.")


class TogetherTextGenerationTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        output = _as_dict(response)["choices"][0]
        return {
            "generated_text": output["text"],
            "details": {
                "finish_reason": output.get("finish_reason"),
                "seed": output.get("seed"),
            },
        }


class TogetherConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        payload = super()._prepare_payload_as_dict(inputs, parameters, provider_mapping_info)
        response_format = parameters.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            json_schema_details = response_format.get("json_schema")
            if isinstance(json_schema_details, dict) and "schema" in json_schema_details:
                payload["response_format"] = {  # type: ignore
                    "type": "json_object",
                    "schema": json_schema_details["schema"],
                }

        return payload


class TogetherTextToImageTask(TogetherTask):
    def __init__(self):
        super().__init__("text-to-image")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        mapped_model = provider_mapping_info.provider_id
        parameters = filter_none(parameters)
        if "num_inference_steps" in parameters:
            parameters["steps"] = parameters.pop("num_inference_steps")
        if "guidance_scale" in parameters:
            parameters["guidance"] = parameters.pop("guidance_scale")

        return {"prompt": inputs, "response_format": "base64", **parameters, "model": mapped_model}

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        response_dict = _as_dict(response)
        return base64.b64decode(response_dict["data"][0]["b64_json"])


class TogetherImageToImageTask(TogetherTask):
    def __init__(self):
        super().__init__("image-to-image")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        mapped_model = provider_mapping_info.provider_id
        image_url = _as_url(inputs, default_mime_type="image/jpeg")

        prompt = parameters.pop("prompt", "")
        parameters = filter_none(parameters)
        if "num_inference_steps" in parameters:
            parameters["steps"] = parameters.pop("num_inference_steps")
        if "guidance_scale" in parameters:
            parameters["guidance"] = parameters.pop("guidance_scale")

        return {
            "prompt": prompt,
            "image_url": image_url,
            "response_format": "base64",
            **parameters,
            "model": mapped_model,
        }

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        response_dict = _as_dict(response)
        return base64.b64decode(response_dict["data"][0]["b64_json"])


class TogetherFeatureExtractionTask(TogetherTask):
    def __init__(self):
        super().__init__("feature-extraction")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        return {
            "input": inputs,
            "model": provider_mapping_info.provider_id,
            **filter_none(parameters),
        }

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        return [item["embedding"] for item in _as_dict(response)["data"]]


class TogetherTextToSpeechTask(TogetherTask):
    def __init__(self):
        super().__init__("text-to-speech")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        # `voice` is required by the Together API and is model-specific
        # (see https://docs.together.ai/docs/text-to-speech#supported-voices),
        # so we don't set a default and let the API surface a clear error if missing.
        return {
            "input": inputs,
            "model": provider_mapping_info.provider_id,
            **filter_none(parameters),
        }

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        if isinstance(response, bytes):
            return response
        raise ValueError(f"Expected raw audio bytes for text-to-speech, got {type(response).__name__}.")


class TogetherAutomaticSpeechRecognitionTask(TogetherTask):
    def __init__(self):
        super().__init__("automatic-speech-recognition")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        # The Together /v1/audio/transcriptions endpoint expects multipart/form-data.
        # We build the body in `_prepare_payload_as_bytes` instead.
        return None

    def _prepare_payload_as_bytes(
        self,
        inputs: Any,
        parameters: dict,
        provider_mapping_info: InferenceProviderMapping,
        extra_payload: dict | None,
    ) -> MimeBytes | None:
        fields: dict[str, Any] = {"model": provider_mapping_info.provider_id}

        if isinstance(inputs, str) and inputs.startswith(("http://", "https://")):
            # The API also accepts a public HTTPS URL string in the `file` form field.
            fields["file"] = inputs
        else:
            audio = _open_as_mime_bytes(inputs)
            mime_type = audio.mime_type or "audio/wav"
            extension = mime_type.rsplit("/", 1)[-1]
            fields["file"] = (f"audio.{extension}", bytes(audio), mime_type)

        for key, value in filter_none(parameters or {}).items():
            fields[key] = str(value) if not isinstance(value, (str, bytes, tuple)) else value
        for key, value in (extra_payload or {}).items():
            fields[key] = str(value) if not isinstance(value, (str, bytes, tuple)) else value

        body, content_type = encode_multipart_formdata(fields)
        return MimeBytes(body, mime_type=content_type)

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        return {"text": _as_dict(response).get("text", "")}


class TogetherVideoTask(TogetherTask, ABC):
    """Base class for Together's asynchronous video generation tasks."""

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        if request_params is None:
            raise ValueError("A `RequestParameters` object is required to poll Together video jobs.")

        job = _as_dict(response)
        job_id = job.get("id")
        if not job_id:
            raise ValueError("No job ID found in Together video generation response.")

        # Status polling lives at the same /v2/videos URL with the job ID appended.
        status_url = f"{request_params.url}/{job_id}"

        logger.info("Generating video, polling for completion...")
        status = job.get("status")
        while status == "in_progress":
            time.sleep(_VIDEO_POLLING_INTERVAL)
            status_response = get_session().get(status_url, headers=request_params.headers)
            hf_raise_for_status(status_response)
            job = status_response.json()
            status = job.get("status")

        if status == "failed":
            error = job.get("error") or {}
            raise RuntimeError(f"Together video generation failed: {error.get('message') or 'Unknown error'}")
        if status != "completed":
            raise RuntimeError(f"Unexpected Together video job status: {status!r}")

        video_url = (job.get("outputs") or {}).get("video_url")
        if not video_url:
            raise ValueError("No video URL found in completed Together video job.")

        video_response = get_session().get(video_url)
        hf_raise_for_status(video_response)
        return video_response.content


class TogetherTextToVideoTask(TogetherVideoTask):
    def __init__(self):
        super().__init__("text-to-video")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        return {
            "prompt": inputs,
            "model": provider_mapping_info.provider_id,
            **filter_none(parameters),
        }


class TogetherImageTextToVideoTask(TogetherVideoTask):
    def __init__(self):
        super().__init__("image-text-to-video")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        # Follows the HF `image-text-to-video` schema: image goes in `inputs`, prompt in `parameters.prompt`.
        image_url = _as_url(inputs, default_mime_type="image/jpeg")
        parameters = filter_none(parameters)
        prompt = parameters.pop("prompt", None)

        payload: dict[str, Any] = {
            "model": provider_mapping_info.provider_id,
            "media": {"reference_images": [image_url]},
            **parameters,
        }
        if prompt is not None:
            payload["prompt"] = prompt
        return payload
