import base64
from typing import Any

from huggingface_hub.hf_api import InferenceProviderMapping
from huggingface_hub.inference._common import RequestParameters, _as_dict

from ._common import BaseConversationalTask, BaseTextGenerationTask, TaskProviderHelper, filter_none


_PROVIDER = "lambdaq"
_BASE_URL = "https://api.lambdaq.org"


class LambdaQConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)


class LambdaQTextGenerationTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL)

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        params = filter_none(parameters.copy())
        # An OpenAI-shaped completions route ignores `max_new_tokens` rather than rejecting it,
        # so leaving it untranslated does not fail the request: it generates to the context
        # limit and bills for every token of it.
        if "max_new_tokens" in params:
            params["max_tokens"] = params.pop("max_new_tokens")
        # `model` is applied last so parameters cannot override the mapped provider model.
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


class LambdaQFeatureExtractionTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task="feature-extraction")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/embeddings"

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


class LambdaQTextToImageTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task="text-to-image")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/images/generations"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        params = filter_none(parameters.copy())
        # The route is OpenAI-shaped, where the output size is one `size` string. A model handed
        # `width`/`height` instead does not fail: it silently returns its default resolution,
        # which on a per-image price is a wasted generation.
        if "width" in params and "height" in params:
            params["size"] = f"{params.pop('width')}x{params.pop('height')}"
        # `model` is applied last so parameters cannot override the mapped provider model.
        return {
            "prompt": inputs,
            **params,
            "model": provider_mapping_info.provider_id,
        }

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        image = _as_dict(response)["data"][0]
        b64_json = image.get("b64_json")
        if b64_json:
            return base64.b64decode(b64_json)
        # Some upstreams hand back a link to the generated image instead of the bytes.
        url = image.get("url")
        if url:
            from huggingface_hub.utils import get_session

            return get_session().get(url).content
        raise ValueError("Unexpected output format from LambdaQ text-to-image API: no image returned.")


class LambdaQTextToSpeechTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(provider=_PROVIDER, base_url=_BASE_URL, task="text-to-speech")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/audio/speech"

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
