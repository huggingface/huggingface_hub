# Original implementation taken from the `text-generation` Python client (see https://pypi.org/project/text-generation/
# and https://github.com/huggingface/text-generation-inference/tree/main/clients/python)
#
# See './src/huggingface_hub/inference/_text_generation.py' for details.
import asyncio
import json
import warnings
from unittest.mock import MagicMock, patch

import httpx2
import pytest

from huggingface_hub import AsyncInferenceClient, InferenceClient, TextGenerationOutputPrefillToken
from huggingface_hub.errors import BadRequestError, HfHubHTTPError
from huggingface_hub.inference._common import (
    _UNSUPPORTED_TEXT_GENERATION_KWARGS,
    GenerationError,
    IncompleteGenerationError,
    OverloadedError,
    _get_unsupported_text_generation_kwargs,
    raise_text_generation_error,
)
from huggingface_hub.inference._common import ValidationError as TextGenerationValidationError


pytestmark = pytest.mark.inference


class TestTextGenerationErrors:
    def test_generation_error(self):
        error = _mocked_error({"error_type": "generation", "error": "test"})
        with pytest.raises(GenerationError):
            raise_text_generation_error(error)

    def test_incomplete_generation_error(self):
        error = _mocked_error({"error_type": "incomplete_generation", "error": "test"})
        with pytest.raises(IncompleteGenerationError):
            raise_text_generation_error(error)

    def test_overloaded_error(self):
        error = _mocked_error({"error_type": "overloaded", "error": "test"})
        with pytest.raises(OverloadedError):
            raise_text_generation_error(error)

    def test_validation_error(self):
        error = _mocked_error({"error_type": "validation", "error": "test"})
        with pytest.raises(TextGenerationValidationError):
            raise_text_generation_error(error)


def _mocked_error(payload: dict) -> MagicMock:
    error = HfHubHTTPError("message", response=MagicMock())
    error.response.json.return_value = payload
    return error


@pytest.mark.skip("Temporary skipping TestTextGenerationClientVCR tests")
@pytest.mark.production
class TestTextGenerationClientVCR:
    """Use VCR test to avoid making requests to the prod infra."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        mocker.patch.dict("huggingface_hub.inference._common._UNSUPPORTED_TEXT_GENERATION_KWARGS", {})
        self.client = InferenceClient(model="google/flan-t5-xxl")
        yield

    def test_generate_no_details(self):
        response = self.client.text_generation("test", details=False, max_new_tokens=1)

        assert response == ""

    def test_generate_with_details(self):
        response = self.client.text_generation("test", details=True, max_new_tokens=1, decoder_input_details=True)

        assert response.generated_text == ""
        assert response.details.finish_reason == "length"
        assert response.details.generated_tokens == 1
        assert response.details.seed is None
        assert len(response.details.prefill) == 1
        assert response.details.prefill[0] == TextGenerationOutputPrefillToken(id=0, text="<pad>", logprob=None)
        assert len(response.details.tokens) == 1
        assert response.details.tokens[0].id == 3
        assert response.details.tokens[0].text == " "
        assert not response.details.tokens[0].special

    def test_generate_best_of(self):
        response = self.client.text_generation(
            "test", max_new_tokens=1, best_of=2, do_sample=True, decoder_input_details=True, details=True
        )

        assert response.details.seed is not None
        assert response.details.best_of_sequences is not None
        assert len(response.details.best_of_sequences) == 1
        assert response.details.best_of_sequences[0].seed is not None

    def test_generate_validation_error(self):
        with pytest.raises(TextGenerationValidationError):
            self.client.text_generation("test", max_new_tokens=10_000)

    def test_generate_stream_no_details(self):
        responses = [
            response for response in self.client.text_generation("test", max_new_tokens=1, stream=True, details=True)
        ]

        assert len(responses) == 1
        response = responses[0]

        assert response.generated_text == ""
        assert response.details.finish_reason == "length"
        assert response.details.generated_tokens == 1
        assert response.details.seed is None

    def test_generate_stream_with_details(self):
        responses = [
            response for response in self.client.text_generation("test", max_new_tokens=1, stream=True, details=True)
        ]

        assert len(responses) == 1
        response = responses[0]

        assert response.generated_text == ""
        assert response.details.finish_reason == "length"
        assert response.details.generated_tokens == 1
        assert response.details.seed is None

    def test_generate_non_tgi_endpoint(self):
        text = self.client.text_generation("0 1 2", model="gpt2", max_new_tokens=10)
        assert text == " 3 4 5 6 7 8 9 10 11 12"
        assert "gpt2" in _UNSUPPORTED_TEXT_GENERATION_KWARGS

        # Watermark is ignored (+ warning)
        with pytest.warns(UserWarning):
            self.client.text_generation("4 5 6", model="gpt2", max_new_tokens=10, watermark=True)

        # Return as detail even if details=True (+ warning)
        with pytest.warns(UserWarning):
            text = self.client.text_generation("0 1 2", model="gpt2", max_new_tokens=10, details=True)
            assert isinstance(text, str)

        # Return as stream raises error
        with pytest.raises(ValueError):
            self.client.text_generation("0 1 2", model="gpt2", max_new_tokens=10, stream=True)

    def test_generate_non_tgi_endpoint_regression_test(self):
        # Regression test for https://github.com/huggingface/huggingface_hub/issues/2135
        with pytest.warns(UserWarning, match="Ignoring following parameters: return_full_text"):
            text = self.client.text_generation(
                prompt="How are you today?", max_new_tokens=20, model="google/flan-t5-large", return_full_text=True
            )
        assert text == "I am at work"

    def test_generate_with_grammar(self):
        # Example taken from https://huggingface.co/docs/text-generation-inference/conceptual/guidance#the-grammar-parameter
        response = self.client.text_generation(
            prompt="I saw a puppy a cat and a raccoon during my bike ride in the park",
            max_new_tokens=100,
            model="HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
            repetition_penalty=1.3,
            grammar={
                "type": "json",
                "value": {
                    "properties": {
                        "location": {"type": "string"},
                        "activity": {"type": "string"},
                        "animals_seen": {"type": "integer", "minimum": 1, "maximum": 5},
                        "animals": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["location", "activity", "animals_seen", "animals"],
                },
            },
        )
        assert json.loads(response) == {
            "activity": "biking",
            "animals": [],
            "animals_seen": 3,
            "location": "park",
        }


class TestTextGenerationUnsupportedKwargs:
    @pytest.fixture(autouse=True)
    def clear_unsupported_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("huggingface_hub.inference._common._UNSUPPORTED_TEXT_GENERATION_KWARGS", {})

    def test_unsupported_kwargs_isolated_by_client_model(self) -> None:
        sent: list[tuple[str, dict]] = []

        def fake_inner_post(client_self, request_parameters, *, stream=False):
            url, payload = request_parameters.url, request_parameters.json
            sent.append((url, dict(payload["parameters"])))
            if url.startswith("http://plain") and "stop" in payload["parameters"]:
                resp = httpx2.Response(400, request=httpx2.Request("POST", url))
                raise BadRequestError(
                    "The following `model_kwargs` are not used by the model: ['stop', 'watermark']",
                    response=resp,
                )
            return json.dumps([{"generated_text": "ok"}]).encode()

        non_tgi_client = InferenceClient(model="http://plain-transformers.local")
        tgi_client = InferenceClient(model="http://tgi.local")
        tgi_async_client = AsyncInferenceClient(model="http://tgi.local")

        async def async_fake_inner_post(client_self, request_parameters, *, stream=False):
            return fake_inner_post(client_self, request_parameters, stream=stream)

        with (
            patch.object(InferenceClient, "_inner_post", fake_inner_post),
            patch.object(AsyncInferenceClient, "_inner_post", async_fake_inner_post),
        ):
            with pytest.warns(UserWarning, match="Ignoring following parameters: stop, watermark"):
                result = non_tgi_client.text_generation("hi", stop=["\n"], watermark=True)

            assert result == "ok"
            # Initial failed request + single retry without unsupported kwargs (no redundant second failure)
            assert len(sent) == 2
            assert sent[1] == ("http://plain-transformers.local", {})
            assert _get_unsupported_text_generation_kwargs("http://plain-transformers.local") == ["stop", "watermark"]
            assert _get_unsupported_text_generation_kwargs(None) == []

            sent.clear()
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                tgi_result = tgi_client.text_generation("hi", stop=["\n"], watermark=True)
                async_tgi_result = asyncio.run(tgi_async_client.text_generation("hi", stop=["\n"], watermark=True))

            assert tgi_result == "ok"
            assert async_tgi_result == "ok"
            assert recorded == []
            assert sent == [
                ("http://tgi.local", {"stop": ["\n"], "watermark": True}),
                ("http://tgi.local", {"stop": ["\n"], "watermark": True}),
            ]

            # Explicit model= override on the non-TGI client should also stay isolated
            sent.clear()
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                override_result = non_tgi_client.text_generation(
                    "hi", model="http://tgi.local", stop=["\n"], watermark=True
                )

            assert override_result == "ok"
            assert recorded == []
            assert sent == [("http://tgi.local", {"stop": ["\n"], "watermark": True})]

    def test_unsupported_kwargs_streaming_isolated_by_client_model(self) -> None:
        sent: list[tuple[str, dict, bool]] = []

        def fake_inner_post(client_self, request_parameters, *, stream=False):
            url, payload = request_parameters.url, request_parameters.json
            sent.append((url, dict(payload["parameters"]), stream))
            if url.startswith("http://plain") and "stop" in payload["parameters"]:
                resp = httpx2.Response(400, request=httpx2.Request("POST", url))
                raise BadRequestError(
                    "The following `model_kwargs` are not used by the model: ['stop', 'watermark']",
                    response=resp,
                )
            if stream:
                return iter(
                    [
                        'data:{"index":0,"token":{"id":1,"text":"ok","logprob":0.0,"special":false},"generated_text":"ok","details":null}',
                        "data: [DONE]",
                    ]
                )
            return json.dumps([{"generated_text": "ok"}]).encode()

        non_tgi_client = InferenceClient(model="http://plain-transformers.local")
        tgi_client = InferenceClient(model="http://tgi.local")

        with patch.object(InferenceClient, "_inner_post", fake_inner_post):
            with pytest.raises(ValueError, match="Cannot return output as a stream"):
                non_tgi_client.text_generation("hi", stream=True, stop=["\n"])
            # First request triggers fallback, retry immediately raises ValueError without a second POST
            assert len(sent) == 1

            sent.clear()
            tokens = list(tgi_client.text_generation("hi", stream=True, stop=["\n"]))
            assert tokens == ["ok"]
            assert sent == [("http://tgi.local", {"stop": ["\n"]}, True)]
