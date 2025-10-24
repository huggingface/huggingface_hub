# Original implementation taken from the `text-generation` Python client (see https://pypi.org/project/text-generation/
# and https://github.com/huggingface/text-generation-inference/tree/main/clients/python)
#
# See './src/huggingface_hub/inference/_text_generation.py' for details.
import json
import unittest
from unittest.mock import MagicMock, patch

import pytest

from huggingface_hub import InferenceClient, TextGenerationOutputPrefillToken
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.inference._common import (
    _UNSUPPORTED_TEXT_GENERATION_KWARGS,
    GenerationError,
    IncompleteGenerationError,
    OverloadedError,
    raise_text_generation_error,
)
from huggingface_hub.inference._common import ValidationError as TextGenerationValidationError

from .testing_utils import with_production_testing


class TestTextGenerationErrors(unittest.TestCase):
    def test_generation_error(self):
        error = _mocked_error({"error_type": "generation", "error": "test"})
        with self.assertRaises(GenerationError):
            raise_text_generation_error(error)

    def test_incomplete_generation_error(self):
        error = _mocked_error({"error_type": "incomplete_generation", "error": "test"})
        with self.assertRaises(IncompleteGenerationError):
            raise_text_generation_error(error)

    def test_overloaded_error(self):
        error = _mocked_error({"error_type": "overloaded", "error": "test"})
        with self.assertRaises(OverloadedError):
            raise_text_generation_error(error)

    def test_validation_error(self):
        error = _mocked_error({"error_type": "validation", "error": "test"})
        with self.assertRaises(TextGenerationValidationError):
            raise_text_generation_error(error)


def _mocked_error(payload: dict) -> MagicMock:
    error = HfHubHTTPError("message", response=MagicMock())
    error.response.json.return_value = payload
    return error


@pytest.mark.skip("Temporary skipping TestTextGenerationClientVCR tests")
@with_production_testing
@patch.dict("huggingface_hub.inference._common._UNSUPPORTED_TEXT_GENERATION_KWARGS", {})
class TestTextGenerationClientVCR(unittest.TestCase):
    """Use VCR test to avoid making requests to the prod infra."""

    def setUp(self) -> None:
        self.client = InferenceClient(model="google/flan-t5-xxl")
        return super().setUp()

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
        with self.assertRaises(TextGenerationValidationError):
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
        self.assertEqual(text, " 3 4 5 6 7 8 9 10 11 12")
        self.assertIn("gpt2", _UNSUPPORTED_TEXT_GENERATION_KWARGS)

        # Watermark is ignored (+ warning)
        with self.assertWarns(UserWarning):
            self.client.text_generation("4 5 6", model="gpt2", max_new_tokens=10, watermark=True)

        # Return as detail even if details=True (+ warning)
        with self.assertWarns(UserWarning):
            text = self.client.text_generation("0 1 2", model="gpt2", max_new_tokens=10, details=True)
            self.assertIsInstance(text, str)

        # Return as stream raises error
        with self.assertRaises(ValueError):
            self.client.text_generation("0 1 2", model="gpt2", max_new_tokens=10, stream=True)

    def test_generate_non_tgi_endpoint_regression_test(self):
        # Regression test for https://github.com/huggingface/huggingface_hub/issues/2135
        with self.assertWarnsRegex(UserWarning, "Ignoring following parameters: return_full_text"):
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
