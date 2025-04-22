# Original implementation taken from the `text-generation` Python client (see https://pypi.org/project/text-generation/
# and https://github.com/huggingface/text-generation-inference/tree/main/clients/python)
#
# See './src/huggingface_hub/inference/_text_generation.py' for details.
import unittest
from typing import Dict
from unittest.mock import MagicMock

from requests import HTTPError

from huggingface_hub.inference._common import (
    GenerationError,
    IncompleteGenerationError,
    OverloadedError,
    raise_text_generation_error,
)
from huggingface_hub.inference._common import ValidationError as TextGenerationValidationError


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


def _mocked_error(payload: Dict) -> MagicMock:
    error = HTTPError(response=MagicMock())
    error.response.json.return_value = payload
    return error
