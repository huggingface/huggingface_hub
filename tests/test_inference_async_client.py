import unittest
from unittest.mock import patch

import pytest

from huggingface_hub import AsyncInferenceClient
from huggingface_hub.inference._common import _NON_TGI_SERVERS
from huggingface_hub.inference._text_generation import (
    FinishReason,
    InputToken,
)
from huggingface_hub.inference._text_generation import (
    ValidationError as TextGenerationValidationError,
)


@pytest.mark.vcr
@patch.dict("huggingface_hub.inference._common._NON_TGI_SERVERS", {})
class TestTextGenerationAsyncClientVCR(unittest.IsolatedAsyncioTestCase):
    """Same as TestTextGenerationClientVCR but with async."""

    def setUp(self) -> None:
        self.client = AsyncInferenceClient(model="google/flan-t5-xxl")
        return super().setUp()

    async def test_generate_no_details(self):
        response = await self.client.text_generation("test", details=False, max_new_tokens=1)

        assert response == ""

    async def test_generate_with_details(self):
        response = await self.client.text_generation(
            "test", details=True, max_new_tokens=1, decoder_input_details=True
        )

        assert response.generated_text == ""
        assert response.details.finish_reason == FinishReason.Length
        assert response.details.generated_tokens == 1
        assert response.details.seed is None
        assert len(response.details.prefill) == 1
        assert response.details.prefill[0] == InputToken(id=0, text="<pad>", logprob=None)
        assert len(response.details.tokens) == 1
        assert response.details.tokens[0].id == 3
        assert response.details.tokens[0].text == " "
        assert not response.details.tokens[0].special

    async def test_generate_best_of(self):
        response = await self.client.text_generation(
            "test", max_new_tokens=1, best_of=2, do_sample=True, decoder_input_details=True, details=True
        )

        assert response.details.seed is not None
        assert response.details.best_of_sequences is not None
        assert len(response.details.best_of_sequences) == 1
        assert response.details.best_of_sequences[0].seed is not None

    async def test_generate_validation_error(self):
        with self.assertRaises(TextGenerationValidationError):
            await self.client.text_generation("test", max_new_tokens=10_000)

    async def test_generate_non_tgi_endpoint(self):
        text = await self.client.text_generation("0 1 2", model="gpt2", max_new_tokens=10)
        self.assertEqual(text, " 3 4 5 6 7 8 9 10 11 12")
        self.assertIn("gpt2", _NON_TGI_SERVERS)

        # Watermark is ignored (+ warning)
        with self.assertWarns(UserWarning):
            await self.client.text_generation("4 5 6", model="gpt2", max_new_tokens=10, watermark=True)

        # Return as detail even if details=True (+ warning)
        with self.assertWarns(UserWarning):
            text = await self.client.text_generation("0 1 2", model="gpt2", max_new_tokens=10, details=True)
            self.assertIsInstance(text, str)

        # Return as stream raises error
        with self.assertRaises(ValueError):
            await self.client.text_generation("0 1 2", model="gpt2", max_new_tokens=10, stream=True)


class TestTextGenerationAsyncClient_NoVCR(unittest.IsolatedAsyncioTestCase):
    """Unfortunately VCR-py doesn't play nicely with aiohttp + stream."""

    def setUp(self) -> None:
        self.client = AsyncInferenceClient(model="google/flan-t5-xxl")
        return super().setUp()

    async def test_generate_stream_no_details(self):
        iterator = await self.client.text_generation("test", max_new_tokens=1, stream=True, details=True)
        responses = []
        async for response in iterator:
            responses.append(response)

        assert len(responses) == 1
        response = responses[0]

        assert response.generated_text == ""
        assert response.details.finish_reason == FinishReason.Length
        assert response.details.generated_tokens == 1
        assert response.details.seed is None

    async def test_generate_stream_with_details(self):
        responses = [
            response
            async for response in await self.client.text_generation(
                "test", max_new_tokens=1, stream=True, details=True
            )
        ]

        assert len(responses) == 1
        response = responses[0]

        assert response.generated_text == ""
        assert response.details.finish_reason == FinishReason.Length
        assert response.details.generated_tokens == 1
        assert response.details.seed is None
