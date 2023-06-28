"""Contains tests for AsyncInferenceClient.

Tests are run directly with pytest instead of unittest.TestCase as it's much easier to run with asyncio.

Not all tasks are tested. We extensively test `text_generation` method since it's the most complex one (has different
return types + uses streaming requests on demand). Tests are mostly duplicates from test_inference_text_generation.py`.

For completeness we also run a test on a simple task (`test_async_sentence_similarity`) and assume all other tasks
work as well.
"""
import pytest

import huggingface_hub.inference._common
from huggingface_hub import AsyncInferenceClient
from huggingface_hub.inference._common import _is_tgi_server
from huggingface_hub.inference._text_generation import FinishReason, InputToken
from huggingface_hub.inference._text_generation import ValidationError as TextGenerationValidationError


@pytest.fixture(autouse=True)
def patch_non_tgi_server(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(huggingface_hub.inference._common, "_NON_TGI_SERVERS", set())


@pytest.fixture
def async_client() -> AsyncInferenceClient:
    return AsyncInferenceClient(model="google/flan-t5-xxl")


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_generate_no_details(async_client: AsyncInferenceClient) -> None:
    response = await async_client.text_generation("test", details=False, max_new_tokens=1)
    assert response == ""


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_generate_with_details(async_client: AsyncInferenceClient) -> None:
    response = await async_client.text_generation("test", details=True, max_new_tokens=1, decoder_input_details=True)

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


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_generate_best_of(async_client: AsyncInferenceClient) -> None:
    response = await async_client.text_generation(
        "test", max_new_tokens=1, best_of=2, do_sample=True, decoder_input_details=True, details=True
    )

    assert response.details.seed is not None
    assert response.details.best_of_sequences is not None
    assert len(response.details.best_of_sequences) == 1
    assert response.details.best_of_sequences[0].seed is not None


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_generate_validation_error(async_client: AsyncInferenceClient) -> None:
    with pytest.raises(TextGenerationValidationError):
        await async_client.text_generation("test", max_new_tokens=10_000)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_generate_non_tgi_endpoint(async_client: AsyncInferenceClient) -> None:
    text = await async_client.text_generation("0 1 2", model="gpt2", max_new_tokens=10)
    assert text == " 3 4 5 6 7 8 9 10 11 12"
    assert not _is_tgi_server("gpt2")

    # Watermark is ignored (+ warning)
    with pytest.warns(UserWarning):
        await async_client.text_generation("4 5 6", model="gpt2", max_new_tokens=10, watermark=True)

    # Return as detail even if details=True (+ warning)
    with pytest.warns(UserWarning):
        text = await async_client.text_generation("0 1 2", model="gpt2", max_new_tokens=10, details=True)
    assert isinstance(text, str)

    # Return as stream raises error
    with pytest.raises(ValueError):
        await async_client.text_generation("0 1 2", model="gpt2", max_new_tokens=10, stream=True)


@pytest.mark.asyncio
async def test_async_generate_stream_no_details(async_client: AsyncInferenceClient) -> None:
    iterator = await async_client.text_generation("test", max_new_tokens=1, stream=True, details=True)
    responses = []
    async for response in iterator:
        responses.append(response)

    assert len(responses) == 1
    response = responses[0]

    assert response.generated_text == ""
    assert response.details.finish_reason == FinishReason.Length
    assert response.details.generated_tokens == 1
    assert response.details.seed is None


@pytest.mark.asyncio
async def test_async_generate_stream_with_details(async_client: AsyncInferenceClient) -> None:
    responses = [
        response
        async for response in await async_client.text_generation("test", max_new_tokens=1, stream=True, details=True)
    ]

    assert len(responses) == 1
    response = responses[0]

    assert response.generated_text == ""
    assert response.details.finish_reason == FinishReason.Length
    assert response.details.generated_tokens == 1
    assert response.details.seed is None


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_sentence_similarity() -> None:
    async_client = AsyncInferenceClient()
    scores = await async_client.sentence_similarity(
        "Machine learning is so easy.",
        other_sentences=[
            "Deep learning is so straightforward.",
            "This is so difficult, like rocket science.",
            "I can't believe how much I struggled with this.",
        ],
    )
    assert scores == [0.7785726189613342, 0.4587625563144684, 0.2906219959259033]
