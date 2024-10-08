# coding=utf-8
# Copyright 2023-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains tests for AsyncInferenceClient.

Tests are run directly with pytest instead of unittest.TestCase as it's much easier to run with asyncio.

Not all tasks are tested. We extensively test `text_generation` method since it's the most complex one (has different
return types + uses streaming requests on demand). Tests are mostly duplicates from test_inference_text_generation.py`.

For completeness we also run a test on a simple task (`test_async_sentence_similarity`) and assume all other tasks
work as well.
"""

import asyncio
import inspect
from unittest.mock import Mock, patch

import pytest
from aiohttp import ClientResponseError

import huggingface_hub.inference._common
from huggingface_hub import (
    AsyncInferenceClient,
    ChatCompletionOutput,
    ChatCompletionOutputComplete,
    ChatCompletionOutputMessage,
    ChatCompletionOutputUsage,
    ChatCompletionStreamOutput,
    InferenceClient,
    InferenceTimeoutError,
    TextGenerationOutputPrefillToken,
)
from huggingface_hub.inference._common import ValidationError as TextGenerationValidationError
from huggingface_hub.inference._common import _get_unsupported_text_generation_kwargs

from .test_inference_client import CHAT_COMPLETE_NON_TGI_MODEL, CHAT_COMPLETION_MESSAGES, CHAT_COMPLETION_MODEL
from .testing_utils import with_production_testing


@pytest.fixture(autouse=True)
def patch_non_tgi_server(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(huggingface_hub.inference._common, "_UNSUPPORTED_TEXT_GENERATION_KWARGS", {})


@pytest.fixture
def tgi_client() -> AsyncInferenceClient:
    return AsyncInferenceClient(model="google/flan-t5-xxl")


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_generate_no_details(tgi_client: AsyncInferenceClient) -> None:
    response = await tgi_client.text_generation("test", details=False, max_new_tokens=1)
    assert response == ""


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_generate_with_details(tgi_client: AsyncInferenceClient) -> None:
    response = await tgi_client.text_generation("test", details=True, max_new_tokens=1, decoder_input_details=True)

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


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_generate_best_of(tgi_client: AsyncInferenceClient) -> None:
    response = await tgi_client.text_generation(
        "test", max_new_tokens=1, best_of=2, do_sample=True, decoder_input_details=True, details=True
    )

    assert response.details.seed is not None
    assert response.details.best_of_sequences is not None
    assert len(response.details.best_of_sequences) == 1
    assert response.details.best_of_sequences[0].seed is not None


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_generate_validation_error(tgi_client: AsyncInferenceClient) -> None:
    with pytest.raises(TextGenerationValidationError):
        await tgi_client.text_generation("test", max_new_tokens=10_000)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_generate_non_tgi_endpoint(tgi_client: AsyncInferenceClient) -> None:
    text = await tgi_client.text_generation("0 1 2", model="gpt2", max_new_tokens=10)
    assert text == " 3 4 5 6 7 8 9 10 11 12"
    assert _get_unsupported_text_generation_kwargs("gpt2") == ["details", "stop", "watermark", "decoder_input_details"]

    # Watermark is ignored (+ warning)
    with pytest.warns(UserWarning):
        await tgi_client.text_generation("4 5 6", model="gpt2", max_new_tokens=10, watermark=True)

    # Return as detail even if details=True (+ warning)
    with pytest.warns(UserWarning):
        text = await tgi_client.text_generation("0 1 2", model="gpt2", max_new_tokens=10, details=True)
    assert isinstance(text, str)

    # Return as stream raises error
    with pytest.raises(ValueError):
        await tgi_client.text_generation("0 1 2", model="gpt2", max_new_tokens=10, stream=True)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_generate_stream_no_details(tgi_client: AsyncInferenceClient) -> None:
    responses = [
        response async for response in await tgi_client.text_generation("test", max_new_tokens=1, stream=True)
    ]

    assert len(responses) == 1
    response = responses[0]

    assert isinstance(response, str)
    assert response == " "


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_generate_stream_with_details(tgi_client: AsyncInferenceClient) -> None:
    responses = [
        response
        async for response in await tgi_client.text_generation("test", max_new_tokens=1, stream=True, details=True)
    ]

    assert len(responses) == 1
    response = responses[0]

    assert response.generated_text == ""
    assert response.details.finish_reason == "length"
    assert response.details.generated_tokens == 1
    assert response.details.seed is None


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_chat_completion_no_stream() -> None:
    async_client = AsyncInferenceClient(model=CHAT_COMPLETION_MODEL)
    output = await async_client.chat_completion(CHAT_COMPLETION_MESSAGES, max_tokens=10)
    assert isinstance(output.created, int)
    assert output == ChatCompletionOutput(
        id="",
        model="HuggingFaceH4/zephyr-7b-beta",
        system_fingerprint="1.4.3-sha-e6bb3ff",
        usage=ChatCompletionOutputUsage(completion_tokens=10, prompt_tokens=47, total_tokens=57),
        choices=[
            ChatCompletionOutputComplete(
                finish_reason="length",
                index=0,
                message=ChatCompletionOutputMessage(
                    content="Deep learning is a subfield of machine learning that",
                    role="assistant",
                ),
            )
        ],
        created=output.created,
    )


@pytest.mark.vcr
@pytest.mark.asyncio
@with_production_testing
async def test_async_chat_completion_not_tgi_no_stream() -> None:
    async_client = AsyncInferenceClient(model=CHAT_COMPLETE_NON_TGI_MODEL)
    output = await async_client.chat_completion(CHAT_COMPLETION_MESSAGES, max_tokens=10)
    assert isinstance(output.created, int)
    assert output == ChatCompletionOutput(
        choices=[
            ChatCompletionOutputComplete(
                finish_reason="eos_token",
                index=0,
                message=ChatCompletionOutputMessage(role="assistant", content="Hello, advisor.", tool_calls=None),
                logprobs=None,
            )
        ],
        created=1721741143,
        id="",
        model="microsoft/DialoGPT-small",
        system_fingerprint="2.1.1-sha-4dfdb48",
        usage=ChatCompletionOutputUsage(completion_tokens=5, prompt_tokens=13, total_tokens=18),
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_chat_completion_with_stream() -> None:
    async_client = AsyncInferenceClient(model=CHAT_COMPLETION_MODEL)
    output = await async_client.chat_completion(CHAT_COMPLETION_MESSAGES, max_tokens=10, stream=True)

    all_items = [item async for item in output]
    generated_text = ""
    for item in all_items:
        assert isinstance(item, ChatCompletionStreamOutput)
        assert len(item.choices) == 1
        generated_text += item.choices[0].delta.content
    last_item = all_items[-1]

    assert generated_text == "Deep learning is a subfield of machine learning that"

    # Last item has a finish reason
    assert last_item.choices[0].finish_reason == "length"


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


def test_sync_vs_async_signatures() -> None:
    client = InferenceClient()
    async_client = AsyncInferenceClient()

    # Some methods have to be tested separately.
    special_methods = ["post", "text_generation", "chat_completion"]

    # Post: this is not automatically tested. No need to test its signature separately.

    # text-generation/chat-completion: return type changes from Iterable[...] to AsyncIterable[...] but input parameters are the same
    for name in ["text_generation", "chat_completion"]:
        sync_method = getattr(client, name)
        assert not inspect.iscoroutinefunction(sync_method)
        async_method = getattr(async_client, name)
        assert inspect.iscoroutinefunction(async_method)

        sync_sig = inspect.signature(sync_method)
        async_sig = inspect.signature(async_method)
        assert sync_sig.parameters == async_sig.parameters
        assert sync_sig.return_annotation != async_sig.return_annotation

    # Check that all methods are consistent between InferenceClient and AsyncInferenceClient
    for name in dir(client):
        if not inspect.ismethod(getattr(client, name)):  # not a method
            continue
        if name.startswith("_"):  # not public method
            continue
        if name in special_methods:  # tested separately
            continue

        # Check that the sync method is not async
        sync_method = getattr(client, name)
        assert not inspect.iscoroutinefunction(sync_method)

        # Check that the async method is async
        async_method = getattr(async_client, name)
        assert inspect.iscoroutinefunction(async_method)

        # Check that expected inputs and outputs are the same
        sync_sig = inspect.signature(sync_method)
        async_sig = inspect.signature(async_method)
        assert sync_sig.parameters == async_sig.parameters
        assert sync_sig.return_annotation == async_sig.return_annotation


@pytest.mark.asyncio
async def test_get_status_too_big_model() -> None:
    model_status = await AsyncInferenceClient().get_model_status("facebook/nllb-moe-54b")
    assert model_status.loaded is False
    assert model_status.state == "TooBig"
    assert model_status.compute_type == "cpu"
    assert model_status.framework == "transformers"


@pytest.mark.asyncio
async def test_get_status_loaded_model() -> None:
    model_status = await AsyncInferenceClient().get_model_status("bigscience/bloom")
    assert model_status.loaded is True
    assert model_status.state == "Loaded"
    assert isinstance(model_status.compute_type, dict)  # e.g. {'gpu': {'gpu': 'a100', 'count': 8}}
    assert model_status.framework == "text-generation-inference"


@pytest.mark.asyncio
async def test_get_status_unknown_model() -> None:
    with pytest.raises(ClientResponseError):
        await AsyncInferenceClient().get_model_status("unknown/model")


@pytest.mark.asyncio
async def test_get_status_model_as_url() -> None:
    with pytest.raises(NotImplementedError):
        await AsyncInferenceClient().get_model_status("https://unkown/model")


@pytest.mark.asyncio
async def test_list_deployed_models_single_frameworks() -> None:
    models_by_task = await AsyncInferenceClient().list_deployed_models("text-generation-inference")
    assert isinstance(models_by_task, dict)
    for task, models in models_by_task.items():
        assert isinstance(task, str)
        assert isinstance(models, list)
        for model in models:
            assert isinstance(model, str)

    assert "text-generation" in models_by_task
    assert "bigscience/bloom" in models_by_task["text-generation"]


@pytest.mark.asyncio
async def test_async_generate_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _mock_aiohttp_client_timeout(*args, **kwargs):
        raise asyncio.TimeoutError

    monkeypatch.setattr("aiohttp.ClientSession.post", _mock_aiohttp_client_timeout)
    with pytest.raises(InferenceTimeoutError):
        await AsyncInferenceClient(timeout=1).text_generation("test")


class CustomException(Exception):
    """Mock any exception that could happen while making a POST request."""


@patch("aiohttp.ClientSession.post", side_effect=CustomException())
@patch("aiohttp.ClientSession.close")
@pytest.mark.asyncio
async def test_close_connection_on_post_error(mock_close: Mock, mock_post: Mock) -> None:
    async_client = AsyncInferenceClient()

    with pytest.raises(CustomException):
        await async_client.post(model="http://127.0.0.1/api", json={})

    mock_close.assert_called_once()


@pytest.mark.vcr
@pytest.mark.asyncio
@with_production_testing
async def test_openai_compatibility_base_url_and_api_key():
    client = AsyncInferenceClient(
        base_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
        api_key="my-api-key",
    )
    output = await client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count to 10"},
        ],
        stream=False,
        max_tokens=1024,
    )
    assert output.choices[0].message.content == "1, 2, 3, 4, 5, 6, 7, 8, 9, 10!"


@pytest.mark.vcr
@pytest.mark.asyncio
@with_production_testing
async def test_openai_compatibility_without_base_url():
    client = AsyncInferenceClient()
    output = await client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count to 10"},
        ],
        stream=False,
        max_tokens=1024,
    )
    assert output.choices[0].message.content == "1, 2, 3, 4, 5, 6, 7, 8, 9, 10!"


@pytest.mark.vcr
@pytest.mark.asyncio
@with_production_testing
async def test_openai_compatibility_with_stream_true():
    client = AsyncInferenceClient()
    output = await client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count to 10"},
        ],
        stream=True,
        max_tokens=1024,
    )

    chunked_text = [chunk.choices[0].delta.content async for chunk in output]
    assert len(chunked_text) == 34
    assert "".join(chunked_text) == "Here it goes:\n\n1, 2, 3, 4, 5, 6, 7, 8, 9, 10!"


@pytest.mark.vcr
@pytest.mark.asyncio
@with_production_testing
async def test_http_session_correctly_closed() -> None:
    """
    Regression test for #2493.
    Async client should close the HTTP session after the request is done.
    This is always done except for streamed responses if the stream is not fully consumed.
    Fixed by keeping a list of sessions and closing them all when deleting the client.

    See https://github.com/huggingface/huggingface_hub/issues/2493.
    """

    client = AsyncInferenceClient("meta-llama/Meta-Llama-3.1-8B-Instruct")
    kwargs = {"prompt": "Hi", "stream": True, "max_new_tokens": 1}

    # Test create session + close it + check correctly unregistered
    await client.text_generation(**kwargs)
    assert len(client._sessions) == 1
    await list(client._sessions)[0].close()
    assert len(client._sessions) == 0

    # Test create multiple sessions + close AsyncInferenceClient + check correctly unregistered
    await client.text_generation(**kwargs)
    await client.text_generation(**kwargs)
    await client.text_generation(**kwargs)

    assert len(client._sessions) == 3
    await client.close()
    assert len(client._sessions) == 0


@pytest.mark.asyncio
async def test_use_async_with_inference_client():
    with patch("huggingface_hub.AsyncInferenceClient.close") as mock_close:
        async with AsyncInferenceClient():
            pass
    mock_close.assert_called_once()


@pytest.mark.asyncio
@patch("aiohttp.ClientSession._request")
async def test_client_responses_correctly_closed(request_mock: Mock) -> None:
    """
    Regression test for #2521.
    Async client must close the ClientResponse objects when exiting the async context manager.
    Fixed by closing the response objects when the session is closed.

    See https://github.com/huggingface/huggingface_hub/issues/2521.
    """
    async with AsyncInferenceClient() as client:
        session = client._get_client_session()
        response1 = await session.get("http://this-is-a-fake-url.com")
        response2 = await session.post("http://this-is-a-fake-url.com", json={})

    # Response objects are closed when the AsyncInferenceClient is closed
    response1.close.assert_called_once()
    response2.close.assert_called_once()


@pytest.mark.asyncio
async def test_warns_if_client_deleted_with_opened_sessions():
    client = AsyncInferenceClient()
    session = client._get_client_session()
    with pytest.warns(UserWarning):
        client.__del__()
    await session.close()
