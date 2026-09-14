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
import re
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import numpy as np
import pytest

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
    constants,
)
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.inference._common import ValidationError as TextGenerationValidationError
from huggingface_hub.inference._common import _get_unsupported_text_generation_kwargs

from .test_inference_client import CHAT_COMPLETE_NON_TGI_MODEL, CHAT_COMPLETION_MESSAGES, CHAT_COMPLETION_MODEL


pytestmark = pytest.mark.inference


@pytest.fixture(autouse=True)
def patch_non_tgi_server(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(huggingface_hub.inference._common, "_UNSUPPORTED_TEXT_GENERATION_KWARGS", {})


@pytest.fixture
def tgi_client() -> AsyncInferenceClient:
    return AsyncInferenceClient(model="openai-community/gpt2")


@pytest.mark.asyncio
@pytest.mark.production
@pytest.mark.skip("Temporary skipping this test")
async def test_async_generate_no_details(tgi_client: AsyncInferenceClient) -> None:
    response = await tgi_client.text_generation("test", details=False, max_new_tokens=1)
    assert isinstance(response, str)
    assert response == "."


@pytest.mark.asyncio
@pytest.mark.production
@pytest.mark.skip("Temporary skipping this test")
async def test_async_generate_with_details(tgi_client: AsyncInferenceClient) -> None:
    response = await tgi_client.text_generation("test", details=True, max_new_tokens=1, decoder_input_details=True)

    assert response.generated_text == "."
    assert response.details.finish_reason == "length"
    assert response.details.generated_tokens == 1
    assert response.details.seed is None
    assert len(response.details.prefill) == 1
    assert response.details.prefill[0] == TextGenerationOutputPrefillToken(id=9288, logprob=None, text="test")
    assert len(response.details.tokens) == 1
    assert response.details.tokens[0].id == 13
    assert response.details.tokens[0].text == "."
    assert not response.details.tokens[0].special


@pytest.mark.asyncio
@pytest.mark.production
@pytest.mark.skip("Temporary skipping this test")
async def test_async_generate_best_of(tgi_client: AsyncInferenceClient) -> None:
    response = await tgi_client.text_generation(
        "test", max_new_tokens=1, best_of=2, do_sample=True, decoder_input_details=True, details=True
    )

    assert response.details.seed is not None
    assert response.details.best_of_sequences is not None
    assert len(response.details.best_of_sequences) == 1
    assert response.details.best_of_sequences[0].seed is not None


@pytest.mark.asyncio
@pytest.mark.production
@pytest.mark.skip("Temporary skipping this test")
async def test_async_generate_validation_error(tgi_client: AsyncInferenceClient) -> None:
    with pytest.raises(TextGenerationValidationError):
        await tgi_client.text_generation("test", max_new_tokens=10_000)


@pytest.mark.asyncio
@pytest.mark.skip("skipping this test, as InferenceAPI seems to not throw an error when sending unsupported params")
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


@pytest.mark.skip("Temporary skipping this test")
@pytest.mark.asyncio
@pytest.mark.production
async def test_async_generate_stream_no_details(tgi_client: AsyncInferenceClient) -> None:
    responses = [
        response async for response in await tgi_client.text_generation("test", max_new_tokens=1, stream=True)
    ]

    assert len(responses) == 1
    response = responses[0]

    assert isinstance(response, str)
    assert response == "."


@pytest.mark.skip("Temporary skipping this test")
@pytest.mark.asyncio
@pytest.mark.production
async def test_async_generate_stream_with_details(tgi_client: AsyncInferenceClient) -> None:
    responses = [
        response
        async for response in await tgi_client.text_generation("test", max_new_tokens=1, stream=True, details=True)
    ]

    assert len(responses) == 1
    response = responses[0]

    assert response.generated_text == "."
    assert response.details.finish_reason == "length"
    assert response.details.generated_tokens == 1
    assert response.details.seed is None


@pytest.mark.skip("Temporary skipping this test")
@pytest.mark.asyncio
@pytest.mark.production
async def test_async_chat_completion_no_stream() -> None:
    async_client = AsyncInferenceClient(model=CHAT_COMPLETION_MODEL)
    output = await async_client.chat_completion(CHAT_COMPLETION_MESSAGES, max_tokens=10)
    assert isinstance(output.created, int)
    assert output == ChatCompletionOutput(
        id="",
        model="HuggingFaceH4/zephyr-7b-beta",
        system_fingerprint="3.0.1-sha-bb9095a",
        usage=ChatCompletionOutputUsage(completion_tokens=10, prompt_tokens=46, total_tokens=56),
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


@pytest.mark.skip("Temporary skipping this test")
@pytest.mark.asyncio
@pytest.mark.production
async def test_async_chat_completion_not_tgi_no_stream() -> None:
    async_client = AsyncInferenceClient(model=CHAT_COMPLETE_NON_TGI_MODEL)
    output = await async_client.chat_completion(CHAT_COMPLETION_MESSAGES, max_tokens=10)
    assert isinstance(output.created, int)
    assert output == ChatCompletionOutput(
        choices=[
            ChatCompletionOutputComplete(
                finish_reason="length",
                index=0,
                message=ChatCompletionOutputMessage(
                    role="assistant", content="Deep learning isn't even an algorithm though.", tool_calls=None
                ),
                logprobs=None,
            )
        ],
        created=1737562613,
        id="",
        model="microsoft/DialoGPT-small",
        system_fingerprint="3.0.1-sha-bb9095a",
        usage=ChatCompletionOutputUsage(completion_tokens=10, prompt_tokens=13, total_tokens=23),
    )


@pytest.mark.skip("Temporary skipping this test")
@pytest.mark.asyncio
@pytest.mark.production
async def test_async_chat_completion_with_stream() -> None:
    async_client = AsyncInferenceClient(model=CHAT_COMPLETION_MODEL)
    output = await async_client.chat_completion(CHAT_COMPLETION_MESSAGES, max_tokens=10, stream=True)

    all_items = []
    generated_text = ""
    async for item in output:
        all_items.append(item)
        assert isinstance(item, ChatCompletionStreamOutput)
        assert len(item.choices) == 1
        if item.choices[0].delta.content is not None:
            generated_text += item.choices[0].delta.content

    assert len(all_items) > 0
    last_item = all_items[-1]
    assert last_item.choices[0].finish_reason == "length"


@pytest.mark.skip("Temporary skipping this test")
@pytest.mark.asyncio
@pytest.mark.production
async def test_async_sentence_similarity() -> None:
    async_client = AsyncInferenceClient(model="sentence-transformers/all-MiniLM-L6-v2")
    scores = await async_client.sentence_similarity(
        "Machine learning is so easy.",
        other_sentences=[
            "Deep learning is so straightforward.",
            "This is so difficult, like rocket science.",
            "I can't believe how much I struggled with this.",
        ],
    )
    assert scores == [0.7785724997520447, 0.45876249670982362, 0.29062220454216003]


@pytest.mark.asyncio
async def test_async_feature_extraction_accepts_list_inputs() -> None:
    helper = MagicMock()
    helper.prepare_request.return_value = MagicMock()
    helper.get_response.return_value = [[1.0, 2.0], [3.0, 4.0]]
    async_client = AsyncInferenceClient(model="sentence-transformers/all-MiniLM-L6-v2")

    with (
        patch("huggingface_hub.inference._generated._async_client.get_provider_helper", return_value=helper),
        patch.object(AsyncInferenceClient, "_inner_post", AsyncMock(return_value=b"ignored")),
    ):
        embedding = await async_client.feature_extraction(["Hi, who are you?", "How are you?"])

    helper.prepare_request.assert_called_once()
    assert helper.prepare_request.call_args.kwargs["inputs"] == ["Hi, who are you?", "How are you?"]
    np.testing.assert_array_equal(embedding, np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"))


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
        # Since some methods are decorated with @_deprecate_arguments, we need to unwrap the async method to get the actual coroutine function
        # TODO: Remove this once the @_deprecate_arguments decorator is removed from the AsyncInferenceClient methods.
        assert inspect.iscoroutinefunction(inspect.unwrap(async_method))

        # Check that expected inputs and outputs are the same
        sync_sig = inspect.signature(sync_method)
        async_sig = inspect.signature(async_method)
        assert sync_sig.parameters == async_sig.parameters
        assert sync_sig.return_annotation == async_sig.return_annotation


@pytest.mark.asyncio
async def test_async_generate_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _mock_client_post(*args, **kwargs):
        raise asyncio.TimeoutError

    def mock_check_supported_task(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "huggingface_hub.inference._providers.hf_inference._check_supported_task", mock_check_supported_task
    )
    client = AsyncInferenceClient(timeout=1)
    client._async_client = Mock(post=_mock_client_post)
    with pytest.raises(InferenceTimeoutError):
        await client.text_generation("test")


class CustomException(Exception):
    """Mock any exception that could happen while making a POST request."""


@pytest.mark.skip("Temporary skipping this test")
@pytest.mark.asyncio
@pytest.mark.production
async def test_openai_compatibility_base_url_and_api_key():
    client = AsyncInferenceClient(
        base_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
        api_key="my-api-key",
    )
    output = await client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count to 10"},
        ],
        stream=False,
        max_tokens=1024,
    )
    assert "1, 2, 3, 4, 5, 6, 7, 8, 9, 10" in output.choices[0].message.content


@pytest.mark.skip("Temporary skipping this test")
@pytest.mark.asyncio
@pytest.mark.production
async def test_openai_compatibility_without_base_url():
    client = AsyncInferenceClient()
    output = await client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count to 10"},
        ],
        stream=False,
        max_tokens=1024,
    )
    assert "1, 2, 3, 4, 5, 6, 7, 8, 9, 10" in output.choices[0].message.content


@pytest.mark.skip("Temporary skipping this test")
@pytest.mark.asyncio
@pytest.mark.production
async def test_openai_compatibility_with_stream_true():
    client = AsyncInferenceClient()
    output = await client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count to 10"},
        ],
        stream=True,
        max_tokens=1024,
    )

    chunked_text = [
        chunk.choices[0].delta.content async for chunk in output if chunk.choices[0].delta.content is not None
    ]
    assert len(chunked_text) == 35
    output_text = "".join(chunked_text)
    assert "1, 2, 3, 4, 5, 6, 7, 8, 9, 10" in output_text


@pytest.mark.skip("Temporary skipping this test")
@pytest.mark.asyncio
@pytest.mark.production
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
async def test_async_get_endpoint_info_hub_model_id():
    client = AsyncInferenceClient(provider="hf-inference")
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"model_id": "meta-llama/Meta-Llama-3-70B-Instruct"}

    mock_async_client = AsyncMock()
    mock_async_client.get.return_value = mock_response

    with patch.object(client, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        info = await client.get_endpoint_info(model="meta-llama/Meta-Llama-3-70B-Instruct")

    assert info == {"model_id": "meta-llama/Meta-Llama-3-70B-Instruct"}
    requested_url = mock_async_client.get.call_args[0][0]
    assert (
        requested_url == "https://router.huggingface.co/hf-inference/models/meta-llama/Meta-Llama-3-70B-Instruct/info"
    )


@pytest.mark.asyncio
async def test_async_get_endpoint_info_default_client_instance():
    # Test standard client instantiation where provider defaults to None and model is set on client (#4887)
    client = AsyncInferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
    assert client.provider is None
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"model_id": "meta-llama/Meta-Llama-3-70B-Instruct"}

    mock_async_client = AsyncMock()
    mock_async_client.get.return_value = mock_response

    with patch.object(client, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        info = await client.get_endpoint_info()

    assert info == {"model_id": "meta-llama/Meta-Llama-3-70B-Instruct"}
    requested_url = mock_async_client.get.call_args[0][0]
    assert (
        requested_url == "https://router.huggingface.co/hf-inference/models/meta-llama/Meta-Llama-3-70B-Instruct/info"
    )


@pytest.mark.asyncio
async def test_async_get_endpoint_info_custom_endpoint(monkeypatch):
    monkeypatch.setattr(constants, "INFERENCE_ENDPOINT", "https://custom-gateway.internal/hf-inference")
    client = AsyncInferenceClient()
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"model_id": "meta-llama/Meta-Llama-3-70B-Instruct"}

    mock_async_client = AsyncMock()
    mock_async_client.get.return_value = mock_response

    with patch.object(client, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        info = await client.get_endpoint_info(model="meta-llama/Meta-Llama-3-70B-Instruct")

    assert info == {"model_id": "meta-llama/Meta-Llama-3-70B-Instruct"}
    requested_url = mock_async_client.get.call_args[0][0]
    assert (
        requested_url
        == "https://custom-gateway.internal/hf-inference/models/meta-llama/Meta-Llama-3-70B-Instruct/info"
    )


@pytest.mark.asyncio
async def test_async_get_endpoint_info_direct_url():
    client = AsyncInferenceClient("https://custom-endpoint.endpoints.huggingface.cloud")
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"status": "ok"}

    mock_async_client = AsyncMock()
    mock_async_client.get.return_value = mock_response

    with patch.object(client, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        info = await client.get_endpoint_info()

    assert info == {"status": "ok"}
    requested_url = mock_async_client.get.call_args[0][0]
    assert requested_url == "https://custom-endpoint.endpoints.huggingface.cloud/info"


@pytest.mark.asyncio
async def test_async_get_endpoint_info_invalid_provider_or_missing_model():
    client_provider = AsyncInferenceClient(provider="fal-ai")
    with pytest.raises(ValueError, match="Getting endpoint info is not supported on 'fal-ai'."):
        await client_provider.get_endpoint_info(model="some-model")

    client_no_model = AsyncInferenceClient(provider="hf-inference")
    with pytest.raises(ValueError, match="Model id not provided."):
        await client_no_model.get_endpoint_info()


@pytest.mark.asyncio
async def test_async_health_check():
    client = AsyncInferenceClient("https://custom-endpoint.endpoints.huggingface.cloud")
    mock_response = MagicMock(status_code=200)

    mock_async_client = AsyncMock()
    mock_async_client.get.return_value = mock_response

    with patch.object(client, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        assert await client.health_check() is True

    requested_url = mock_async_client.get.call_args[0][0]
    assert requested_url == "https://custom-endpoint.endpoints.huggingface.cloud/health"

    mock_response.status_code = 503
    with patch.object(client, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        assert await client.health_check() is False

    client_hub_model = AsyncInferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct")
    with pytest.raises(ValueError, match="Model must be an Inference Endpoint URL."):
        await client_hub_model.health_check()

    client_invalid_provider = AsyncInferenceClient(
        "https://custom-endpoint.endpoints.huggingface.cloud", provider="fal-ai"
    )
    with pytest.raises(ValueError, match="Health check is not supported on 'fal-ai'."):
        await client_invalid_provider.health_check()


@pytest.mark.asyncio
async def test_async_get_endpoint_info_missing_and_none_arguments():
    # Missing argument with no model at client instantiation
    client_no_model = AsyncInferenceClient()
    with pytest.raises(ValueError, match="Model id not provided."):
        await client_no_model.get_endpoint_info()

    # Explicit None model with no model at client instantiation
    with pytest.raises(ValueError, match="Model id not provided."):
        await client_no_model.get_endpoint_info(model=None)

    # Empty string model with no model at client instantiation
    with pytest.raises(ValueError, match="Model id not provided."):
        await client_no_model.get_endpoint_info(model="")

    # Fallback to instance-level model when model=None is passed
    client_with_model = AsyncInferenceClient("meta-llama/Meta-Llama-3-70B-Instruct")
    mock_response = httpx.Response(
        200,
        request=httpx.Request(
            "GET",
            "https://router.huggingface.co/hf-inference/models/meta-llama/Meta-Llama-3-70B-Instruct/info",
        ),
        json={"model_id": "meta-llama/Meta-Llama-3-70B-Instruct"},
    )
    mock_async_client = AsyncMock()
    mock_async_client.get.return_value = mock_response

    with patch.object(client_with_model, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        info = await client_with_model.get_endpoint_info(model=None)
    assert info == {"model_id": "meta-llama/Meta-Llama-3-70B-Instruct"}
    assert (
        mock_async_client.get.call_args[0][0]
        == "https://router.huggingface.co/hf-inference/models/meta-llama/Meta-Llama-3-70B-Instruct/info"
    )

    # Fallback to instance-level model when model="" is passed
    with patch.object(client_with_model, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        info_empty = await client_with_model.get_endpoint_info(model="")
    assert info_empty == {"model_id": "meta-llama/Meta-Llama-3-70B-Instruct"}
    assert (
        mock_async_client.get.call_args[0][0]
        == "https://router.huggingface.co/hf-inference/models/meta-llama/Meta-Llama-3-70B-Instruct/info"
    )

    # Method argument overrides instance-level model
    override_response = httpx.Response(
        200,
        request=httpx.Request(
            "GET",
            "https://router.huggingface.co/hf-inference/models/override-model/info",
        ),
        json={"model_id": "override-model"},
    )
    mock_async_client.get.return_value = override_response
    with patch.object(client_with_model, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        info_override = await client_with_model.get_endpoint_info(model="override-model")
    assert info_override == {"model_id": "override-model"}
    assert (
        mock_async_client.get.call_args[0][0]
        == "https://router.huggingface.co/hf-inference/models/override-model/info"
    )


@pytest.mark.asyncio
async def test_async_get_endpoint_info_invalid_providers_and_types():
    # Unsupported provider strings
    for provider in ("fal-ai", "together", "replicate", "sambanova", "unsupported-provider", ""):
        client = AsyncInferenceClient(provider=provider)
        with pytest.raises(ValueError, match=re.escape(f"Getting endpoint info is not supported on '{provider}'.")):
            await client.get_endpoint_info(model="some-model")

    # Non-string provider types (int, bool, float, list, dict)
    for invalid_provider in (123, False, 3.14, ["fal-ai"], {"provider": "together"}):
        client = AsyncInferenceClient(provider=invalid_provider)
        with pytest.raises(
            ValueError, match=re.escape(f"Getting endpoint info is not supported on '{invalid_provider}'.")
        ):
            await client.get_endpoint_info(model="some-model")

    # Supported providers: None and "hf-inference"
    assert AsyncInferenceClient(provider=None).provider is None
    assert AsyncInferenceClient(provider="hf-inference").provider == "hf-inference"


@pytest.mark.asyncio
async def test_async_get_endpoint_info_invalid_model_types():
    client = AsyncInferenceClient()
    # Falsy non-string types evaluate to None when no model is set on client
    with pytest.raises(ValueError, match="Model id not provided."):
        await client.get_endpoint_info(model=False)

    # Truthy non-string types fail when string methods (like startswith) are invoked
    for invalid_model in (123, 45.67, ["meta-llama"], {"model": "gpt2"}):
        with pytest.raises((AttributeError, TypeError)):
            await client.get_endpoint_info(model=invalid_model)


@pytest.mark.asyncio
async def test_async_get_endpoint_info_url_formatting_and_trailing_slashes():
    mock_response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://custom-endpoint.endpoints.huggingface.cloud/info"),
        json={"status": "ok"},
    )
    mock_async_client = AsyncMock()
    mock_async_client.get.return_value = mock_response

    # Single trailing slash
    client_single_slash = AsyncInferenceClient("https://custom-endpoint.endpoints.huggingface.cloud/")
    with patch.object(client_single_slash, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        info = await client_single_slash.get_endpoint_info()
    assert info == {"status": "ok"}
    assert mock_async_client.get.call_args[0][0] == "https://custom-endpoint.endpoints.huggingface.cloud/info"

    # Multiple trailing slashes
    client_multi_slash = AsyncInferenceClient("https://custom-endpoint.endpoints.huggingface.cloud///")
    with patch.object(client_multi_slash, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        info = await client_multi_slash.get_endpoint_info()
    assert info == {"status": "ok"}
    assert mock_async_client.get.call_args[0][0] == "https://custom-endpoint.endpoints.huggingface.cloud/info"

    # Local http URL with trailing slash
    client_http = AsyncInferenceClient("http://127.0.0.1:8000/")
    with patch.object(client_http, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        info = await client_http.get_endpoint_info()
    assert info == {"status": "ok"}
    assert mock_async_client.get.call_args[0][0] == "http://127.0.0.1:8000/info"


@pytest.mark.asyncio
async def test_async_get_endpoint_info_error_boundaries():
    client = AsyncInferenceClient()
    mock_async_client = AsyncMock()

    error_statuses = (
        (400, "Bad Request"),
        (401, "Invalid credentials."),
        (403, "Access to gated model forbidden."),
        (404, "Model nonexistent-model does not exist or is not supported."),
        (429, "Too Many Requests."),
        (500, "Internal server error."),
        (502, "Bad Gateway."),
        (503, "Service unavailable."),
        (504, "Gateway Timeout."),
    )

    with patch.object(client, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        for status_code, error_msg in error_statuses:
            mock_async_client.get.return_value = httpx.Response(
                status_code,
                request=httpx.Request(
                    "GET", f"https://router.huggingface.co/hf-inference/models/test-model-{status_code}/info"
                ),
                json={"error": error_msg},
            )
            with pytest.raises(HfHubHTTPError) as exc_info:
                await client.get_endpoint_info(model=f"test-model-{status_code}")
            assert exc_info.value.response.status_code == status_code


@pytest.mark.asyncio
async def test_async_health_check_missing_and_none_arguments():
    # 1. Missing argument with no model at client instantiation
    client_no_model = AsyncInferenceClient()
    with pytest.raises(ValueError, match="Model id not provided."):
        await client_no_model.health_check()

    # 2. Explicit None with no model at client instantiation
    with pytest.raises(ValueError, match="Model id not provided."):
        await client_no_model.health_check(model=None)

    # 3. Empty string with no model at client instantiation
    with pytest.raises(ValueError, match="Model id not provided."):
        await client_no_model.health_check(model="")

    # 4. Fallback to instance-level model when model=None is passed
    client_with_model = AsyncInferenceClient("https://custom-endpoint.endpoints.huggingface.cloud")
    mock_async_client = AsyncMock()
    mock_async_client.get.return_value = httpx.Response(
        200,
        request=httpx.Request("GET", "https://custom-endpoint.endpoints.huggingface.cloud/health"),
    )

    with patch.object(client_with_model, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        assert await client_with_model.health_check(model=None) is True
    assert mock_async_client.get.call_args[0][0] == "https://custom-endpoint.endpoints.huggingface.cloud/health"

    # 5. Method argument overrides instance-level model
    override_resp = httpx.Response(
        200,
        request=httpx.Request("GET", "https://override-endpoint.endpoints.huggingface.cloud/health"),
    )
    mock_async_client.get.return_value = override_resp
    with patch.object(client_with_model, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        assert (
            await client_with_model.health_check(model="https://override-endpoint.endpoints.huggingface.cloud") is True
        )
    assert mock_async_client.get.call_args[0][0] == "https://override-endpoint.endpoints.huggingface.cloud/health"

    # 6. Empty string with instance-level model falls back to instance-level model
    mock_async_client.get.return_value = httpx.Response(
        200,
        request=httpx.Request("GET", "https://custom-endpoint.endpoints.huggingface.cloud/health"),
    )
    with patch.object(client_with_model, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        assert await client_with_model.health_check(model="") is True
    assert mock_async_client.get.call_args[0][0] == "https://custom-endpoint.endpoints.huggingface.cloud/health"


@pytest.mark.asyncio
async def test_async_health_check_invalid_providers_and_types():
    endpoint_url = "https://custom-endpoint.endpoints.huggingface.cloud"
    # Unsupported provider strings
    for provider in ("fal-ai", "together", "replicate", "sambanova", "unknown-provider", ""):
        client = AsyncInferenceClient(endpoint_url, provider=provider)
        with pytest.raises(ValueError, match=re.escape(f"Health check is not supported on '{provider}'.")):
            await client.health_check()

    # Non-string provider types (int, bool, float, list, dict)
    for invalid_provider in (999, False, 1.23, ["fal-ai"], {"provider": "together"}):
        client = AsyncInferenceClient(endpoint_url, provider=invalid_provider)
        with pytest.raises(ValueError, match=re.escape(f"Health check is not supported on '{invalid_provider}'.")):
            await client.health_check()

    # Supported providers: None and "hf-inference"
    mock_async_client = AsyncMock()
    mock_async_client.get.return_value = httpx.Response(
        200,
        request=httpx.Request("GET", "https://custom-endpoint.endpoints.huggingface.cloud/health"),
    )

    client_none = AsyncInferenceClient(endpoint_url, provider=None)
    assert client_none.provider is None
    with patch.object(client_none, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        assert await client_none.health_check() is True

    client_hf = AsyncInferenceClient(endpoint_url, provider="hf-inference")
    assert client_hf.provider == "hf-inference"
    with patch.object(client_hf, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        assert await client_hf.health_check() is True


@pytest.mark.asyncio
async def test_async_health_check_invalid_model_urls():
    client = AsyncInferenceClient()
    # Hub model repo IDs (must be endpoint URLs)
    for hub_id in ("meta-llama/Meta-Llama-3-70B-Instruct", "gpt2", "bert-base-uncased"):
        with pytest.raises(ValueError, match="Model must be an Inference Endpoint URL."):
            await client.health_check(model=hub_id)

    # Non-HTTP(S) schemes, local paths, and malformed strings
    for invalid_url in (
        "ftp://endpoint.example.com",
        "ws://endpoint.example.com",
        "file:///tmp/endpoint",
        "/local/path/to/endpoint",
        "endpoints.huggingface.cloud",
        "   ",
    ):
        with pytest.raises(ValueError, match="Model must be an Inference Endpoint URL."):
            await client.health_check(model=invalid_url)


@pytest.mark.asyncio
async def test_async_health_check_invalid_model_types():
    client = AsyncInferenceClient()
    # Falsy non-string types evaluate to None when no model is set on client
    with pytest.raises(ValueError, match="Model id not provided."):
        await client.health_check(model=False)

    # Truthy non-string types fail when string methods (like startswith) are invoked
    for invalid_model in (123, 45.67, ["https://custom-endpoint.endpoints.huggingface.cloud"], {"url": "http://foo"}):
        with pytest.raises((AttributeError, TypeError)):
            await client.health_check(model=invalid_model)


@pytest.mark.asyncio
async def test_async_health_check_url_formatting_and_trailing_slashes():
    mock_response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://custom-endpoint.endpoints.huggingface.cloud/health"),
    )
    mock_async_client = AsyncMock()
    mock_async_client.get.return_value = mock_response

    # Single trailing slash
    client_single = AsyncInferenceClient("https://custom-endpoint.endpoints.huggingface.cloud/")
    with patch.object(client_single, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        assert await client_single.health_check() is True
    assert mock_async_client.get.call_args[0][0] == "https://custom-endpoint.endpoints.huggingface.cloud/health"

    # Multiple trailing slashes
    client_multi = AsyncInferenceClient("https://custom-endpoint.endpoints.huggingface.cloud///")
    with patch.object(client_multi, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        assert await client_multi.health_check() is True
    assert mock_async_client.get.call_args[0][0] == "https://custom-endpoint.endpoints.huggingface.cloud/health"

    # Local http URL with trailing slash
    client_http = AsyncInferenceClient("http://127.0.0.1:8000/")
    with patch.object(client_http, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        assert await client_http.health_check() is True
    assert mock_async_client.get.call_args[0][0] == "http://127.0.0.1:8000/health"


@pytest.mark.asyncio
async def test_async_health_check_error_boundaries():
    client = AsyncInferenceClient("https://custom-endpoint.endpoints.huggingface.cloud")
    mock_async_client = AsyncMock()

    with patch.object(client, "_get_async_client", AsyncMock(return_value=mock_async_client)):
        # Non-200 status codes that should return False (not raise)
        for status_code in (400, 401, 403, 404, 429, 500, 502, 503, 504, 302):
            mock_async_client.get.return_value = httpx.Response(
                status_code,
                request=httpx.Request("GET", "https://custom-endpoint.endpoints.huggingface.cloud/health"),
            )
            assert await client.health_check() is False, f"Expected status {status_code} to return False"

        # Status code 200 must return True
        mock_async_client.get.return_value = httpx.Response(
            200,
            request=httpx.Request("GET", "https://custom-endpoint.endpoints.huggingface.cloud/health"),
        )
        assert await client.health_check() is True
