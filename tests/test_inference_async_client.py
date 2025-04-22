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
from huggingface_hub import AsyncInferenceClient, InferenceClient, InferenceTimeoutError


@pytest.fixture(autouse=True)
def patch_non_tgi_server(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(huggingface_hub.inference._common, "_UNSUPPORTED_TEXT_GENERATION_KWARGS", {})


@pytest.fixture
def tgi_client() -> AsyncInferenceClient:
    return AsyncInferenceClient(model="openai-community/gpt2")


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
@pytest.mark.skip("Deprecated (get_model_status)")
async def test_get_status_too_big_model() -> None:
    model_status = await AsyncInferenceClient(token=False).get_model_status("facebook/nllb-moe-54b")
    assert model_status.loaded is False
    assert model_status.state == "TooBig"
    assert model_status.compute_type == "cpu"
    assert model_status.framework == "transformers"


@pytest.mark.asyncio
@pytest.mark.skip("Deprecated (get_model_status)")
async def test_get_status_loaded_model() -> None:
    model_status = await AsyncInferenceClient(token=False).get_model_status("bigscience/bloom")
    assert model_status.loaded is True
    assert model_status.state == "Loaded"
    assert isinstance(model_status.compute_type, dict)  # e.g. {'gpu': {'gpu': 'a100', 'count': 8}}
    assert model_status.framework == "text-generation-inference"


@pytest.mark.asyncio
@pytest.mark.skip("Deprecated (get_model_status)")
async def test_get_status_unknown_model() -> None:
    with pytest.raises(ClientResponseError):
        await AsyncInferenceClient(token=False).get_model_status("unknown/model")


@pytest.mark.asyncio
@pytest.mark.skip("Deprecated (get_model_status)")
async def test_get_status_model_as_url() -> None:
    with pytest.raises(NotImplementedError):
        await AsyncInferenceClient(token=False).get_model_status("https://unkown/model")


@pytest.mark.asyncio
@pytest.mark.skip("Deprecated (list_deployed_models)")
async def test_list_deployed_models_single_frameworks() -> None:
    models_by_task = await AsyncInferenceClient().list_deployed_models("text-generation-inference")
    assert isinstance(models_by_task, dict)
    for task, models in models_by_task.items():
        assert isinstance(task, str)
        assert isinstance(models, list)
        for model in models:
            assert isinstance(model, str)

    assert "text-generation" in models_by_task
    assert "HuggingFaceH4/zephyr-7b-beta" in models_by_task["text-generation"]


@pytest.mark.asyncio
async def test_async_generate_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _mock_aiohttp_client_timeout(*args, **kwargs):
        raise asyncio.TimeoutError

    def mock_check_supported_task(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "huggingface_hub.inference._providers.hf_inference._check_supported_task", mock_check_supported_task
    )
    monkeypatch.setattr("aiohttp.ClientSession.post", _mock_aiohttp_client_timeout)
    with pytest.raises(InferenceTimeoutError):
        await AsyncInferenceClient(timeout=1).text_generation("test")


class CustomException(Exception):
    """Mock any exception that could happen while making a POST request."""


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
