# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from huggingface_hub import InferenceClient, constants, hf_hub_download
from huggingface_hub.errors import ValidationError
from huggingface_hub.inference._client import _open_as_binary
from huggingface_hub.inference._common import _stream_chat_completion_response, _stream_text_generation_response
from huggingface_hub.inference._providers import get_provider_helper
from huggingface_hub.inference._providers.hf_inference import _build_chat_completion_url

from .testing_utils import expect_deprecation, with_production_testing


# Define fixtures for the files
@pytest.fixture(scope="module")
@with_production_testing
def audio_file():
    return hf_hub_download(repo_id="Narsil/image_dummy", repo_type="dataset", filename="sample1.flac")


@pytest.fixture(scope="module")
@with_production_testing
def image_file():
    return hf_hub_download(repo_id="Narsil/image_dummy", repo_type="dataset", filename="lena.png")


@pytest.fixture(scope="module")
@with_production_testing
def document_file():
    return hf_hub_download(repo_id="impira/docquery", repo_type="space", filename="contract.jpeg")


class TestOpenAsBinary:
    @pytest.fixture(autouse=True)
    def setup(self, audio_file, image_file, document_file):
        self.audio_file = audio_file
        self.image_file = image_file
        self.document_file = document_file

    def test_open_as_binary_with_none(self) -> None:
        with _open_as_binary(None) as content:
            assert content is None

    def test_open_as_binary_from_str_path(self) -> None:
        with _open_as_binary(self.image_file) as content:
            assert isinstance(content, io.BufferedReader)

    def test_open_as_binary_from_pathlib_path(self) -> None:
        with _open_as_binary(Path(self.image_file)) as content:
            assert isinstance(content, io.BufferedReader)

    def test_open_as_binary_from_url(self) -> None:
        with _open_as_binary("https://huggingface.co/datasets/Narsil/image_dummy/resolve/main/tree.png") as content:
            assert isinstance(content, bytes)

    def test_open_as_binary_opened_file(self) -> None:
        with Path(self.image_file).open("rb") as f:
            with _open_as_binary(f) as content:
                assert content == f
                assert isinstance(content, io.BufferedReader)

    def test_open_as_binary_from_bytes(self) -> None:
        content_bytes = Path(self.image_file).read_bytes()
        with _open_as_binary(content_bytes) as content:
            assert content == content_bytes


class TestHeadersAndCookies:
    @pytest.fixture(autouse=True)
    def setup(self, audio_file, image_file, document_file):
        self.audio_file = audio_file
        self.image_file = image_file
        self.document_file = document_file

    def test_headers_and_cookies(self) -> None:
        client = InferenceClient(headers={"X-My-Header": "foo"}, cookies={"my-cookie": "bar"})
        assert client.headers["X-My-Header"] == "foo"
        assert client.cookies["my-cookie"] == "bar"

    @patch("huggingface_hub.inference._client._bytes_to_image")
    @patch("huggingface_hub.inference._client.get_session")
    @patch("huggingface_hub.inference._providers.hf_inference._check_supported_task")
    def test_accept_header_image(
        self,
        check_supported_task_mock: MagicMock,
        get_session_mock: MagicMock,
        bytes_to_image_mock: MagicMock,
    ) -> None:
        """Test that Accept: image/png header is set for image tasks."""
        client = InferenceClient()

        response = client.text_to_image("An astronaut riding a horse")
        assert response == bytes_to_image_mock.return_value

        headers = get_session_mock().post.call_args_list[0].kwargs["headers"]
        assert headers["Accept"] == "image/png"


class TestListDeployedModels:
    @pytest.fixture(autouse=True)
    def setup(self, audio_file, image_file, document_file):
        self.audio_file = audio_file
        self.image_file = image_file
        self.document_file = document_file

    @expect_deprecation("list_deployed_models")
    @patch("huggingface_hub.inference._client.get_session")
    def test_list_deployed_models_main_frameworks_mock(self, get_session_mock: MagicMock) -> None:
        InferenceClient().list_deployed_models()
        assert len(get_session_mock.return_value.get.call_args_list) == len(constants.MAIN_INFERENCE_API_FRAMEWORKS)

    @expect_deprecation("list_deployed_models")
    @patch("huggingface_hub.inference._client.get_session")
    def test_list_deployed_models_all_frameworks_mock(self, get_session_mock: MagicMock) -> None:
        InferenceClient().list_deployed_models("all")
        assert len(get_session_mock.return_value.get.call_args_list) == len(constants.ALL_INFERENCE_API_FRAMEWORKS)


@pytest.mark.parametrize(
    "stop_signal",
    [
        b"data: [DONE]",
        b"data: [DONE]\n",
        b"data: [DONE] ",
    ],
)
def test_stream_text_generation_response(stop_signal: bytes):
    data = [
        b'data: {"index":1,"token":{"id":4560,"text":" trying","logprob":-2.078125,"special":false},"generated_text":null,"details":null}',
        b"",  # Empty line is skipped
        b"\n",  # Newline is skipped
        b'data: {"index":2,"token":{"id":311,"text":" to","logprob":-0.026245117,"special":false},"generated_text":" trying to","details":null}',
        stop_signal,  # Stop signal
        # Won't parse after
        b'data: {"index":2,"token":{"id":311,"text":" to","logprob":-0.026245117,"special":false},"generated_text":" trying to","details":null}',
    ]
    output = list(_stream_text_generation_response(data, details=False))
    assert len(output) == 2
    assert output == [" trying", " to"]


@pytest.mark.parametrize(
    "stop_signal",
    [
        b"data: [DONE]",
        b"data: [DONE]\n",
        b"data: [DONE] ",
    ],
)
def test_stream_chat_completion_response(stop_signal: bytes):
    data = [
        b'data: {"object":"chat.completion.chunk","id":"","created":1721737661,"model":"","system_fingerprint":"2.1.2-dev0-sha-5fca30e","choices":[{"index":0,"delta":{"role":"assistant","content":"Both"},"logprobs":null,"finish_reason":null}]}',
        b"",  # Empty line is skipped
        b"\n",  # Newline is skipped
        b'data: {"object":"chat.completion.chunk","id":"","created":1721737661,"model":"","system_fingerprint":"2.1.2-dev0-sha-5fca30e","choices":[{"index":0,"delta":{"role":"assistant","content":" Rust"},"logprobs":null,"finish_reason":null}]}',
        stop_signal,  # Stop signal
        # Won't parse after
        b'data: {"index":2,"token":{"id":311,"text":" to","logprob":-0.026245117,"special":false},"generated_text":" trying to","details":null}',
    ]
    output = list(_stream_chat_completion_response(data))
    assert len(output) == 2
    assert output[0].choices[0].delta.content == "Both"
    assert output[1].choices[0].delta.content == " Rust"


def test_chat_completion_error_in_stream():
    """
    Regression test for https://github.com/huggingface/huggingface_hub/issues/2514.
    When an error is encountered in the stream, it should raise a TextGenerationError (e.g. a ValidationError).
    """
    data = [
        b'data: {"object":"chat.completion.chunk","id":"","created":1721737661,"model":"","system_fingerprint":"2.1.2-dev0-sha-5fca30e","choices":[{"index":0,"delta":{"role":"assistant","content":"Both"},"logprobs":null,"finish_reason":null}]}',
        b'data: {"error":"Input validation error: `inputs` tokens + `max_new_tokens` must be <= 4096. Given: 6 `inputs` tokens and 4091 `max_new_tokens`","error_type":"validation"}',
    ]
    with pytest.raises(ValidationError):
        for token in _stream_chat_completion_response(data):
            pass


INFERENCE_API_URL = "https://api-inference.huggingface.co/models"
INFERENCE_ENDPOINT_URL = "https://rur2d6yoccusjxgn.us-east-1.aws.endpoints.huggingface.cloud"  # example
LOCAL_TGI_URL = "http://0.0.0.0:8080"


@pytest.mark.parametrize(
    ("model_url", "expected_url"),
    [
        # Inference API
        (
            f"{INFERENCE_API_URL}/username/repo_name",
            f"{INFERENCE_API_URL}/username/repo_name/v1/chat/completions",
        ),
        # Inference Endpoint
        (
            INFERENCE_ENDPOINT_URL,
            f"{INFERENCE_ENDPOINT_URL}/v1/chat/completions",
        ),
        # Inference Endpoint - full url
        (
            f"{INFERENCE_ENDPOINT_URL}/v1/chat/completions",
            f"{INFERENCE_ENDPOINT_URL}/v1/chat/completions",
        ),
        # Inference Endpoint - url with '/v1' (OpenAI compatibility)
        (
            f"{INFERENCE_ENDPOINT_URL}/v1",
            f"{INFERENCE_ENDPOINT_URL}/v1/chat/completions",
        ),
        # Inference Endpoint - url with '/v1/' (OpenAI compatibility)
        (
            f"{INFERENCE_ENDPOINT_URL}/v1/",
            f"{INFERENCE_ENDPOINT_URL}/v1/chat/completions",
        ),
        # Local TGI with trailing '/v1'
        (
            f"{LOCAL_TGI_URL}/v1",
            f"{LOCAL_TGI_URL}/v1/chat/completions",
        ),
    ],
)
def test_resolve_chat_completion_url(model_url: str, expected_url: str):
    url = _build_chat_completion_url(model_url)
    assert url == expected_url


def test_pass_url_as_base_url():
    client = InferenceClient(base_url="http://localhost:8082/v1/")
    provider = get_provider_helper("hf-inference", "text-generation")
    request = provider.prepare_request(
        inputs="The huggingface_hub library is ", parameters={}, headers={}, model=client.model, api_key=None
    )
    assert request.url == "http://localhost:8082/v1/"


def test_cannot_pass_token_false():
    """Regression test for #2853.

    It is no longer possible to pass `token=False` to the InferenceClient constructor.
    This was a legacy behavior, broken since 0.28.x release as passing token=False does not prevent the token from being
    used. Better to drop this feature altogether and raise an error if `token=False` is passed.

    See https://github.com/huggingface/huggingface_hub/pull/2853.
    """
    with pytest.raises(ValueError):
        InferenceClient(token=False)


class TestBillToOrganization:
    def test_bill_to_added_to_new_headers(self):
        client = InferenceClient(bill_to="huggingface_hub")
        assert client.headers["X-HF-Bill-To"] == "huggingface_hub"

    def test_bill_to_added_to_existing_headers(self):
        headers = {"foo": "bar"}
        client = InferenceClient(bill_to="huggingface_hub", headers=headers)
        assert client.headers["X-HF-Bill-To"] == "huggingface_hub"
        assert client.headers["foo"] == "bar"
        assert headers == {"foo": "bar"}  # do not mutate the original headers

    def test_warning_if_bill_to_already_set(self):
        headers = {"X-HF-Bill-To": "huggingface"}
        with pytest.warns(UserWarning, match="Overriding existing 'huggingface' value in headers with 'openai'."):
            client = InferenceClient(bill_to="openai", headers=headers)
        assert client.headers["X-HF-Bill-To"] == "openai"
        assert headers == {"X-HF-Bill-To": "huggingface"}  # do not mutate the original headers

    def test_warning_if_bill_to_with_direct_calls(self):
        with pytest.warns(
            UserWarning,
            match="You've provided an external provider's API key, so requests will be billed directly by the provider.",
        ):
            InferenceClient(bill_to="openai", token="replicate_key", provider="replicate")
