from datetime import datetime, timezone
from itertools import chain, repeat
from unittest.mock import MagicMock, Mock, patch

import pytest

from huggingface_hub import (
    AsyncInferenceClient,
    HfApi,
    InferenceClient,
    InferenceEndpoint,
    InferenceEndpointError,
    InferenceEndpointTimeoutError,
)


MOCK_INITIALIZING = {
    "name": "my-endpoint-name",
    "type": "protected",
    "accountId": None,
    "provider": {"vendor": "aws", "region": "us-east-1"},
    "healthRoute": "/health",
    "compute": {
        "accelerator": "cpu",
        "instanceType": "intel-icl",
        "instanceSize": "x2",
        "scaling": {"minReplica": 0, "maxReplica": 1},
    },
    "model": {
        "repository": "gpt2",
        "revision": "11c5a3d5811f50298f278a704980280950aedb10",
        "task": "text-generation",
        "framework": "pytorch",
        "image": {"huggingface": {}},
        "secret": {"token": "my-token"},
    },
    "status": {
        "createdAt": "2023-10-26T12:41:53.263078506Z",
        "createdBy": {"id": "6273f303f6d63a28483fde12", "name": "Wauplin"},
        "updatedAt": "2023-10-26T12:41:53.263079138Z",
        "updatedBy": {"id": "6273f303f6d63a28483fde12", "name": "Wauplin"},
        "private": None,
        "state": "pending",
        "message": "Endpoint waiting to be scheduled",
        "readyReplica": 0,
        "targetReplica": 0,
    },
}

MOCK_RUNNING = {
    "name": "my-endpoint-name",
    "type": "protected",
    "accountId": None,
    "provider": {"vendor": "aws", "region": "us-east-1"},
    "healthRoute": "/health",
    "compute": {
        "accelerator": "cpu",
        "instanceType": "intel-icl",
        "instanceSize": "x2",
        "scaling": {"minReplica": 0, "maxReplica": 1},
    },
    "model": {
        "repository": "gpt2",
        "revision": "11c5a3d5811f50298f278a704980280950aedb10",
        "task": "text-generation",
        "framework": "pytorch",
        "image": {"huggingface": {}},
        "secrets": {"token": "my-token"},
    },
    "status": {
        "createdAt": "2023-10-26T12:41:53.263Z",
        "createdBy": {"id": "6273f303f6d63a28483fde12", "name": "Wauplin"},
        "updatedAt": "2023-10-26T12:41:53.263Z",
        "updatedBy": {"id": "6273f303f6d63a28483fde12", "name": "Wauplin"},
        "private": None,
        "state": "running",
        "message": "Endpoint is ready",
        "url": "https://vksrvs8pc1xnifhq.us-east-1.aws.endpoints.huggingface.cloud",
        "readyReplica": 1,
        "targetReplica": 1,
    },
}

MOCK_FAILED = {
    "name": "my-endpoint-name",
    "type": "protected",
    "accountId": None,
    "provider": {"vendor": "aws", "region": "us-east-1"},
    "healthRoute": "/health",
    "compute": {
        "accelerator": "cpu",
        "instanceType": "intel-icl",
        "instanceSize": "x2",
        "scaling": {"minReplica": 0, "maxReplica": 1},
    },
    "model": {
        "repository": "gpt2",
        "revision": "11c5a3d5811f50298f278a704980280950aedb10",
        "task": "text-generation",
        "framework": "pytorch",
        "image": {"huggingface": {}},
        "secrets": {"token": "my-token"},
    },
    "status": {
        "createdAt": "2023-10-26T12:41:53.263Z",
        "createdBy": {"id": "6273f303f6d63a28483fde12", "name": "Wauplin"},
        "updatedAt": "2023-10-26T12:41:53.263Z",
        "updatedBy": {"id": "6273f303f6d63a28483fde12", "name": "Wauplin"},
        "private": None,
        "state": "failed",
        "message": "Endpoint failed to deploy",
        "readyReplica": 0,
        "targetReplica": 1,
    },
}
# added for test_wait_update function
MOCK_UPDATE = {
    "name": "my-endpoint-name",
    "type": "protected",
    "accountId": None,
    "provider": {"vendor": "aws", "region": "us-east-1"},
    "healthRoute": "/health",
    "compute": {
        "accelerator": "cpu",
        "instanceType": "intel-icl",
        "instanceSize": "x2",
        "scaling": {"minReplica": 0, "maxReplica": 1},
    },
    "model": {
        "repository": "gpt2",
        "revision": "11c5a3d5811f50298f278a704980280950aedb10",
        "task": "text-generation",
        "framework": "pytorch",
        "image": {"huggingface": {}},
        "secret": {"token": "my-token"},
    },
    "status": {
        "createdAt": "2023-10-26T12:41:53.263078506Z",
        "createdBy": {"id": "6273f303f6d63a28483fde12", "name": "Wauplin"},
        "updatedAt": "2023-10-26T12:41:53.263079138Z",
        "updatedBy": {"id": "6273f303f6d63a28483fde12", "name": "Wauplin"},
        "private": None,
        "state": "updating",
        "url": "https://vksrvs8pc1xnifhq.us-east-1.aws.endpoints.huggingface.cloud",
        "message": "Endpoint waiting for the update",
        "readyReplica": 0,
        "targetReplica": 1,
    },
}


def test_from_raw_initialization():
    """Test InferenceEndpoint is correctly initialized from raw dict."""
    endpoint = InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo")

    # Main attributes parsed correctly
    assert endpoint.name == "my-endpoint-name"
    assert endpoint.namespace == "foo"
    assert endpoint.repository == "gpt2"
    assert endpoint.framework == "pytorch"
    assert endpoint.status == "pending"
    assert endpoint.revision == "11c5a3d5811f50298f278a704980280950aedb10"
    assert endpoint.task == "text-generation"
    assert endpoint.type == "protected"
    assert endpoint.health_route == "/health"

    # Datetime parsed correctly
    assert endpoint.created_at == datetime(2023, 10, 26, 12, 41, 53, 263078, tzinfo=timezone.utc)
    assert endpoint.updated_at == datetime(2023, 10, 26, 12, 41, 53, 263079, tzinfo=timezone.utc)

    # Not initialized yet
    assert endpoint.url is None

    # Raw dict still accessible
    assert endpoint.raw == MOCK_INITIALIZING


def test_from_raw_with_hf_api():
    """Test that the HfApi is correctly passed to the InferenceEndpoint."""
    endpoint = InferenceEndpoint.from_raw(
        MOCK_INITIALIZING, namespace="foo", api=HfApi(library_name="my-library", token="hf_***")
    )
    assert endpoint._api.library_name == "my-library"
    assert endpoint._api.token == "hf_***"


def test_get_client_not_ready():
    """Test clients are not created when endpoint is not ready."""
    endpoint = InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo")

    with pytest.raises(InferenceEndpointError):
        assert endpoint.client

    with pytest.raises(InferenceEndpointError):
        assert endpoint.async_client


def test_get_client_ready():
    """Test clients are created correctly when endpoint is ready."""
    endpoint = InferenceEndpoint.from_raw(MOCK_RUNNING, namespace="foo", token="my-token")

    # Endpoint is ready
    assert endpoint.status == "running"
    assert endpoint.url == "https://vksrvs8pc1xnifhq.us-east-1.aws.endpoints.huggingface.cloud"
    assert endpoint.health_route == "/health"

    # => Client available
    client = endpoint.client
    assert isinstance(client, InferenceClient)
    assert client.token == "my-token"

    # => AsyncClient available
    async_client = endpoint.async_client
    assert isinstance(async_client, AsyncInferenceClient)
    assert async_client.token == "my-token"


@patch("huggingface_hub.hf_api.HfApi.get_inference_endpoint")
def test_fetch(mock_get: Mock):
    endpoint = InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo")

    mock_get.return_value = InferenceEndpoint.from_raw(MOCK_RUNNING, namespace="foo")
    endpoint.fetch()

    assert endpoint.status == "running"
    assert endpoint.url == "https://vksrvs8pc1xnifhq.us-east-1.aws.endpoints.huggingface.cloud"
    assert endpoint.health_route == "/health"


@patch("huggingface_hub._inference_endpoints.get_session")
@patch("huggingface_hub.hf_api.HfApi.get_inference_endpoint")
def test_wait_until_running(mock_get: Mock, mock_session: Mock):
    """Test waits until the endpoint is ready."""
    endpoint = InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo")

    mock_get.side_effect = [
        InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo"),
        InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo"),
        InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo"),
        InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo"),
        InferenceEndpoint.from_raw(MOCK_RUNNING, namespace="foo"),
        InferenceEndpoint.from_raw(MOCK_RUNNING, namespace="foo"),
    ]
    mock_session.return_value = Mock()
    mock_session.return_value.get.side_effect = [
        Mock(status_code=400),  # url is provisioned but not yet ready
        Mock(status_code=200),  # endpoint is ready
    ]

    endpoint.wait(refresh_every=0.01)

    assert endpoint.status == "running"
    assert len(mock_get.call_args_list) == 6

    # Ensure the health route has been called
    assert mock_session.return_value.get.call_count == 2
    for call in mock_session.return_value.get.call_args_list:
        assert call[0][0] == "https://vksrvs8pc1xnifhq.us-east-1.aws.endpoints.huggingface.cloud/health"


@patch("huggingface_hub.hf_api.HfApi.get_inference_endpoint")
def test_wait_timeout(mock_get: Mock):
    """Test waits until timeout error is raised."""
    endpoint = InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo")

    mock_get.side_effect = [
        InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo"),
        InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo"),
        InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo"),
        InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo"),
    ]
    with pytest.raises(InferenceEndpointTimeoutError):
        endpoint.wait(timeout=0.1, refresh_every=0.05)

    assert endpoint.status == "pending"
    assert len(mock_get.call_args_list) == 2


@patch("huggingface_hub.hf_api.HfApi.get_inference_endpoint")
def test_wait_failed(mock_get: Mock):
    """Test waits until timeout error is raised."""
    endpoint = InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo")

    mock_get.side_effect = [
        InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo"),
        InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo"),
        InferenceEndpoint.from_raw(MOCK_FAILED, namespace="foo"),
    ]
    with pytest.raises(InferenceEndpointError, match=".*failed to deploy.*"):
        endpoint.wait(refresh_every=0.001)


@patch("huggingface_hub.hf_api.HfApi.get_inference_endpoint")
@patch("huggingface_hub._inference_endpoints.get_session")
def test_wait_update(mock_get_session, mock_get_inference_endpoint):
    """Test that wait() returns when the endpoint transitions to running."""
    endpoint = InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo")
    # Create an iterator that yields three MOCK_UPDATE responses,and then infinitely yields MOCK_RUNNING responses.
    responses = chain(
        [InferenceEndpoint.from_raw(MOCK_UPDATE, namespace="foo")] * 3,
        repeat(InferenceEndpoint.from_raw(MOCK_RUNNING, namespace="foo")),
    )
    mock_get_inference_endpoint.side_effect = lambda *args, **kwargs: next(responses)

    # Patch the get_session().get() call to always return a fake response with status_code 200.
    fake_response = MagicMock()
    fake_response.status_code = 200
    mock_get_session.return_value.get.return_value = fake_response

    endpoint.wait(refresh_every=0.05)
    assert endpoint.status == "running"


@patch("huggingface_hub.hf_api.HfApi.pause_inference_endpoint")
def test_pause(mock: Mock):
    """Test `pause` calls the correct alias."""
    endpoint = InferenceEndpoint.from_raw(MOCK_RUNNING, namespace="foo")
    mock.return_value = InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo")
    endpoint.pause()
    mock.assert_called_once_with(namespace="foo", name="my-endpoint-name", token=None)


@patch("huggingface_hub.hf_api.HfApi.resume_inference_endpoint")
def test_resume(mock: Mock):
    """Test `resume` calls the correct alias."""
    endpoint = InferenceEndpoint.from_raw(MOCK_RUNNING, namespace="foo")
    mock.return_value = InferenceEndpoint.from_raw(MOCK_INITIALIZING, namespace="foo")
    endpoint.resume()
    mock.assert_called_once_with(namespace="foo", name="my-endpoint-name", token=None, running_ok=True)
