from unittest.mock import MagicMock

import pytest

from huggingface_hub import constants
from huggingface_hub.utils._xet import (
    XetMetadata,
    _fetch_xet_metadata_with_url,
    parse_xet_headers,
    refresh_xet_metadata,
)


def test_parse_valid_headers_minimal() -> None:
    headers = {
        constants.HUGGINGFACE_HEADER_X_XET_ENDPOINT: "https://xet.example.com",
        constants.HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN: "xet_token_abc",
        constants.HUGGINGFACE_HEADER_X_XET_EXPIRATION: "1234567890",
    }

    metadata = parse_xet_headers(headers)

    assert metadata is not None
    assert metadata.endpoint == "https://xet.example.com"
    assert metadata.access_token == "xet_token_abc"
    assert metadata.expiration_unix_epoch == 1234567890
    assert metadata.refresh_route is None
    assert metadata.file_hash is None


def test_parse_valid_headers_full() -> None:
    headers = {
        constants.HUGGINGFACE_HEADER_X_XET_ENDPOINT: "https://xet.example.com",
        constants.HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN: "xet_token_abc",
        constants.HUGGINGFACE_HEADER_X_XET_EXPIRATION: "1234567890",
        constants.HUGGINGFACE_HEADER_X_XET_REFRESH_ROUTE: "/api/refresh",
        constants.HUGGINGFACE_HEADER_X_XET_HASH: "sha256:abcdef",
    }

    metadata = parse_xet_headers(headers)

    assert metadata is not None
    assert metadata.endpoint == "https://xet.example.com"
    assert metadata.access_token == "xet_token_abc"
    assert metadata.expiration_unix_epoch == 1234567890
    assert metadata.refresh_route == "/api/refresh"
    assert metadata.file_hash == "sha256:abcdef"


@pytest.mark.parametrize(
    "missing_key",
    [
        constants.HUGGINGFACE_HEADER_X_XET_ENDPOINT,
        constants.HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN,
        constants.HUGGINGFACE_HEADER_X_XET_EXPIRATION,
    ],
)
def test_parse_missing_required_header(missing_key: str) -> None:
    headers = {
        constants.HUGGINGFACE_HEADER_X_XET_ENDPOINT: "https://xet.example.com",
        constants.HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN: "xet_token_abc",
        constants.HUGGINGFACE_HEADER_X_XET_EXPIRATION: "1234567890",
    }

    # Remove the key to test
    headers.pop(missing_key)

    metadata = parse_xet_headers(headers)
    assert metadata is None


def test_parse_invalid_expiration() -> None:
    """Test parsing headers with invalid expiration format returns None."""
    headers = {
        constants.HUGGINGFACE_HEADER_X_XET_ENDPOINT: "https://xet.example.com",
        constants.HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN: "xet_token_abc",
        constants.HUGGINGFACE_HEADER_X_XET_EXPIRATION: "not-a-number",
    }

    metadata = parse_xet_headers(headers)
    assert metadata is None


def test_refresh_metadata_success(mocker) -> None:
    initial_metadata = XetMetadata(
        endpoint="https://example.xethub.hf.co",
        access_token="old_token",
        expiration_unix_epoch=1234567890,
        refresh_route="/api/models/username/repo_name/xet-read-token/token",
    )

    # Mock headers for the refreshed response
    mock_response = MagicMock()
    mock_response.headers = {
        constants.HUGGINGFACE_HEADER_X_XET_ENDPOINT: "https://example.xethub.hf.co",
        constants.HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN: "new_token",
        constants.HUGGINGFACE_HEADER_X_XET_EXPIRATION: "1234599999",
        constants.HUGGINGFACE_HEADER_X_XET_REFRESH_ROUTE: "/api/models/username/repo_name/xet-read-token/token",
    }

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mocker.patch("huggingface_hub.utils._xet.get_session", return_value=mock_session)

    headers = {"user-agent": "user-agent-example"}
    refreshed_metadata = refresh_xet_metadata(
        xet_metadata=initial_metadata,
        headers=headers,
    )

    # Verify the request
    expected_url = f"{constants.ENDPOINT}/api/models/username/repo_name/xet-read-token/token"
    mock_session.get.assert_called_once_with(
        headers=headers,
        url=expected_url,
    )

    assert refreshed_metadata.endpoint == "https://example.xethub.hf.co"
    assert refreshed_metadata.access_token == "new_token"
    assert refreshed_metadata.expiration_unix_epoch == 1234599999


def test_refresh_metadata_custom_endpoint(mocker) -> None:
    initial_metadata = XetMetadata(
        endpoint="https://example.xethub.hf.co",
        access_token="old_token",
        expiration_unix_epoch=1234567890,
        refresh_route="/api/models/username/repo_name/xet-read-token/token",
    )

    custom_endpoint = "https://custom.xethub.hf.co"

    # Mock headers for the refreshed response
    mock_response = MagicMock()
    mock_response.headers = {
        constants.HUGGINGFACE_HEADER_X_XET_ENDPOINT: "https://custom.xethub.hf.co",
        constants.HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN: "new_token",
        constants.HUGGINGFACE_HEADER_X_XET_EXPIRATION: "1234599999",
    }

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mocker.patch("huggingface_hub.utils._xet.get_session", return_value=mock_session)

    headers = {"user-agent": "user-agent-example"}
    refresh_xet_metadata(
        xet_metadata=initial_metadata,
        headers=headers,
        endpoint=custom_endpoint,
    )

    # Verify the request used the custom endpoint
    expected_url = f"{custom_endpoint}/api/models/username/repo_name/xet-read-token/token"
    mock_session.get.assert_called_once_with(
        headers=headers,
        url=expected_url,
    )


def test_refresh_metadata_missing_refresh_route() -> None:
    # Create metadata without refresh_route
    metadata = XetMetadata(
        endpoint="https://example.xethub.hf.co",
        access_token="token123",
        expiration_unix_epoch=1234567890,
        # No refresh_route
    )

    headers = {"user-agent": "user-agent-example"}

    # Verify it raises ValueError
    with pytest.raises(ValueError, match="The provided xet metadata does not contain a refresh endpoint."):
        refresh_xet_metadata(
            xet_metadata=metadata,
            headers=headers,
        )


def test_fetch_xet_metadata_with_url(mocker) -> None:
    mock_response = MagicMock()
    mock_response.headers = {
        constants.HUGGINGFACE_HEADER_X_XET_ENDPOINT: "https://example.xethub.hf.co",
        constants.HUGGINGFACE_HEADER_X_XET_ACCESS_TOKEN: "xet_token123",
        constants.HUGGINGFACE_HEADER_X_XET_EXPIRATION: "1234567890",
    }

    # Mock the session.get method
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mocker.patch("huggingface_hub.utils._xet.get_session", return_value=mock_session)

    # Call the function
    url = "https://example.xethub.hf.co/api/models/username/repo_name/xet-read-token/token"
    headers = {"user-agent": "user-agent-example"}
    metadata = _fetch_xet_metadata_with_url(url=url, headers=headers)

    # Verify the request
    mock_session.get.assert_called_once_with(
        headers=headers,
        url=url,
    )

    # Verify returned metadata
    assert metadata.endpoint == "https://example.xethub.hf.co"
    assert metadata.access_token == "xet_token123"
    assert metadata.expiration_unix_epoch == 1234567890


def test_fetch_xet_metadata_with_url_invalid_response(mocker) -> None:
    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "application/json"}  # No XET headers

    # Mock the session.get method
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mocker.patch("huggingface_hub.utils._xet.get_session", return_value=mock_session)

    url = "https://example.xethub.hf.co/api/models/username/repo_name/xet-read-token/token"
    headers = {"user-agent": "user-agent-example"}

    with pytest.raises(ValueError, match="Xet headers have not been correctly set by the server."):
        _fetch_xet_metadata_with_url(url=url, headers=headers)
