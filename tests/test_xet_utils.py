from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch

from huggingface_hub import constants
from huggingface_hub.utils._xet import (
    XetFileData,
    _fetch_xet_connection_info_with_url,
    parse_xet_connection_info_from_headers,
    parse_xet_file_data_from_response,
    refresh_xet_connection_info,
)


def test_parse_valid_headers_file_info() -> None:
    mock_response = MagicMock()
    mock_response.headers = {
        "X-Xet-Hash": "sha256:abcdef",
        "X-Xet-Refresh-Route": "/api/refresh",
    }
    mock_response.links = {}

    file_data = parse_xet_file_data_from_response(mock_response)

    assert file_data is not None
    assert file_data.refresh_route == "/api/refresh"
    assert file_data.file_hash == "sha256:abcdef"


def test_parse_valid_headers_file_info_with_link() -> None:
    mock_response = MagicMock()
    mock_response.headers = {
        "X-Xet-Hash": "sha256:abcdef",
    }
    mock_response.links = {
        "xet-auth": {"url": "/api/refresh"},
    }

    file_data = parse_xet_file_data_from_response(mock_response)

    assert file_data is not None
    assert file_data.refresh_route == "/api/refresh"
    assert file_data.file_hash == "sha256:abcdef"


def test_parse_invalid_headers_file_info() -> None:
    mock_response = MagicMock()
    mock_response.headers = {"X-foo": "bar"}
    mock_response.links = {}
    assert parse_xet_file_data_from_response(mock_response) is None


@pytest.mark.parametrize(
    "refresh_route, expected_refresh_route",
    [
        (
            "/api/refresh",
            "/api/refresh",
        ),
        (
            "https://huggingface.co/api/refresh",
            "https://xet.example.com/api/refresh",
        ),
    ],
)
def test_parse_header_file_info_with_endpoint(refresh_route: str, expected_refresh_route: str) -> None:
    mock_response = MagicMock()
    mock_response.headers = {
        "X-Xet-Hash": "sha256:abcdef",
        "X-Xet-Refresh-Route": refresh_route,
    }
    mock_response.links = {}

    file_data = parse_xet_file_data_from_response(mock_response, endpoint="https://xet.example.com")

    assert file_data is not None
    assert file_data.refresh_route == expected_refresh_route
    assert file_data.file_hash == "sha256:abcdef"


def test_parse_valid_headers_connection_info() -> None:
    headers = {
        "X-Xet-Cas-Url": "https://xet.example.com",
        "X-Xet-Access-Token": "xet_token_abc",
        "X-Xet-Token-Expiration": "1234567890",
    }

    connection_info = parse_xet_connection_info_from_headers(headers)

    assert connection_info is not None
    assert connection_info.endpoint == "https://xet.example.com"
    assert connection_info.access_token == "xet_token_abc"
    assert connection_info.expiration_unix_epoch == 1234567890


def test_parse_valid_headers_full() -> None:
    mock_response = MagicMock()
    mock_response.headers = {
        "X-Xet-Cas-Url": "https://xet.example.com",
        "X-Xet-Access-Token": "xet_token_abc",
        "X-Xet-Token-Expiration": "1234567890",
        "X-Xet-Refresh-Route": "/api/refresh",
        "X-Xet-Hash": "sha256:abcdef",
    }
    mock_response.links = {}

    file_metadata = parse_xet_file_data_from_response(mock_response)
    connection_info = parse_xet_connection_info_from_headers(mock_response.headers)

    assert file_metadata is not None
    assert file_metadata.refresh_route == "/api/refresh"
    assert file_metadata.file_hash == "sha256:abcdef"

    assert connection_info is not None
    assert connection_info.endpoint == "https://xet.example.com"
    assert connection_info.access_token == "xet_token_abc"
    assert connection_info.expiration_unix_epoch == 1234567890


@pytest.mark.parametrize(
    "missing_key",
    [
        "X-Xet-Cas-Url",
        "X-Xet-Access-Token",
        "X-Xet-Token-Expiration",
    ],
)
def test_parse_missing_required_header(missing_key: str) -> None:
    headers = {
        "X-Xet-Cas-Url": "https://xet.example.com",
        "X-Xet-Access-Token": "xet_token_abc",
        "X-Xet-Token-Expiration": "1234567890",
    }

    # Remove the key to test
    headers.pop(missing_key)

    connection_info = parse_xet_connection_info_from_headers(headers)
    assert connection_info is None


def test_parse_invalid_expiration() -> None:
    """Test parsing headers with invalid expiration format returns None."""
    headers = {
        "X-Xet-Cas-Url": "https://xet.example.com",
        "X-Xet-Access-Token": "xet_token_abc",
        "X-Xet-Token-Expiration": "not-a-number",
    }

    connection_info = parse_xet_connection_info_from_headers(headers)
    assert connection_info is None


def test_refresh_metadata_success(mocker) -> None:
    # Mock headers for the refreshed response
    mock_response = MagicMock()
    mock_response.headers = {
        "X-Xet-Cas-Url": "https://example.xethub.hf.co",
        "X-Xet-Access-Token": "new_token",
        "X-Xet-Token-Expiration": "1234599999",
        "X-Xet-Refresh-Route": f"{constants.ENDPOINT}/api/models/username/repo_name/xet-read-token/token",
    }

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mocker.patch("huggingface_hub.utils._xet.get_session", return_value=mock_session)

    headers = {"user-agent": "user-agent-example"}
    refreshed_connection = refresh_xet_connection_info(
        file_data=XetFileData(
            refresh_route=f"{constants.ENDPOINT}/api/models/username/repo_name/xet-read-token/token",
            file_hash="sha256:abcdef",
        ),
        headers=headers,
    )

    # Verify the request
    expected_url = f"{constants.ENDPOINT}/api/models/username/repo_name/xet-read-token/token"
    mock_session.get.assert_called_once_with(
        headers=headers,
        url=expected_url,
        params=None,
    )

    assert refreshed_connection.endpoint == "https://example.xethub.hf.co"
    assert refreshed_connection.access_token == "new_token"
    assert refreshed_connection.expiration_unix_epoch == 1234599999


def test_refresh_metadata_custom_endpoint(mocker) -> None:
    custom_endpoint = "https://custom.xethub.hf.co"

    # Mock headers for the refreshed response
    mock_response = MagicMock()
    mock_response.headers = {
        "X-Xet-Cas-Url": "https://custom.xethub.hf.co",
        "X-Xet-Access-Token": "new_token",
        "X-Xet-Token-Expiration": "1234599999",
    }

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mocker.patch("huggingface_hub.utils._xet.get_session", return_value=mock_session)

    headers = {"user-agent": "user-agent-example"}
    refresh_xet_connection_info(
        file_data=XetFileData(
            refresh_route=f"{custom_endpoint}/api/models/username/repo_name/xet-read-token/token",
            file_hash="sha256:abcdef",
        ),
        headers=headers,
    )

    # Verify the request used the custom endpoint
    expected_url = f"{custom_endpoint}/api/models/username/repo_name/xet-read-token/token"
    mock_session.get.assert_called_once_with(
        headers=headers,
        url=expected_url,
        params=None,
    )


def test_refresh_metadata_missing_refresh_route() -> None:
    # Create metadata without refresh_route
    headers = {"user-agent": "user-agent-example"}

    # Verify it raises ValueError
    with pytest.raises(ValueError, match="The provided xet metadata does not contain a refresh endpoint."):
        refresh_xet_connection_info(
            file_data=XetFileData(
                refresh_route=None,
                file_hash="sha256:abcdef",
            ),
            headers=headers,
        )


def test_fetch_xet_metadata_with_url(mocker) -> None:
    mock_response = MagicMock()
    mock_response.headers = {
        "X-Xet-Cas-Url": "https://example.xethub.hf.co",
        "X-Xet-Access-Token": "xet_token123",
        "X-Xet-Token-Expiration": "1234567890",
    }

    # Mock the session.get method
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mocker.patch("huggingface_hub.utils._xet.get_session", return_value=mock_session)

    # Call the function
    url = "https://example.xethub.hf.co/api/models/username/repo_name/xet-read-token/token"
    headers = {"user-agent": "user-agent-example"}
    metadata = _fetch_xet_connection_info_with_url(url=url, headers=headers)

    # Verify the request
    mock_session.get.assert_called_once_with(
        headers=headers,
        url=url,
        params=None,
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
        _fetch_xet_connection_info_with_url(url=url, headers=headers)


def test_env_var_hf_hub_disable_xet() -> None:
    """Test that setting HF_HUB_DISABLE_XET results in is_xet_available() returning False."""
    from huggingface_hub.utils._runtime import is_xet_available

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_DISABLE_XET", True)

    assert not is_xet_available()
