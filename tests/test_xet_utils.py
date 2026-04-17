import multiprocessing
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from _pytest.monkeypatch import MonkeyPatch

from huggingface_hub import HfApi, constants
from huggingface_hub.utils._xet import (
    XetFileData,
    XetSessionHolder,
    XetTokenType,
    _fetch_xet_connection_info_with_url,
    fetch_xet_connection_info_from_repo_info,
    parse_xet_connection_info_from_headers,
    parse_xet_file_data_from_response,
    refresh_xet_connection_info,
)

from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import repo_name


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

    http_backoff_mock = mocker.patch("huggingface_hub.utils._xet.http_backoff", return_value=mock_response)

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
    http_backoff_mock.assert_called_once_with(
        "GET",
        expected_url,
        headers=headers,
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

    http_backoff_mock = mocker.patch("huggingface_hub.utils._xet.http_backoff", return_value=mock_response)

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
    http_backoff_mock.assert_called_once_with(
        "GET",
        expected_url,
        headers=headers,
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
    http_backoff_mock = mocker.patch("huggingface_hub.utils._xet.http_backoff", return_value=mock_response)

    # Call the function
    url = "https://example.xethub.hf.co/api/models/username/repo_name/xet-read-token/token"
    headers = {"user-agent": "user-agent-example"}
    metadata = _fetch_xet_connection_info_with_url(url=url, headers=headers)

    # Verify the request
    http_backoff_mock.assert_called_once_with(
        "GET",
        url,
        headers=headers,
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
    mocker.patch("huggingface_hub.utils._xet.http_backoff", return_value=mock_response)

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


def test_xet_token_reset_after_repo_deletion() -> None:
    """Test Xet token is reset after repo deletion.

    Regression test for https://github.com/huggingface/huggingface_hub/issues/3829
    """
    # Create a repo
    api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
    repo_id = api.create_repo(repo_id=repo_name()).repo_id

    # Get XET token
    xet_info_read = fetch_xet_connection_info_from_repo_info(
        token_type=XetTokenType.READ,
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        headers=api._build_hf_headers(),
    )

    xet_info_write = fetch_xet_connection_info_from_repo_info(
        token_type=XetTokenType.WRITE,
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        headers=api._build_hf_headers(),
    )

    # Recreate the repo
    api.delete_repo(repo_id=repo_id)
    api.create_repo(repo_id)

    # Get XET token for the new repo (same repo ID, same revision)
    xet_info_read_new = fetch_xet_connection_info_from_repo_info(
        token_type=XetTokenType.READ,
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        headers=api._build_hf_headers(),
    )
    xet_info_write_new = fetch_xet_connection_info_from_repo_info(
        token_type=XetTokenType.WRITE,
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        headers=api._build_hf_headers(),
    )

    # XET token must have changed (reset after repo deletion)
    assert xet_info_read.access_token != xet_info_read_new.access_token
    assert xet_info_write.access_token != xet_info_write_new.access_token


# ---------------------------------------------------------------------------
# Fork-safety tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="os.fork() not available on Windows")
def test_xet_session_holder_fork_safety_unit():
    """Unit test: XetSessionHolder detects fork and creates a fresh session in child.

    Uses os.fork() directly. The child process writes a pass/fail byte to a
    pipe and exits; the parent reads it and asserts success.
    """
    mock_parent = MagicMock(name="parent_session")
    mock_child = MagicMock(name="child_session")
    sessions = [mock_parent, mock_child]

    holder = XetSessionHolder()

    with patch("hf_xet.XetSession", side_effect=sessions):
        # Create session in the parent.
        parent_session = holder.get()
        assert parent_session is mock_parent
        parent_pid = os.getpid()
        assert holder._session_pid == parent_pid

        r_fd, w_fd = os.pipe()
        child_pid = os.fork()

        if child_pid == 0:
            # ---- child process ----
            os.close(r_fd)
            try:
                child_session = holder.get()
                ok = (
                    child_session is mock_child  # fresh session created
                    and holder._session_pid == os.getpid()  # PID updated
                    and holder._session_pid != parent_pid  # different from parent
                )
                os.write(w_fd, b"\x01" if ok else b"\x00")
            except Exception:
                os.write(w_fd, b"\x00")
            finally:
                os.close(w_fd)
                os._exit(0)
        else:
            # ---- parent process ----
            os.close(w_fd)
            result = os.read(r_fd, 1)
            os.close(r_fd)
            os.waitpid(child_pid, 0)
            assert result == b"\x01", "Child process reported fork-safety failure"


def _worker_get_session_pid(_):
    """Multiprocessing worker: create a XetSessionHolder and return its session PID."""
    holder = XetSessionHolder()
    with patch("hf_xet.XetSession", return_value=MagicMock(name="worker_session")):
        holder.get()
        return holder._session_pid


@pytest.mark.skipif(sys.platform == "win32", reason="fork start method not available on Windows")
def test_xet_session_holder_fork_safety_multiprocessing():
    """Integration test: XetSessionHolder works correctly in multiprocessing fork workers.

    Simulates a workload where the parent creates a session and then forks worker processes.
    Each worker must get its own fresh session rather than the inherited (broken) one.
    """
    holder = XetSessionHolder()

    with patch("hf_xet.XetSession", return_value=MagicMock(name="parent_session")):
        holder.get()
        parent_pid = os.getpid()
        assert holder._session_pid == parent_pid

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=2) as pool:
        worker_pids = pool.map(_worker_get_session_pid, range(2))

    # Each worker must have recorded its own PID (not the parent's).
    for wpid in worker_pids:
        assert wpid != parent_pid, f"Worker used parent's session PID {parent_pid}"
        assert wpid is not None
