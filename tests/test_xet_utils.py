import multiprocessing
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from _pytest.monkeypatch import MonkeyPatch

from huggingface_hub.utils._xet import (
    XetSessionHolder,
    XetTokenType,
    parse_xet_file_data_from_response,
    xet_connection_info_refresh_url,
)


pytestmark = pytest.mark.xet


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


def test_parse_valid_headers_full() -> None:
    mock_response = MagicMock()
    mock_response.headers = {
        "X-Xet-Refresh-Route": "/api/refresh",
        "X-Xet-Hash": "sha256:abcdef",
    }
    mock_response.links = {}

    file_metadata = parse_xet_file_data_from_response(mock_response)
    assert file_metadata is not None
    assert file_metadata.refresh_route == "/api/refresh"
    assert file_metadata.file_hash == "sha256:abcdef"


def test_env_var_hf_hub_disable_xet() -> None:
    """Test that setting HF_HUB_DISABLE_XET results in is_xet_available() returning False."""
    from huggingface_hub.utils._runtime import is_xet_available

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_DISABLE_XET", True)

    assert not is_xet_available()


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
                os.write(w_fd, b"SUCCESS" if ok else b"FAILURE")
            except Exception:
                os.write(w_fd, b"FAILURE")
            finally:
                os.close(w_fd)
                os._exit(0)
        else:
            # ---- parent process ----
            os.close(w_fd)
            result = os.read(r_fd, 7)
            os.close(r_fd)
            os.waitpid(child_pid, 0)
            assert result == b"SUCCESS", "Child process reported fork-safety failure"


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


@pytest.mark.parametrize(
    "kwargs, expected_suffix",
    [
        (
            {"token_type": XetTokenType.WRITE, "repo_id": "user/mymodel", "repo_type": "model", "revision": "main"},
            "/api/models/user/mymodel/xet-write-token/main",
        ),
        (
            {"token_type": XetTokenType.WRITE, "repo_id": "user/mymodel", "repo_type": "model", "revision": None},
            "/api/models/user/mymodel/xet-write-token/None",
        ),
        (
            {"token_type": XetTokenType.WRITE, "repo_id": "user/mybucket", "repo_type": "bucket", "revision": None},
            "/api/buckets/user/mybucket/xet-write-token",
        ),
        (
            {
                "token_type": XetTokenType.WRITE,
                "repo_id": "user/mybucket",
                "repo_type": "bucket",
                "revision": "some-rev",
            },
            "/api/buckets/user/mybucket/xet-write-token/some-rev",
        ),
        (
            {"token_type": XetTokenType.READ, "repo_id": "user/myds", "repo_type": "dataset", "revision": "v1"},
            "/api/datasets/user/myds/xet-read-token/v1",
        ),
    ],
)
def test_xet_connection_info_refresh_url(kwargs, expected_suffix):
    endpoint = "https://huggingface.co"
    url = xet_connection_info_refresh_url(**kwargs, endpoint=endpoint)
    assert url == endpoint + expected_suffix
