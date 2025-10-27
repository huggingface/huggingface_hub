import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing import Process, Queue
from typing import Generator, Optional
from unittest.mock import Mock, call, patch
from urllib.parse import urlparse
from uuid import UUID

import httpx
import pytest
from httpx import ConnectTimeout, HTTPError

from huggingface_hub.constants import ENDPOINT
from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled
from huggingface_hub.utils._http import (
    _adjust_range_header,
    default_client_factory,
    fix_hf_endpoint_in_url,
    get_async_session,
    get_session,
    hf_raise_for_status,
    http_backoff,
    set_client_factory,
)


URL = "https://www.google.com"


class TestHttpBackoff(unittest.TestCase):
    def setUp(self) -> None:
        get_session_mock = Mock()
        self.mock_request = get_session_mock().request

        self.patcher = patch("huggingface_hub.utils._http.get_session", get_session_mock)
        self.patcher.start()

    def tearDown(self) -> None:
        self.patcher.stop()

    def test_backoff_no_errors(self) -> None:
        """Test normal usage of `http_backoff`."""
        data_mock = Mock()
        response = http_backoff("GET", URL, data=data_mock)
        self.mock_request.assert_called_once_with(method="GET", url=URL, data=data_mock)
        self.assertIs(response, self.mock_request())

    def test_backoff_3_calls(self) -> None:
        """Test `http_backoff` with 2 fails."""
        response_mock = Mock()
        self.mock_request.side_effect = (ValueError(), ValueError(), response_mock)
        response = http_backoff(  # retry on ValueError, instant retry
            "GET", URL, retry_on_exceptions=ValueError, base_wait_time=0.0
        )
        self.assertEqual(self.mock_request.call_count, 3)
        self.mock_request.assert_has_calls(
            calls=[
                call(method="GET", url=URL),
                call(method="GET", url=URL),
                call(method="GET", url=URL),
            ]
        )
        self.assertIs(response, response_mock)

    def test_backoff_on_exception_until_max(self) -> None:
        """Test `http_backoff` until max limit is reached with exceptions."""
        self.mock_request.side_effect = ConnectTimeout("Connection timeout")

        with self.assertRaises(ConnectTimeout):
            http_backoff("GET", URL, base_wait_time=0.0, max_retries=3)

        self.assertEqual(self.mock_request.call_count, 4)

    def test_backoff_on_status_code_until_max(self) -> None:
        """Test `http_backoff` until max limit is reached with status codes."""
        mock_503 = Mock()
        mock_503.status_code = 503
        mock_504 = Mock()
        mock_504.status_code = 504
        mock_504.raise_for_status.side_effect = HTTPError("HTTP Error")
        self.mock_request.side_effect = (mock_503, mock_504, mock_503, mock_504)

        with self.assertRaises(HTTPError):
            http_backoff(
                "GET",
                URL,
                base_wait_time=0.0,
                max_retries=3,
                retry_on_status_codes=(503, 504),
            )

        self.assertEqual(self.mock_request.call_count, 4)

    def test_backoff_on_exceptions_and_status_codes(self) -> None:
        """Test `http_backoff` until max limit with status codes and exceptions."""
        mock_503 = Mock()
        mock_503.status_code = 503
        self.mock_request.side_effect = (mock_503, ConnectTimeout("Connection timeout"))

        with self.assertRaises(ConnectTimeout):
            http_backoff("GET", URL, base_wait_time=0.0, max_retries=1)

        self.assertEqual(self.mock_request.call_count, 2)

    def test_backoff_on_valid_status_code(self) -> None:
        """Test `http_backoff` until max limit with a valid status code.

        Quite a corner case: the user wants to retry is status code is 200. Requests are
        retried but in the end, the HTTP 200 response is returned if the server returned
        only 200 responses.
        """
        mock_200 = Mock()
        mock_200.status_code = 200
        self.mock_request.side_effect = (mock_200, mock_200, mock_200, mock_200)

        response = http_backoff("GET", URL, base_wait_time=0.0, max_retries=3, retry_on_status_codes=200)

        self.assertEqual(self.mock_request.call_count, 4)
        self.assertIs(response, mock_200)

    def test_backoff_sleep_time(self) -> None:
        """Test `http_backoff` sleep time goes exponential until max limit.

        Since timing between 2 requests is sleep duration + some other stuff, this test
        can be unstable. However, sleep durations between 10ms and 50ms should be enough
        to make the approximation that measured durations are the "sleep time" waited by
        `http_backoff`. If this is not the case, just increase `base_wait_time`,
        `max_wait_time` and `expected_sleep_times` with bigger values.
        """
        sleep_times = []

        def _side_effect_timer() -> Generator[ConnectTimeout, None, None]:
            t0 = time.time()
            while True:
                yield ConnectTimeout("Connection timeout")
                t1 = time.time()
                sleep_times.append(round(t1 - t0, 1))
                t0 = t1

        self.mock_request.side_effect = _side_effect_timer()

        with self.assertRaises(ConnectTimeout):
            http_backoff("GET", URL, base_wait_time=0.1, max_wait_time=0.5, max_retries=5)

        self.assertEqual(self.mock_request.call_count, 6)

        # Assert sleep times are exponential until plateau
        expected_sleep_times = [0.1, 0.2, 0.4, 0.5, 0.5]
        self.assertListEqual(sleep_times, expected_sleep_times)


class TestConfigureSession(unittest.TestCase):
    def setUp(self) -> None:
        # Reconfigure + clear session cache between each test
        set_client_factory(default_client_factory)

    @classmethod
    def tearDownClass(cls) -> None:
        # Clear all sessions after tests
        set_client_factory(default_client_factory)

    @staticmethod
    def _factory() -> httpx.Client:
        client = httpx.Client()
        client.headers.update({"x-test-header": "4"})
        return client

    def test_default_configuration(self) -> None:
        client = get_session()
        # Check httpx.Client default configuration
        self.assertTrue(client.follow_redirects)
        self.assertIsNotNone(client.timeout)

    def test_set_configuration(self) -> None:
        set_client_factory(self._factory)

        # Check headers have been set correctly
        client = get_session()
        self.assertNotEqual(client.headers, {"x-test-header": "4"})
        self.assertEqual(client.headers["x-test-header"], "4")

    def test_get_session_twice(self):
        client_1 = get_session()
        client_2 = get_session()
        self.assertIs(client_1, client_2)  # exact same instance

    def test_get_session_twice_but_reconfigure_in_between(self):
        """Reconfiguring the session clears the cache."""
        client_1 = get_session()
        set_client_factory(self._factory)

        client_2 = get_session()
        self.assertIsNot(client_1, client_2)
        self.assertIsNone(client_1.headers.get("x-test-header"))
        self.assertEqual(client_2.headers["x-test-header"], "4")

    def test_get_session_multiple_threads(self):
        N = 3
        clients = [None] * N

        def _get_session_in_thread(index: int) -> None:
            time.sleep(0.1)
            clients[index] = get_session()

        # Get main thread client
        main_client = get_session()

        # Start 3 threads and get clients in each of them
        threads = [threading.Thread(target=_get_session_in_thread, args=(index,)) for index in range(N)]
        for th in threads:
            th.start()
            print(th)
        for th in threads:
            th.join()

        # Check all clients are the same instance (httpx is thread-safe)
        for i in range(N):
            self.assertIs(main_client, clients[i])
            for j in range(N):
                self.assertIs(clients[i], clients[j])

    @unittest.skipIf(os.name == "nt", "Works differently on Windows.")
    def test_get_session_in_forked_process(self):
        # Get main process client
        main_client = get_session()

        def _child_target():
            # Put `repr(client)` in queue because putting the `Client` object directly would duplicate it.
            # Repr looks like this: "<httpx.Client object at 0x7f5adcc41e40>"
            process_queue.put(repr(get_session()))

        # Fork a new process and get client in it
        process_queue = Queue()
        Process(target=_child_target).start()
        child_client = process_queue.get()

        # Check clients are the same instance
        self.assertEqual(repr(main_client), child_client)


class OfflineModeSessionTest(unittest.TestCase):
    def tearDown(self) -> None:
        return super().tearDown()

    @patch("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    def test_offline_mode(self):
        set_client_factory(default_client_factory)
        client = get_session()
        with self.assertRaises(OfflineModeIsEnabled):
            client.get("https://huggingface.co")


class TestUniqueRequestId(unittest.TestCase):
    api_endpoint = ENDPOINT + "/api/tasks"  # any endpoint is fine

    def test_request_id_is_used_by_server(self):
        response = get_session().get(self.api_endpoint)

        request_id = response.request.headers.get("X-Amzn-Trace-Id")
        response_id = response.headers.get("x-request-id")
        self.assertIn(request_id, response_id)
        self.assertTrue(_is_uuid(request_id))

    def test_request_id_is_unique(self):
        response_1 = get_session().get(self.api_endpoint)
        response_2 = get_session().get(self.api_endpoint)

        request_id_1 = response_1.request.headers["X-Amzn-Trace-Id"]
        request_id_2 = response_2.request.headers["X-Amzn-Trace-Id"]
        self.assertNotEqual(request_id_1, request_id_2)

        self.assertTrue(_is_uuid(request_id_1))
        self.assertTrue(_is_uuid(request_id_2))

    def test_request_id_not_overwritten(self):
        response = get_session().get(self.api_endpoint, headers={"x-request-id": "custom-id"})

        request_id = response.request.headers["x-request-id"]
        self.assertEqual(request_id, "custom-id")

        response_id = response.headers["x-request-id"]
        self.assertEqual(response_id, "custom-id")


def _is_uuid(string: str) -> bool:
    # Taken from https://stackoverflow.com/a/33245493
    try:
        uuid_obj = UUID(string)
    except ValueError:
        return False
    return str(uuid_obj) == string


@pytest.mark.parametrize(
    ("base_url", "endpoint", "expected_url"),
    [
        # Staging url => unchanged
        ("https://hub-ci.huggingface.co/resolve/...", None, "https://hub-ci.huggingface.co/resolve/..."),
        # Prod url => unchanged
        ("https://huggingface.co/resolve/...", None, "https://huggingface.co/resolve/..."),
        # Custom endpoint + staging url => fixed
        ("https://hub-ci.huggingface.co/api/models", "https://mirror.co", "https://mirror.co/api/models"),
        # Custom endpoint + prod url => fixed
        ("https://huggingface.co/api/models", "https://mirror.co", "https://mirror.co/api/models"),
    ],
)
def test_fix_hf_endpoint_in_url(base_url: str, endpoint: Optional[str], expected_url: str) -> None:
    assert fix_hf_endpoint_in_url(base_url, endpoint) == expected_url


def test_adjust_range_header():
    # Basic cases
    assert _adjust_range_header(None, 10) == "bytes=10-"
    assert _adjust_range_header("bytes=0-100", 10) == "bytes=10-100"
    assert _adjust_range_header("bytes=-100", 10) == "bytes=-90"
    assert _adjust_range_header("bytes=100-", 10) == "bytes=110-"

    with pytest.raises(RuntimeError):
        _adjust_range_header("invalid", 10)

    with pytest.raises(RuntimeError):
        _adjust_range_header("bytes=-", 10)

    # Multiple ranges
    with pytest.raises(ValueError):
        _adjust_range_header("bytes=0-100,200-300", 10)

    # Resume size exceeds range
    with pytest.raises(RuntimeError):
        _adjust_range_header("bytes=0-100", 150)
    with pytest.raises(RuntimeError):
        _adjust_range_header("bytes=-50", 100)


def test_proxy_env_is_used(monkeypatch):
    """Regression test for https://github.com/huggingface/transformers/issues/41301.

    Test is hacky and uses httpx internal attributes, but it works.
    We just need to test that proxies from env vars are used when creating the client.
    """
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example1.com:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example2.com:8181")

    set_client_factory(default_client_factory)
    client = get_session()
    mounts = client._mounts
    url_patterns = list(mounts.keys())
    assert len(url_patterns) == 2  # http and https

    http_url_pattern = next(url for url in url_patterns if url.pattern == "http://")
    http_proxy_url = mounts[http_url_pattern]._pool._proxy_url
    assert http_proxy_url.scheme == b"http"
    assert http_proxy_url.host == b"proxy.example1.com"
    assert http_proxy_url.port == 8080
    assert http_proxy_url.target == b"/"

    https_url_pattern = next(url for url in url_patterns if url.pattern == "https://")
    https_proxy_url = mounts[https_url_pattern]._pool._proxy_url
    assert https_proxy_url.scheme == b"http"
    assert https_proxy_url.host == b"proxy.example2.com"
    assert https_proxy_url.port == 8181
    assert https_proxy_url.target == b"/"

    # Reset
    set_client_factory(default_client_factory)


def test_client_get_request():
    # Check that sync client works
    client = get_session()
    response = client.get("https://huggingface.co")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_async_client_get_request():
    # Check that async client works
    client = get_async_session()
    response = await client.get("https://huggingface.co")
    assert response.status_code == 200


class FakeServerHandler(BaseHTTPRequestHandler):
    """Fake server handler to test client behavior."""

    def do_GET(self):
        parsed = urlparse(self.path)

        # Health check endpoint (always succeeds)
        if parsed.path == "/health":
            self._send_response(200, b"OK")
            return

        # Main endpoint (always fails with 500)
        self._send_response(500, b"This is a 500 error")

    def _send_response(self, status_code, body):
        self.send_response(status_code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module", autouse=True)
def fake_server():
    # Find a free port
    host, port = "127.0.0.1", 8000
    for port in range(port, 8100):
        try:
            server = HTTPServer((host, port), FakeServerHandler)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("Could not find a free port")

    url = f"http://{host}:{port}"

    # Start server in a separate thread and wait until it's ready
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    for _ in range(1000):  # up to 10 seconds
        try:
            if httpx.get(f"{url}/health", timeout=0.01).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.01)
    else:
        server.shutdown()
        raise RuntimeError("Fake server failed to start")

    yield url
    server.shutdown()


def _check_raise_status(response: httpx.Response):
    """Common assertions for 500 error tests."""
    with pytest.raises(HfHubHTTPError) as exc_info:
        hf_raise_for_status(response)
    assert exc_info.value.response.status_code == 500
    assert "This is a 500 error" in str(exc_info.value)


def test_raise_on_status_sync_non_stream(fake_server: str):
    response = get_session().get(fake_server)
    _check_raise_status(response)


def test_raise_on_status_sync_stream(fake_server: str):
    with get_session().stream("GET", fake_server) as response:
        _check_raise_status(response)


@pytest.mark.asyncio
async def test_raise_on_status_async_non_stream(fake_server: str):
    response = await get_async_session().get(fake_server)
    _check_raise_status(response)


@pytest.mark.asyncio
async def test_raise_on_status_async_stream(fake_server: str):
    async with get_async_session().stream("GET", fake_server) as response:
        _check_raise_status(response)
