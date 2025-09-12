import os
import threading
import time
import unittest
from multiprocessing import Process, Queue
from typing import Generator, Optional
from unittest.mock import Mock, call, patch
from uuid import UUID

import httpx
import pytest
from httpx import ConnectTimeout, HTTPError

from huggingface_hub.constants import ENDPOINT
from huggingface_hub.errors import OfflineModeIsEnabled
from huggingface_hub.utils._http import (
    HfHubTransport,
    _adjust_range_header,
    default_client_factory,
    fix_hf_endpoint_in_url,
    get_session,
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
        assert response is self.mock_request()

    def test_backoff_3_calls(self) -> None:
        """Test `http_backoff` with 2 fails."""
        response_mock = Mock()
        self.mock_request.side_effect = (ValueError(), ValueError(), response_mock)
        response = http_backoff(  # retry on ValueError, instant retry
            "GET", URL, retry_on_exceptions=ValueError, base_wait_time=0.0
        )
        assert self.mock_request.call_count == 3
        self.mock_request.assert_has_calls(
            calls=[
                call(method="GET", url=URL),
                call(method="GET", url=URL),
                call(method="GET", url=URL),
            ]
        )
        assert response is response_mock

    def test_backoff_on_exception_until_max(self) -> None:
        """Test `http_backoff` until max limit is reached with exceptions."""
        self.mock_request.side_effect = ConnectTimeout("Connection timeout")

        with self.assertRaises(ConnectTimeout):
            http_backoff("GET", URL, base_wait_time=0.0, max_retries=3)

        assert self.mock_request.call_count == 4

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

        assert self.mock_request.call_count == 4

    def test_backoff_on_exceptions_and_status_codes(self) -> None:
        """Test `http_backoff` until max limit with status codes and exceptions."""
        mock_503 = Mock()
        mock_503.status_code = 503
        self.mock_request.side_effect = (mock_503, ConnectTimeout("Connection timeout"))

        with self.assertRaises(ConnectTimeout):
            http_backoff("GET", URL, base_wait_time=0.0, max_retries=1)

        assert self.mock_request.call_count == 2

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

        assert self.mock_request.call_count == 4
        assert response is mock_200

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

        assert self.mock_request.call_count == 6

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
        assert client.follow_redirects
        assert client.timeout is not None
        # Check that it's using the HfHubTransport
        assert isinstance(client._transport, HfHubTransport)

    def test_set_configuration(self) -> None:
        set_client_factory(self._factory)

        # Check headers have been set correctly
        client = get_session()
        assert client.headers != {"x-test-header": "4"}
        assert client.headers["x-test-header"] == "4"

    def test_get_session_twice(self):
        client_1 = get_session()
        client_2 = get_session()
        assert client_1 is client_2  # exact same instance

    def test_get_session_twice_but_reconfigure_in_between(self):
        """Reconfiguring the session clears the cache."""
        client_1 = get_session()
        set_client_factory(self._factory)

        client_2 = get_session()
        assert client_1 is not client_2
        assert client_1.headers.get("x-test-header" is None)
        assert client_2.headers["x-test-header"] == "4"

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
            assert main_client is clients[i]
            for j in range(N):
                assert clients[i] is clients[j]

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
        assert repr(main_client) == child_client


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
        assert request_id in response_id
        assert _is_uuid(request_id)

    def test_request_id_is_unique(self):
        response_1 = get_session().get(self.api_endpoint)
        response_2 = get_session().get(self.api_endpoint)

        request_id_1 = response_1.request.headers["X-Amzn-Trace-Id"]
        request_id_2 = response_2.request.headers["X-Amzn-Trace-Id"]
        assert request_id_1 != request_id_2

        assert _is_uuid(request_id_1)
        assert _is_uuid(request_id_2)

    def test_request_id_not_overwritten(self):
        response = get_session().get(self.api_endpoint, headers={"x-request-id": "custom-id"})

        request_id = response.request.headers["x-request-id"]
        assert request_id == "custom-id"

        response_id = response.headers["x-request-id"]
        assert response_id == "custom-id"


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
