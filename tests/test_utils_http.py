import threading
import time
import unittest
from typing import Generator
from unittest.mock import Mock, call, patch

import requests
from requests import ConnectTimeout, HTTPError

from huggingface_hub.utils._http import configure_http_backend, get_session, http_backoff


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
        self.mock_request.side_effect = ConnectTimeout()

        with self.assertRaises(ConnectTimeout):
            http_backoff("GET", URL, base_wait_time=0.0, max_retries=3)

        self.assertEqual(self.mock_request.call_count, 4)

    def test_backoff_on_status_code_until_max(self) -> None:
        """Test `http_backoff` until max limit is reached with status codes."""
        mock_503 = Mock()
        mock_503.status_code = 503
        mock_504 = Mock()
        mock_504.status_code = 504
        mock_504.raise_for_status.side_effect = HTTPError()
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
        self.mock_request.side_effect = (mock_503, ConnectTimeout())

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
                yield ConnectTimeout()
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
        configure_http_backend()

    @classmethod
    def tearDownClass(cls) -> None:
        # Clear all sessions after tests
        configure_http_backend()

    @staticmethod
    def _factory() -> requests.Session:
        session = requests.Session()
        session.headers.update({"x-test-header": 4})
        return session

    def test_default_configuration(self) -> None:
        session = get_session()
        self.assertEqual(session.headers["connection"], "keep-alive")  # keep connection alive by default
        self.assertIsNone(session.auth)
        self.assertEqual(session.proxies, {})
        self.assertEqual(session.verify, True)
        self.assertIsNone(session.cert)
        self.assertEqual(session.max_redirects, 30)
        self.assertEqual(session.trust_env, True)
        self.assertEqual(session.hooks, {"response": []})

    def test_set_configuration(self) -> None:
        configure_http_backend(backend_factory=self._factory)

        # Check headers have been set correctly
        session = get_session()
        self.assertNotEqual(session.headers, {"x-test-header": 4})
        self.assertEqual(session.headers["x-test-header"], 4)

    def test_get_session_twice(self):
        session_1 = get_session()
        session_2 = get_session()
        self.assertIs(session_1, session_2)  # exact same instance

    def test_get_session_twice_but_reconfigure_in_between(self):
        """Reconfiguring the session clears the cache."""
        session_1 = get_session()
        configure_http_backend(backend_factory=self._factory)

        session_2 = get_session()
        self.assertIsNot(session_1, session_2)
        self.assertIsNone(session_1.headers.get("x-test-header"))
        self.assertEqual(session_2.headers["x-test-header"], 4)

    def test_get_session_multiple_threads(self):
        N = 3
        sessions = [None] * N

        def _get_session_in_thread(index: int) -> None:
            time.sleep(0.01)
            sessions[index] = get_session()

        # Get main thread session
        main_session = get_session()

        # Start 3 threads and get sessions in each of them
        threads = [threading.Thread(target=_get_session_in_thread, args=(index,)) for index in range(N)]
        for th in threads:
            th.start()
            print(th)
        for th in threads:
            th.join()

        # Check all sessions are different
        for i in range(N):
            self.assertIsNot(main_session, sessions[i])
            for j in range(N):
                if i != j:
                    self.assertIsNot(sessions[i], sessions[j])
