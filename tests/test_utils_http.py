import time
import unittest
from typing import Generator
from unittest.mock import Mock, call, patch

from huggingface_hub.utils._http import http_backoff
from requests import ConnectTimeout, HTTPError


URL = "https://www.google.com"


@patch("huggingface_hub.utils._http.requests.request")
class TestHttpBackoff(unittest.TestCase):
    def test_backoff_no_errors(self, mock_request: Mock) -> None:
        """Test normal usage of `http_backoff`."""
        data_mock = Mock()
        response = http_backoff("GET", URL, data=data_mock)
        mock_request.assert_called_once_with(method="GET", url=URL, data=data_mock)
        self.assertIs(response, mock_request())

    def test_backoff_3_calls(self, mock_request: Mock) -> None:
        """Test `http_backoff` with 2 fails."""
        response_mock = Mock()
        mock_request.side_effect = (ValueError(), ValueError(), response_mock)
        response = http_backoff(  # retry on ValueError, instant retry
            "GET", URL, retry_on_exceptions=ValueError, base_wait_time=0.0
        )
        self.assertEqual(mock_request.call_count, 3)
        mock_request.assert_has_calls(
            calls=[
                call(method="GET", url=URL),
                call(method="GET", url=URL),
                call(method="GET", url=URL),
            ]
        )
        self.assertIs(response, response_mock)

    def test_backoff_on_exception_until_max(self, mock_request: Mock) -> None:
        """Test `http_backoff` until max limit is reached with exceptions."""
        mock_request.side_effect = ConnectTimeout()

        with self.assertRaises(ConnectTimeout):
            http_backoff("GET", URL, base_wait_time=0.0, max_retries=3)

        self.assertEqual(mock_request.call_count, 4)

    def test_backoff_on_status_code_until_max(self, mock_request: Mock) -> None:
        """Test `http_backoff` until max limit is reached with status codes."""
        mock_503 = Mock()
        mock_503.status_code = 503
        mock_504 = Mock()
        mock_504.status_code = 504
        mock_504.raise_for_status.side_effect = HTTPError()
        mock_request.side_effect = (mock_503, mock_504, mock_503, mock_504)

        with self.assertRaises(HTTPError):
            http_backoff(
                "GET",
                URL,
                base_wait_time=0.0,
                max_retries=3,
                retry_on_status_codes=(503, 504),
            )

        self.assertEqual(mock_request.call_count, 4)

    def test_backoff_on_exceptions_and_status_codes(self, mock_request: Mock) -> None:
        """Test `http_backoff` until max limit with status codes and exceptions."""
        mock_503 = Mock()
        mock_503.status_code = 503
        mock_request.side_effect = (mock_503, ConnectTimeout())

        with self.assertRaises(ConnectTimeout):
            http_backoff("GET", URL, base_wait_time=0.0, max_retries=1)

        self.assertEqual(mock_request.call_count, 2)

    def test_backoff_on_valid_status_code(self, mock_request: Mock) -> None:
        """Test `http_backoff` until max limit with a valid status code.

        Quite a corner case: the user wants to retry is status code is 200. Requests are
        retried but in the end, the HTTP 200 response is returned if the server returned
        only 200 responses.
        """
        mock_200 = Mock()
        mock_200.status_code = 200
        mock_request.side_effect = (mock_200, mock_200, mock_200, mock_200)

        response = http_backoff(
            "GET", URL, base_wait_time=0.0, max_retries=3, retry_on_status_codes=200
        )

        self.assertEqual(mock_request.call_count, 4)
        self.assertIs(response, mock_200)

    def test_backoff_sleep_time(self, mock_request: Mock) -> None:
        """Test `http_backoff` sleep time goes exponential until max limit.

        Since timing between 2 requests is sleep duration + some other stuff, this test
        can be unstable. However, sleep durations between 4ms and 20ms should be enough
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
                sleep_times.append(round(t1 - t0, 3))
                t0 = t1

        mock_request.side_effect = _side_effect_timer()

        with self.assertRaises(ConnectTimeout):
            http_backoff(
                "GET", URL, base_wait_time=0.004, max_wait_time=0.02, max_retries=5
            )

        self.assertEqual(mock_request.call_count, 6)

        # Assert sleep times are exponential until plateau
        expected_sleep_times = [0.004, 0.008, 0.016, 0.02, 0.02]
        self.assertListEqual(sleep_times, expected_sleep_times)
