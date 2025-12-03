import unittest
from unittest.mock import Mock, call, patch

from huggingface_hub.utils._pagination import paginate

from .testing_utils import handle_injection_in_test


class TestPagination(unittest.TestCase):
    @patch("huggingface_hub.utils._pagination.get_session")
    @patch("huggingface_hub.utils._pagination.http_backoff")
    @patch("huggingface_hub.utils._pagination.hf_raise_for_status")
    @handle_injection_in_test
    def test_mocked_paginate(
        self, mock_get_session: Mock, mock_http_backoff: Mock, mock_hf_raise_for_status: Mock
    ) -> None:
        mock_get = mock_get_session().get
        mock_params = Mock()
        mock_headers = Mock()

        # Simulate page 1
        mock_response_page_1 = Mock()
        mock_response_page_1.json.return_value = [1, 2, 3]
        mock_response_page_1.links = {"next": {"url": "url_p2"}}

        # Simulate page 2
        mock_response_page_2 = Mock()
        mock_response_page_2.json.return_value = [4, 5, 6]
        mock_response_page_2.links = {"next": {"url": "url_p3"}}

        # Simulate page 3
        mock_response_page_3 = Mock()
        mock_response_page_3.json.return_value = [7, 8]
        mock_response_page_3.links = {}

        # Mock response
        mock_get.side_effect = [
            mock_response_page_1,
        ]
        mock_http_backoff.side_effect = [
            mock_response_page_2,
            mock_response_page_3,
        ]

        results = paginate("url", params=mock_params, headers=mock_headers)

        # Requests are made only when generator is yielded
        assert mock_get.call_count == 0

        # Results after concatenating pages
        assert list(results) == [1, 2, 3, 4, 5, 6, 7, 8]

        # All pages requested: 3 requests, 3 raise for status
        # First request with `get_session.get` (we want at least 1 request to succeed correctly) and 2 with `http_backoff`
        assert mock_get.call_count == 1
        assert mock_http_backoff.call_count == 2
        assert mock_hf_raise_for_status.call_count == 3

        # Params not passed to next pages
        assert mock_get.call_args_list == [call("url", params=mock_params, headers=mock_headers)]
        assert mock_http_backoff.call_args_list == [
            call("GET", "url_p2", headers=mock_headers),
            call("GET", "url_p3", headers=mock_headers),
        ]

    def test_paginate_hf_api(self) -> None:
        # Real test: paginate over huggingface models
        # Use enumerate and stop after first page to avoid loading all repos
        for num, _ in enumerate(paginate("https://huggingface.co/api/models?limit=2", params={}, headers={})):
            if num == 5:
                break
        else:
            self.fail("Did not get more than 5 repos")
