import unittest
from unittest.mock import Mock, call, patch

from huggingface_hub.utils._pagination import paginate

from .testing_utils import handle_injection_in_test


class TestPagination(unittest.TestCase):
    @patch("huggingface_hub.utils._pagination.get_session")
    @patch("huggingface_hub.utils._pagination.hf_raise_for_status")
    @handle_injection_in_test
    def test_mocked_paginate(self, mock_get_session: Mock, mock_hf_raise_for_status: Mock) -> None:
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
            mock_response_page_2,
            mock_response_page_3,
        ]

        results = paginate("url", params=mock_params, headers=mock_headers)

        # Requests are made only when generator is yielded
        self.assertEqual(mock_get.call_count, 0)

        # Results after concatenating pages
        self.assertListEqual(list(results), [1, 2, 3, 4, 5, 6, 7, 8])

        # All pages requested: 3 requests, 3 raise for status
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(mock_hf_raise_for_status.call_count, 3)

        # Params not passed to next pages
        self.assertListEqual(
            mock_get.call_args_list,
            [
                call("url", params=mock_params, headers=mock_headers),
                call("url_p2", headers=mock_headers),
                call("url_p3", headers=mock_headers),
            ],
        )

    def test_paginate_github_api(self) -> None:
        # Real test: paginate over huggingface repos on Github
        # Use enumerate and stop after first page to avoid loading all repos
        for num, _ in enumerate(
            paginate("https://api.github.com/orgs/huggingface/repos?limit=4", params={}, headers={})
        ):
            if num == 6:
                break
        else:
            self.fail("Did not get more than 6 repos")
