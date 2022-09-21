import unittest
from unittest.mock import Mock, patch

from huggingface_hub.utils._headers import build_hf_headers

from .testing_utils import handle_injection


FAKE_TOKEN = "123456789"
FAKE_TOKEN_ORG = "api_org_123456789"
FAKE_TOKEN_HEADER = {"authorization": f"Bearer {FAKE_TOKEN}"}


@patch("huggingface_hub.utils._headers.HfFolder")
@handle_injection
class TestHeadersUtilsBuildHeadersNew(unittest.TestCase):
    def test_use_auth_token_str(self) -> None:
        self.assertEqual(build_hf_headers(use_auth_token=FAKE_TOKEN), FAKE_TOKEN_HEADER)

    def test_use_auth_token_true_no_cached_token(self, mock_HfFolder: Mock) -> None:
        mock_HfFolder().get_token.return_value = None
        with self.assertRaises(EnvironmentError):
            build_hf_headers(use_auth_token=True)

    def test_use_auth_token_true_has_cached_token(self, mock_HfFolder: Mock) -> None:
        mock_HfFolder().get_token.return_value = FAKE_TOKEN
        self.assertEqual(build_hf_headers(use_auth_token=True), FAKE_TOKEN_HEADER)

    def test_use_auth_token_false(self, mock_HfFolder: Mock) -> None:
        mock_HfFolder().get_token.return_value = FAKE_TOKEN
        self.assertEqual(build_hf_headers(use_auth_token=False), {})

    def test_use_auth_token_none_no_cached_token(self, mock_HfFolder: Mock) -> None:
        mock_HfFolder().get_token.return_value = None
        self.assertEqual(build_hf_headers(), {})

    def test_use_auth_token_none_has_cached_token(self, mock_HfFolder: Mock) -> None:
        mock_HfFolder().get_token.return_value = FAKE_TOKEN
        self.assertEqual(build_hf_headers(), FAKE_TOKEN_HEADER)

    def test_write_action_org_token(self) -> None:
        with self.assertRaises(ValueError):
            build_hf_headers(use_auth_token=FAKE_TOKEN_ORG, is_write_action=True)

    def test_write_action_none_token(self, mock_HfFolder: Mock) -> None:
        mock_HfFolder().get_token.return_value = None
        with self.assertRaises(ValueError):
            build_hf_headers(is_write_action=True)

    def test_write_action_use_auth_token_false(self) -> None:
        with self.assertRaises(ValueError):
            build_hf_headers(use_auth_token=False, is_write_action=True)
