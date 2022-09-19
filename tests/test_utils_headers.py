import unittest
from unittest.mock import Mock, patch

from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub.utils._headers import (
    _get_token_to_send,
    _is_private,
    _is_valid_token,
    _validate_token_to_send,
    build_hf_headers,
)

from .testing_utils import handle_injection_in_test


FAKE_URL = "FAKE_URL"
FAKE_ENDPOINT = "hf.co/fake_endpoint"
FAKE_TOKEN = "123456789"
FAKE_TOKEN_2 = "987654321"


class TestHeadersUtilsBuildHeaders(unittest.TestCase):
    @patch("huggingface_hub.utils._headers._get_token_to_send")
    @patch("huggingface_hub.utils._headers._validate_token_to_send")
    @handle_injection_in_test
    def test_build_hf_headers(
        self, mock__get_token_to_send: Mock, mock__validate_token_to_send: Mock
    ) -> None:
        # Mock all args (sub-helpers are unit-tested afterwards)
        mock__get_token_to_send.return_value = FAKE_TOKEN
        mock_token = Mock()
        mock_use_auth_token = Mock()
        mock_is_write_action = Mock()
        mock_url = Mock()
        mock_endpoint = Mock()
        mock_repo_id = Mock()
        mock_repo_type = Mock()

        # Make call
        headers = build_hf_headers(
            token=mock_token,
            use_auth_token=mock_use_auth_token,
            is_write_action=mock_is_write_action,
            url=mock_url,
            endpoint=mock_endpoint,
            repo_id=mock_repo_id,
            repo_type=mock_repo_type,
        )

        # `_get_token_to_send` called
        mock__get_token_to_send.assert_called_once_with(
            token=mock_token,
            use_auth_token=mock_use_auth_token,
            is_write_action=mock_is_write_action,
            url=mock_url,
            endpoint=mock_endpoint,
            repo_id=mock_repo_id,
            repo_type=mock_repo_type,
        )

        # `_validate_token_to_send` called on result
        mock__validate_token_to_send.assert_called_once_with(
            FAKE_TOKEN, endpoint=mock_endpoint, is_write_action=mock_is_write_action
        )

        # Returned the header with auth
        self.assertEqual(headers, {"authorization": f"Bearer {FAKE_TOKEN}"})


class TestHeadersUtilsGetTokenToSend(unittest.TestCase):
    def test_get_token_to_send_on_token(self) -> None:
        self.assertEqual(
            _get_token_to_send(
                token=FAKE_TOKEN,
                use_auth_token=None,
                is_write_action=False,
                repo_id=None,
                repo_type=None,
                endpoint=None,
                url=None,
            ),
            FAKE_TOKEN,
        )

    def test_get_token_to_send_on_use_auth_token_str(self) -> None:
        self.assertEqual(
            _get_token_to_send(
                token=None,
                use_auth_token=FAKE_TOKEN,
                is_write_action=False,
                repo_id=None,
                repo_type=None,
                endpoint=None,
                url=None,
            ),
            FAKE_TOKEN,
        )

    def test_get_token_to_send_on_both_tokens_args(self) -> None:
        self.assertEqual(
            _get_token_to_send(
                token=FAKE_TOKEN,  # has priority
                use_auth_token=FAKE_TOKEN_2,
                is_write_action=False,
                repo_id=None,
                repo_type=None,
                endpoint=None,
                url=None,
            ),
            FAKE_TOKEN,
        )

    def test_get_token_to_send_on_use_auth_token_false(self) -> None:
        self.assertIsNone(
            _get_token_to_send(
                token=None,
                use_auth_token=False,  # explicit
                is_write_action=False,
                repo_id=None,
                repo_type=None,
                endpoint=None,
                url=None,
            )
        )

    @patch("huggingface_hub.utils._headers.HfFolder")
    def test_get_token_to_send_on_use_auth_token_true_and_cached_token(
        self, mock_HfFolder: Mock
    ) -> None:
        mock_HfFolder().get_token.return_value = FAKE_TOKEN
        self.assertEqual(
            _get_token_to_send(
                token=None,
                use_auth_token=True,  # explicit
                is_write_action=False,
                repo_id=None,
                repo_type=None,
                endpoint=None,
                url=None,
            ),
            FAKE_TOKEN,
        )

    @patch("huggingface_hub.utils._headers.HfFolder")
    def test_get_token_to_send_on_use_auth_token_true_and_no_cached_token(
        self, mock_HfFolder: Mock
    ) -> None:
        mock_HfFolder().get_token.return_value = None
        with self.assertRaises(EnvironmentError):
            _get_token_to_send(
                token=None,
                use_auth_token=True,  # explicit but token not found in cache
                is_write_action=False,
                repo_id=None,
                repo_type=None,
                endpoint=None,
                url=None,
            )

    @patch("huggingface_hub.utils._headers._is_private")
    def test_get_token_to_send_on_use_auth_token_none_and_not_private(
        self, mock__is_private: Mock
    ) -> None:
        mock__is_private.return_value = False
        # Not found -> no need to send the token
        self.assertIsNone(
            _get_token_to_send(
                token=None,
                use_auth_token=None,
                is_write_action=False,
                repo_id=None,
                repo_type=None,
                endpoint=None,
                url=None,
            )
        )

    @patch("huggingface_hub.utils._headers._is_private")
    @patch("huggingface_hub.utils._headers.HfFolder")
    @handle_injection_in_test
    def test_get_token_to_send_on_use_auth_token_none_and_private_and_no_token(
        self, mock__is_private: Mock, mock_HfFolder: Mock
    ) -> None:
        mock_HfFolder().get_token.return_value = None
        mock__is_private.return_value = True
        with self.assertRaises(EnvironmentError):
            # Private repo and token not found
            _get_token_to_send(
                token=None,
                use_auth_token=True,
                is_write_action=False,
                repo_id=None,
                repo_type=None,
                endpoint=None,
                url=None,
            )

    @patch("huggingface_hub.utils._headers._is_private")
    @patch("huggingface_hub.utils._headers.HfFolder")
    @handle_injection_in_test
    def test_get_token_to_send_on_use_auth_token_none_and_private_and_token_found(
        self, mock__is_private: Mock, mock_HfFolder: Mock
    ) -> None:
        mock_HfFolder().get_token.return_value = FAKE_TOKEN
        mock__is_private.return_value = True

        # Private repo and token found
        self.assertEqual(
            _get_token_to_send(
                token=None,
                use_auth_token=True,
                is_write_action=False,
                repo_id=None,
                repo_type=None,
                endpoint=None,
                url=None,
            ),
            FAKE_TOKEN,
        )

    @patch("huggingface_hub.utils._headers._is_private")
    @patch("huggingface_hub.utils._headers.HfFolder")
    @handle_injection_in_test
    def test_get_token_to_send_on_use_auth_token_none_and_public_and_write_action(
        self, mock__is_private: Mock, mock_HfFolder: Mock
    ) -> None:
        mock_HfFolder().get_token.return_value = FAKE_TOKEN
        mock__is_private.return_value = False

        # Public repo, write access required and token found
        self.assertEqual(
            _get_token_to_send(
                token=None,
                use_auth_token=True,
                is_write_action=True,  # Token required !
                repo_id=None,
                repo_type=None,
                endpoint=None,
                url=None,
            ),
            FAKE_TOKEN,
        )


class TestHeadersUtilsValidateTokenToSend(unittest.TestCase):
    def test_validate_token_to_send_on_org_token_and_write_access_required(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            _validate_token_to_send(
                token="api_org_123456789", endpoint=None, is_write_action=True
            )

    def test_validate_token_to_send_on_user_token_and_write_access_required(
        self,
    ) -> None:
        _validate_token_to_send(
            token="hf_123456789", endpoint=None, is_write_action=True
        )

    @patch("huggingface_hub.utils._headers._is_valid_token")
    def test_validate_token_to_send_check_token_validity(
        self, mock__is_valid_token: Mock
    ) -> None:
        mock__is_valid_token.return_value = True
        _validate_token_to_send(
            token=FAKE_TOKEN, endpoint=FAKE_ENDPOINT, is_write_action=False
        )
        mock__is_valid_token.assert_called_once_with(FAKE_ENDPOINT, FAKE_TOKEN)

    @patch("huggingface_hub.utils._headers._is_valid_token")
    def test_validate_token_to_send_check_token_validity_not_valid(
        self, mock__is_valid_token: Mock
    ) -> None:
        mock__is_valid_token.return_value = False
        with self.assertRaises(ValueError):
            _validate_token_to_send(
                token=FAKE_TOKEN, endpoint=FAKE_ENDPOINT, is_write_action=False
            )
        mock__is_valid_token.assert_called_once_with(FAKE_ENDPOINT, FAKE_TOKEN)


class TestHeadersUtilsIsPrivate(unittest.TestCase):
    def test_is_private_all_none(self) -> None:
        with self.assertRaises(ValueError):
            _is_private(url=None, endpoint=None, repo_id=None, repo_type=None)

    @patch("huggingface_hub.utils._headers.requests.head")
    def test_is_private_url_public_repo(self, mock_head: Mock) -> None:
        mock_head.status_code = 200
        self.assertFalse(
            _is_private(url=FAKE_URL, endpoint=None, repo_id=None, repo_type=None)
        )
        mock_head.assert_called_once_with(FAKE_URL)

    @patch("huggingface_hub.utils._headers.requests.head")
    @patch("huggingface_hub.utils._headers.hf_raise_for_status")
    @handle_injection_in_test
    def test_is_private_url_private_repo(self, mock_hf_raise_for_status: Mock) -> None:
        mock_hf_raise_for_status.side_effect = RepositoryNotFoundError(
            "repo not found", response=None
        )
        self.assertTrue(
            _is_private(url=FAKE_URL, endpoint=None, repo_id=None, repo_type=None)
        )

    @patch("huggingface_hub.utils._headers.requests.head")
    @patch("huggingface_hub.utils._headers.hf_raise_for_status")
    @handle_injection_in_test
    def test_is_private_on_url_any_error(self, mock_hf_raise_for_status: Mock) -> None:
        mock_hf_raise_for_status.side_effect = Exception()
        self.assertFalse(
            _is_private(url=FAKE_URL, endpoint=None, repo_id=None, repo_type=None)
        )

    def test_is_private_generic_api_call(self) -> None:
        self.assertFalse(
            _is_private(url=None, endpoint=FAKE_ENDPOINT, repo_id=None, repo_type=None)
        )

    @patch("huggingface_hub.utils._headers.requests.head")
    @handle_injection_in_test
    def test_is_private_on_public_model(self, mock_head: Mock) -> None:
        mock_head.status_code = 200
        self.assertFalse(
            _is_private(
                url=None, endpoint=FAKE_ENDPOINT, repo_id="model_1", repo_type=None
            )
        )
        mock_head.assert_called_once_with(f"{FAKE_ENDPOINT}/api/models/model_1")

    @patch("huggingface_hub.utils._headers.requests.head")
    @handle_injection_in_test
    def test_is_private_on_public_dataset(self, mock_head: Mock) -> None:
        mock_head.status_code = 200
        self.assertFalse(
            _is_private(
                url=None,
                endpoint=FAKE_ENDPOINT,
                repo_id="dataset_1",
                repo_type="dataset",
            )
        )
        mock_head.assert_called_once_with(f"{FAKE_ENDPOINT}/api/datasets/dataset_1")

    @patch("huggingface_hub.utils._headers.requests.head")
    @patch("huggingface_hub.utils._headers.hf_raise_for_status")
    @handle_injection_in_test
    def test_is_private_on_private_model(self, mock_hf_raise_for_status: Mock) -> None:
        mock_hf_raise_for_status.side_effect = RepositoryNotFoundError(
            "repo not found", response=None
        )
        self.assertTrue(
            _is_private(
                url=None, endpoint=FAKE_ENDPOINT, repo_id="model_1", repo_type=None
            )
        )

    @patch("huggingface_hub.utils._headers.requests.head")
    @patch("huggingface_hub.utils._headers.hf_raise_for_status")
    @handle_injection_in_test
    def test_is_private_on_repo_any_error(self, mock_hf_raise_for_status: Mock) -> None:
        mock_hf_raise_for_status.side_effect = Exception("not found")
        self.assertFalse(
            _is_private(
                url=None, endpoint=FAKE_ENDPOINT, repo_id="model_1", repo_type=None
            )
        )


class TestHeadersUtilsIsValid(unittest.TestCase):
    @patch("huggingface_hub.utils._headers.requests.get")
    def test_token_is_valid_on_valid_token(self, mock_get: Mock) -> None:
        mock_get.status_code = 200
        self.assertTrue(_is_valid_token(endpoint=FAKE_ENDPOINT, token=FAKE_TOKEN))

        mock_get.assert_called_once_with(
            f"{FAKE_ENDPOINT}/api/whoami-v2",
            headers={"authorization": f"Bearer {FAKE_TOKEN}"},
        )

    @patch("huggingface_hub.utils._headers.requests.get")
    @patch("huggingface_hub.utils._headers.hf_raise_for_status")
    @handle_injection_in_test
    def test_token_is_valid_on_not_valid_token(
        self, mock_hf_raise_for_status: Mock
    ) -> None:
        mock_hf_raise_for_status.side_effect = Exception
        self.assertFalse(_is_valid_token(endpoint=FAKE_ENDPOINT, token=FAKE_TOKEN))
