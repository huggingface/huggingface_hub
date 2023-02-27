import unittest
from unittest.mock import Mock, patch

from huggingface_hub.utils import get_hf_hub_version, get_python_version
from huggingface_hub.utils._headers import _deduplicate_user_agent, _http_user_agent, build_hf_headers

from .testing_utils import handle_injection, handle_injection_in_test


# Only for tests that are not related to user agent
DEFAULT_USER_AGENT = _http_user_agent()

FAKE_TOKEN = "123456789"
FAKE_TOKEN_ORG = "api_org_123456789"
FAKE_TOKEN_HEADER = {
    "authorization": f"Bearer {FAKE_TOKEN}",
    "user-agent": DEFAULT_USER_AGENT,
}
NO_AUTH_HEADER = {"user-agent": DEFAULT_USER_AGENT}


@patch("huggingface_hub.utils._headers.HfFolder")
@handle_injection
class TestAuthHeadersUtil(unittest.TestCase):
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
        self.assertEqual(build_hf_headers(use_auth_token=False), NO_AUTH_HEADER)

    def test_use_auth_token_none_no_cached_token(self, mock_HfFolder: Mock) -> None:
        mock_HfFolder().get_token.return_value = None
        self.assertEqual(build_hf_headers(), NO_AUTH_HEADER)

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

    def test_implicit_use_disabled(self, mock_HfFolder: Mock) -> None:
        with patch(  # not as decorator to avoid friction with @handle_injection
            "huggingface_hub.constants.HF_HUB_DISABLE_IMPLICIT_TOKEN", True
        ):
            mock_HfFolder().get_token.return_value = FAKE_TOKEN
            self.assertEqual(build_hf_headers(), NO_AUTH_HEADER)  # token is not sent

    def test_implicit_use_disabled_but_explicit_use(self, mock_HfFolder: Mock) -> None:
        with patch(  # not as decorator to avoid friction with @handle_injection
            "huggingface_hub.constants.HF_HUB_DISABLE_IMPLICIT_TOKEN", True
        ):
            mock_HfFolder().get_token.return_value = FAKE_TOKEN

            # This is not an implicit use so we still send it
            self.assertEqual(build_hf_headers(use_auth_token=True), FAKE_TOKEN_HEADER)


class TestUserAgentHeadersUtil(unittest.TestCase):
    def _get_user_agent(self, **kwargs) -> str:
        return build_hf_headers(**kwargs)["user-agent"]

    @patch("huggingface_hub.utils._headers.get_fastai_version")
    @patch("huggingface_hub.utils._headers.get_fastcore_version")
    @patch("huggingface_hub.utils._headers.get_tf_version")
    @patch("huggingface_hub.utils._headers.get_torch_version")
    @patch("huggingface_hub.utils._headers.is_fastai_available")
    @patch("huggingface_hub.utils._headers.is_fastcore_available")
    @patch("huggingface_hub.utils._headers.is_tf_available")
    @patch("huggingface_hub.utils._headers.is_torch_available")
    @handle_injection_in_test
    def test_default_user_agent(
        self,
        mock_get_fastai_version: Mock,
        mock_get_fastcore_version: Mock,
        mock_get_tf_version: Mock,
        mock_get_torch_version: Mock,
        mock_is_fastai_available: Mock,
        mock_is_fastcore_available: Mock,
        mock_is_tf_available: Mock,
        mock_is_torch_available: Mock,
    ) -> None:
        mock_get_fastai_version.return_value = "fastai_version"
        mock_get_fastcore_version.return_value = "fastcore_version"
        mock_get_tf_version.return_value = "tf_version"
        mock_get_torch_version.return_value = "torch_version"
        mock_is_fastai_available.return_value = True
        mock_is_fastcore_available.return_value = True
        mock_is_tf_available.return_value = True
        mock_is_torch_available.return_value = True
        self.assertEqual(
            self._get_user_agent(),
            (
                f"unknown/None; hf_hub/{get_hf_hub_version()};"
                f" python/{get_python_version()}; torch/torch_version;"
                " tensorflow/tf_version; fastai/fastai_version;"
                " fastcore/fastcore_version"
            ),
        )

    @patch("huggingface_hub.utils._headers.is_torch_available")
    @patch("huggingface_hub.utils._headers.is_tf_available")
    @handle_injection_in_test
    def test_user_agent_with_library_name_multiple_missing(
        self, mock_is_torch_available: Mock, mock_is_tf_available: Mock
    ) -> None:
        mock_is_torch_available.return_value = False
        mock_is_tf_available.return_value = False
        self.assertNotIn("torch", self._get_user_agent())
        self.assertNotIn("tensorflow", self._get_user_agent())

    def test_user_agent_with_library_name_and_version(self) -> None:
        self.assertTrue(
            self._get_user_agent(
                library_name="foo",
                library_version="bar",
            ).startswith("foo/bar;")
        )

    def test_user_agent_with_library_name_no_version(self) -> None:
        self.assertTrue(self._get_user_agent(library_name="foo").startswith("foo/None;"))

    def test_user_agent_with_custom_agent_string(self) -> None:
        self.assertTrue(self._get_user_agent(user_agent="this is a custom agent").endswith("this is a custom agent"))

    def test_user_agent_with_custom_agent_dict(self) -> None:
        self.assertTrue(self._get_user_agent(user_agent={"a": "b", "c": "d"}).endswith("a/b; c/d"))

    def test_user_agent_deduplicate(self) -> None:
        self.assertEqual(
            _deduplicate_user_agent(
                "python/3.7; python/3.8; hf_hub/0.12; transformers/None; hf_hub/0.12; python/3.7; diffusers/0.12.1"
            ),
            # 1. "python" is kept twice with different values
            # 2. "python/3.7" second occurrence is removed
            # 3. "hf_hub" second occurrence is removed
            # 4. order is preserved
            "python/3.7; python/3.8; hf_hub/0.12; transformers/None; diffusers/0.12.1",
        )
