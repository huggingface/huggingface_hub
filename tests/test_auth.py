import os
import tempfile
from unittest.mock import patch

import pytest

from huggingface_hub import constants
from huggingface_hub._login import (
    _device_code_login,
    _poll_for_token,
    _request_device_code,
    _set_active_token,
    _validate_and_save_token,
    auth_switch,
    logout,
)
from huggingface_hub.utils._auth import _get_token_by_name, _get_token_from_file, _save_token, get_stored_tokens

from .testing_constants import ENDPOINT_STAGING, OTHER_TOKEN, TOKEN


@pytest.fixture(autouse=True)
def use_tmp_file_paths():
    """
    Fixture to temporarily override HF_TOKEN_PATH, HF_STORED_TOKENS_PATH, and ENDPOINT.

    This fixture patches the constants in the huggingface_hub module to use the
    specified paths and the staging endpoint. It also ensures that the files are
    deleted after all tests in the module are completed.
    """
    with tempfile.TemporaryDirectory() as tmp_hf_home:
        hf_token_path = os.path.join(tmp_hf_home, "token")
        hf_stored_tokens_path = os.path.join(tmp_hf_home, "stored_tokens")
        with patch.multiple(
            constants,
            HF_TOKEN_PATH=hf_token_path,
            HF_STORED_TOKENS_PATH=hf_stored_tokens_path,
            ENDPOINT=ENDPOINT_STAGING,
        ):
            yield


class TestGetTokenByName:
    def test_get_existing_token(self):
        _save_token(TOKEN, "test_token")
        token = _get_token_by_name("test_token")
        assert token == TOKEN

    def test_get_non_existent_token(self):
        assert _get_token_by_name("non_existent") is None


class TestSaveToken:
    def test_save_new_token(self):
        _save_token(TOKEN, "new_token")

        stored_tokens = get_stored_tokens()
        assert "new_token" in stored_tokens
        assert stored_tokens["new_token"] == TOKEN

    def test_overwrite_existing_token(self):
        _save_token(TOKEN, "test_token")
        _save_token("new_token", "test_token")

        assert _get_token_by_name("test_token") == "new_token"


class TestSetActiveToken:
    def test_set_active_token_success(self):
        _save_token(TOKEN, "test_token")
        _set_active_token("test_token", add_to_git_credential=False)
        assert _get_token_from_file() == TOKEN

    def test_set_active_token_non_existent(self):
        non_existent_token = "non_existent"
        with pytest.raises(ValueError, match="Token non_existent not found in .*"):
            _set_active_token(non_existent_token, add_to_git_credential=False)


class TestLogin:
    @patch(
        "huggingface_hub.hf_api.whoami",
        return_value={
            "auth": {
                "accessToken": {
                    "displayName": "test_token",
                    "role": "write",
                    "createdAt": "2024-01-01T00:00:00.000Z",
                }
            }
        },
    )
    def test_login_success(self, mock_whoami):
        _validate_and_save_token(TOKEN, add_to_git_credential=False)

        assert _get_token_by_name("test_token") == TOKEN
        assert _get_token_from_file() == TOKEN


class TestLogout:
    def test_logout_deletes_files(self):
        _save_token(TOKEN, "test_token")
        _set_active_token("test_token", add_to_git_credential=False)

        assert os.path.exists(constants.HF_TOKEN_PATH)
        assert os.path.exists(constants.HF_STORED_TOKENS_PATH)

        logout()
        # Check that both files are deleted
        assert not os.path.exists(constants.HF_TOKEN_PATH)
        assert not os.path.exists(constants.HF_STORED_TOKENS_PATH)

    def test_logout_specific_token(self):
        # Create two tokens
        _save_token(TOKEN, "token_1")
        _save_token(OTHER_TOKEN, "token_2")

        logout("token_1")
        # Check that HF_STORED_TOKENS_PATH still exists
        assert os.path.exists(constants.HF_STORED_TOKENS_PATH)
        # Check that token_1 is removed
        stored_tokens = get_stored_tokens()
        assert "token_1" not in stored_tokens
        assert "token_2" in stored_tokens

    def test_logout_active_token(self):
        _save_token(TOKEN, "active_token")
        _set_active_token("active_token", add_to_git_credential=False)

        logout("active_token")

        # Check that both files are deleted
        assert not os.path.exists(constants.HF_TOKEN_PATH)
        stored_tokens = get_stored_tokens()
        assert "active_token" not in stored_tokens


class TestAuthSwitch:
    def test_auth_switch_existing_token(self):
        # Add two access tokens
        _save_token(TOKEN, "test_token_1")
        _save_token(OTHER_TOKEN, "test_token_2")
        # Set `test_token_1` as the active token
        _set_active_token("test_token_1", add_to_git_credential=False)

        # Switch to `test_token_2`
        auth_switch("test_token_2", add_to_git_credential=False)

        assert _get_token_from_file() == OTHER_TOKEN

    def test_auth_switch_nonexisting_token(self):
        with patch("huggingface_hub.utils._auth._get_token_by_name", return_value=None):
            with pytest.raises(ValueError):
                auth_switch("nonexistent_token")


class TestValidateAndSaveTokenName:
    """Test token name extraction and fallback in _validate_and_save_token."""

    @patch(
        "huggingface_hub.hf_api.whoami",
        return_value={
            "auth": {
                "accessToken": {
                    "displayName": "my-pat-token",
                    "role": "write",
                    "createdAt": "2024-01-01T00:00:00.000Z",
                }
            },
            "name": "testuser",
        },
    )
    def test_token_name_from_whoami(self, mock_whoami):
        """When whoami returns displayName, use it."""
        _validate_and_save_token(TOKEN, add_to_git_credential=False)
        assert _get_token_by_name("my-pat-token") == TOKEN

    @patch(
        "huggingface_hub.hf_api.whoami",
        return_value={"name": "testuser", "auth": {}},
    )
    def test_token_name_fallback_to_username(self, mock_whoami):
        """When whoami has no displayName (e.g. OAuth token), fallback to oauth-{username}."""
        _validate_and_save_token(TOKEN, add_to_git_credential=False)
        assert _get_token_by_name("oauth-testuser") == TOKEN

    @patch(
        "huggingface_hub.hf_api.whoami",
        return_value={"name": "testuser", "auth": {}},
    )
    def test_token_name_override(self, mock_whoami):
        """When token_name is explicitly provided, use it."""
        _validate_and_save_token(TOKEN, add_to_git_credential=False, token_name="custom-name")
        assert _get_token_by_name("custom-name") == TOKEN


class TestRequestDeviceCode:
    def test_success(self):
        mock_response = type(
            "Response",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "device_code": "device-xxx",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "https://huggingface.co/oauth/device",
                    "verification_uri_complete": "https://huggingface.co/oauth/device?code=ABCD-EFGH",
                    "interval": 5,
                    "expires_in": 900,
                },
            },
        )()
        with patch("huggingface_hub._login.get_session") as mock_session:
            mock_session.return_value.post.return_value = mock_response
            result = _request_device_code()
            assert result["user_code"] == "ABCD-EFGH"
            assert result["device_code"] == "device-xxx"

    def test_failure(self):
        mock_response = type(
            "Response",
            (),
            {
                "status_code": 400,
                "text": "bad request",
            },
        )()
        with patch("huggingface_hub._login.get_session") as mock_session:
            mock_session.return_value.post.return_value = mock_response
            with pytest.raises(RuntimeError, match="Failed to request device code"):
                _request_device_code()


class TestPollForToken:
    def test_success_after_pending(self):
        """Token is returned after one authorization_pending response."""
        pending_response = type(
            "Response",
            (),
            {
                "json": lambda self: {"error": "authorization_pending"},
            },
        )()
        success_response = type(
            "Response",
            (),
            {
                "json": lambda self: {"access_token": "hf_oauth_test123"},
            },
        )()
        with (
            patch("huggingface_hub._login.get_session") as mock_session,
            patch("huggingface_hub._login.time.sleep"),
        ):
            mock_session.return_value.post.side_effect = [pending_response, success_response]
            token = _poll_for_token("device-xxx", interval=1, expires_in=60)
            assert token == "hf_oauth_test123"

    def test_slow_down_increases_interval(self):
        """slow_down response increases the polling interval."""
        slow_down_response = type(
            "Response",
            (),
            {
                "json": lambda self: {"error": "slow_down"},
            },
        )()
        success_response = type(
            "Response",
            (),
            {
                "json": lambda self: {"access_token": "hf_oauth_test123"},
            },
        )()
        with (
            patch("huggingface_hub._login.get_session") as mock_session,
            patch("huggingface_hub._login.time.sleep") as mock_sleep,
        ):
            mock_session.return_value.post.side_effect = [slow_down_response, success_response]
            token = _poll_for_token("device-xxx", interval=5, expires_in=60)
            assert token == "hf_oauth_test123"
            # First sleep is interval=5, second sleep is interval=10 (after slow_down)
            assert mock_sleep.call_args_list[0].args[0] == 5
            assert mock_sleep.call_args_list[1].args[0] == 10

    def test_expired_token(self):
        expired_response = type(
            "Response",
            (),
            {
                "json": lambda self: {"error": "expired_token"},
            },
        )()
        with (
            patch("huggingface_hub._login.get_session") as mock_session,
            patch("huggingface_hub._login.time.sleep"),
        ):
            mock_session.return_value.post.return_value = expired_response
            with pytest.raises(RuntimeError, match="Device code expired"):
                _poll_for_token("device-xxx", interval=1, expires_in=60)

    def test_access_denied(self):
        denied_response = type(
            "Response",
            (),
            {
                "json": lambda self: {"error": "access_denied"},
            },
        )()
        with (
            patch("huggingface_hub._login.get_session") as mock_session,
            patch("huggingface_hub._login.time.sleep"),
        ):
            mock_session.return_value.post.return_value = denied_response
            with pytest.raises(RuntimeError, match="Authorization was denied"):
                _poll_for_token("device-xxx", interval=1, expires_in=60)


class TestDeviceCodeLogin:
    @patch(
        "huggingface_hub._login._poll_for_token",
        return_value="hf_oauth_test_token",
    )
    @patch(
        "huggingface_hub._login._request_device_code",
        return_value={
            "device_code": "device-xxx",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://huggingface.co/oauth/device",
            "verification_uri_complete": "https://huggingface.co/oauth/device?code=ABCD-EFGH",
            "interval": 5,
            "expires_in": 900,
        },
    )
    @patch(
        "huggingface_hub.hf_api.whoami",
        return_value={"name": "testuser", "auth": {}},
    )
    def test_device_code_login_success(self, mock_whoami, mock_request, mock_poll):
        _device_code_login(add_to_git_credential=False)
        # Token should be saved
        assert _get_token_by_name("oauth-testuser") == "hf_oauth_test_token"
        assert _get_token_from_file() == "hf_oauth_test_token"
