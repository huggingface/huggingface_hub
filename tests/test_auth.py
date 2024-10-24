import os
import tempfile
from unittest.mock import patch

import pytest

from huggingface_hub import constants
from huggingface_hub._login import _login, _set_active_token, auth_switch, logout
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
        _login(TOKEN, add_to_git_credential=False)

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
