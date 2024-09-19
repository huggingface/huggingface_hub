import os
from unittest.mock import patch

import pytest

from huggingface_hub import constants
from huggingface_hub._login import _login, _set_active_profile, auth_switch, logout
from huggingface_hub.utils._auth import (
    _get_token_from_file,
    _get_token_from_profile,
    _save_token_to_profile,
    get_profiles,
)

from .testing_constants import OTHER_TOKEN, TOKEN


class TestGetTokenFromProfile:
    def test_get_existing_token(self):
        _save_token_to_profile(TOKEN, "test_profile")
        token = _get_token_from_profile("test_profile")
        assert token == TOKEN

    def test_get_non_existent_token(self):
        assert _get_token_from_profile("non_existent") is None


class TestSaveTokenToProfile:
    def test_save_token_new_profile(self):
        _save_token_to_profile(TOKEN, "new_profile")

        profiles = get_profiles()
        assert "new_profile" in profiles
        assert profiles["new_profile"] == TOKEN

    def test_overwrite_existing_profile(self):
        _save_token_to_profile(TOKEN, "test_profile")
        _save_token_to_profile("new_token", "test_profile")

        assert _get_token_from_profile("test_profile") == "new_token"


class TestSetActiveProfile:
    def test_set_active_profile_success(self):
        _save_token_to_profile(TOKEN, "test_profile")
        _set_active_profile("test_profile", add_to_git_credential=False)
        assert _get_token_from_file() == TOKEN

    def test_set_active_profile_non_existent(self):
        non_existent_profile = "non_existent"
        with pytest.raises(
            ValueError, match=rf"Profile {non_existent_profile} not found in {constants.HF_PROFILES_PATH}"
        ):
            _set_active_profile(non_existent_profile, add_to_git_credential=False)


class TestLogin:
    @patch("huggingface_hub.hf_api.get_token_permission", return_value="write")
    def test_login_success(self, mock_get_token_permission):
        _login(TOKEN, add_to_git_credential=False, profile_name="test_profile")

        assert _get_token_from_profile("test_profile") == TOKEN
        assert _get_token_from_file() == TOKEN

    @patch("huggingface_hub.hf_api.get_token_permission")
    def test_login_errors(self, mock_get_token_permission):
        mock_get_token_permission.return_value = None
        with pytest.raises(ValueError, match="Invalid token passed!"):
            _login("invalid_token", add_to_git_credential=False, write_permission=False)

        mock_get_token_permission.return_value = "read"
        with pytest.raises(ValueError, match=r"Token is valid but is 'read-only' and a 'write' token is required.*"):
            _login(TOKEN, add_to_git_credential=False, write_permission=True)


class TestLogout:
    def test_logout_deletes_files(self):
        _save_token_to_profile(TOKEN, "test_profile")
        _set_active_profile("test_profile", add_to_git_credential=False)

        assert os.path.exists(constants.HF_TOKEN_PATH)
        assert os.path.exists(constants.HF_PROFILES_PATH)

        logout()
        # Check that both files are deleted
        assert not os.path.exists(constants.HF_TOKEN_PATH)
        assert not os.path.exists(constants.HF_PROFILES_PATH)

    def test_logout_specific_profile(self):
        # Create two profiles
        _save_token_to_profile(TOKEN, "profile_1")
        _save_token_to_profile(OTHER_TOKEN, "profile_2")

        assert os.path.exists(constants.HF_PROFILES_PATH)

        logout("profile_1")

        # Check that profile_1 is removed
        profiles = get_profiles()
        assert "profile_1" not in profiles
        assert "profile_2" in profiles

    def test_logout_active_profile(self):
        _save_token_to_profile(TOKEN, "active_profile")
        _set_active_profile("active_profile", add_to_git_credential=False)

        logout("active_profile")

        # Check that both files are deleted
        assert not os.path.exists(constants.HF_TOKEN_PATH)
        profiles = get_profiles()
        assert "active_profile" not in profiles


class TestAuthSwitch:
    def test_auth_switch_existing_profile(self):
        # Add two profiles
        _save_token_to_profile(TOKEN, "test_profile_1")
        _save_token_to_profile(OTHER_TOKEN, "test_profile_2")
        # Set `test_profile_1` as the active profile
        _set_active_profile("test_profile_1", add_to_git_credential=False)

        # Switch to `test_profile_2`
        auth_switch("test_profile_2", add_to_git_credential=False)

        assert _get_token_from_file() == OTHER_TOKEN

    def test_auth_switch_nonexistent_profile(self):
        with patch("huggingface_hub.utils._auth._get_token_from_profile", return_value=None):
            with pytest.raises(ValueError):
                auth_switch("nonexistent_profile")
