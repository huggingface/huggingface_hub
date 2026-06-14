import os
import re
import tempfile
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import httpx
import pytest

from huggingface_hub import constants
from huggingface_hub._login import (
    _device_code_login,
    _format_expiration,
    _set_active_token,
    _validate_and_save_token,
    auth_switch,
    logout,
)
from huggingface_hub._oauth_device import poll_device_token, refresh_access_token, request_device_code
from huggingface_hub.errors import DeviceCodeError
from huggingface_hub.utils._auth import (
    _get_token_by_name,
    _get_token_from_file,
    _read_stored_tokens_full,
    _save_token,
    get_stored_tokens,
    get_token,
)

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


def _mock_response(payload: dict, status_code: int = 200, text: str = "") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    response.text = text
    return response


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

    def test_save_token_with_metadata(self):
        _save_token(TOKEN, "oauth_token", refresh_token="rt_secret", expires_at=1752192000)

        # Public reader keeps its `dict[str, str]` shape
        assert get_stored_tokens() == {"oauth_token": TOKEN}
        # Full reader exposes the refresh metadata
        fields = _read_stored_tokens_full()["oauth_token"]
        assert fields["hf_token"] == TOKEN
        assert fields["refresh_token"] == "rt_secret"
        assert fields["expires_at"] == "1752192000"

    def test_overwrite_drops_stale_metadata(self):
        _save_token(TOKEN, "test_token", refresh_token="rt_secret", expires_at=1752192000)
        _save_token(OTHER_TOKEN, "test_token")

        fields = _read_stored_tokens_full()["test_token"]
        assert fields == {"hf_token": OTHER_TOKEN}

    def test_round_trips_percent_in_values(self):
        """Token values are opaque server strings: configparser interpolation must not interpret `%`."""
        _save_token(TOKEN, "oauth_token", refresh_token="rt_with_%_inside", expires_at=1752192000)
        assert _read_stored_tokens_full()["oauth_token"]["refresh_token"] == "rt_with_%_inside"


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
            "name": "testuser",
            "auth": {
                "accessToken": {
                    "displayName": "test_token",
                    "role": "write",
                    "createdAt": "2024-01-01T00:00:00.000Z",
                }
            },
        },
    )
    def test_login_success(self, mock_whoami):
        token_name, username = _validate_and_save_token(TOKEN, add_to_git_credential=False)

        assert token_name == "test_token"
        assert username == "testuser"
        assert _get_token_by_name("test_token") == TOKEN
        assert _get_token_from_file() == TOKEN

    @patch(
        "huggingface_hub.hf_api.whoami",
        return_value={"name": "testuser", "auth": {}},
    )
    def test_token_name_fallback_for_oauth_tokens(self, mock_whoami):
        """OAuth tokens have no displayName in the whoami response: fallback to oauth-{username}."""
        _validate_and_save_token(TOKEN, add_to_git_credential=False)
        assert _get_token_by_name("oauth-testuser") == TOKEN

    @patch(
        "huggingface_hub.hf_api.whoami",
        return_value={"name": "testuser", "auth": {}},
    )
    def test_refresh_metadata_persisted(self, mock_whoami):
        _validate_and_save_token(TOKEN, add_to_git_credential=False, refresh_token="rt_secret", expires_at=1752192000)
        fields = _read_stored_tokens_full()["oauth-testuser"]
        assert fields["refresh_token"] == "rt_secret"
        assert fields["expires_at"] == "1752192000"


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

    def test_logout_preserves_other_tokens_metadata(self):
        _save_token(TOKEN, "token_1")
        _save_token(OTHER_TOKEN, "token_2", refresh_token="rt_secret", expires_at=1752192000)

        logout("token_1")
        fields = _read_stored_tokens_full()["token_2"]
        assert fields["refresh_token"] == "rt_secret"
        assert fields["expires_at"] == "1752192000"

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


class TestRequestDeviceCode:
    def test_success_normalizes_response(self):
        """Server values are kept; missing optional fields are filled with defaults."""
        response = _mock_response(
            {
                "device_code": "device-xxx",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://huggingface.co/oauth/device",
            }
        )
        with patch("huggingface_hub._oauth_device.get_session") as mock_session:
            mock_session.return_value.post.return_value = response
            result = request_device_code()
        assert result["user_code"] == "ABCD-EFGH"
        assert result["device_code"] == "device-xxx"
        assert result["verification_uri_complete"] == "https://huggingface.co/oauth/device"
        assert result["interval"] == 5
        assert result["expires_in"] == 900

    def test_failure(self):
        response = httpx.Response(
            400, text="bad request", request=httpx.Request("POST", "https://hub.test/oauth/device")
        )
        with patch("huggingface_hub._oauth_device.get_session") as mock_session:
            mock_session.return_value.post.return_value = response
            with pytest.raises(DeviceCodeError, match="Failed to request device code"):
                request_device_code()


def _device_info(**overrides) -> dict:
    return {
        "device_code": "device-xxx",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://huggingface.co/oauth/device",
        "verification_uri_complete": "https://huggingface.co/oauth/device",
        "interval": 1,
        "expires_in": 60,
        **overrides,
    }


class TestPollDeviceToken:
    def test_success_after_pending(self):
        """The full token response is returned after an authorization_pending response."""
        on_pending = MagicMock()
        with (
            patch("huggingface_hub._oauth_device.get_session") as mock_session,
            patch("huggingface_hub._oauth_device.time.sleep"),
        ):
            mock_session.return_value.post.side_effect = [
                _mock_response({"error": "authorization_pending"}),
                _mock_response({"access_token": "hf_oauth_123", "refresh_token": "rt_123", "expires_in": 2592000}),
            ]
            response = poll_device_token(_device_info(), on_pending=on_pending)
        assert response == {"access_token": "hf_oauth_123", "refresh_token": "rt_123", "expires_in": 2592000}
        on_pending.assert_called_once()

    def test_slow_down_increases_interval(self):
        with (
            patch("huggingface_hub._oauth_device.get_session") as mock_session,
            patch("huggingface_hub._oauth_device.time.sleep") as mock_sleep,
        ):
            mock_session.return_value.post.side_effect = [
                _mock_response({"error": "slow_down"}),
                _mock_response({"access_token": "hf_oauth_123"}),
            ]
            poll_device_token(_device_info(interval=5))
        # The first poll happens immediately; after slow_down the interval is bumped to 10.
        assert [call.args[0] for call in mock_sleep.call_args_list] == [10]

    def test_transient_failures_keep_polling(self):
        """Network blips, 5xx, non-JSON pages and non-OAuth JSON errors mid-poll must not abort the login."""
        non_json = MagicMock()
        non_json.status_code = 429
        non_json.json.side_effect = ValueError("not json")
        with (
            patch("huggingface_hub._oauth_device.get_session") as mock_session,
            patch("huggingface_hub._oauth_device.time.sleep"),
        ):
            mock_session.return_value.post.side_effect = [
                httpx.ConnectError("network blip"),
                _mock_response({}, status_code=502),
                non_json,
                _mock_response({"message": "forbidden"}, status_code=403),  # JSON without an `error` field
                _mock_response({"access_token": "hf_oauth_123"}),
            ]
            response = poll_device_token(_device_info())
        assert response == {"access_token": "hf_oauth_123"}

    @pytest.mark.parametrize(
        ("error", "match"),
        [("expired_token", "Device code expired"), ("access_denied", "Authorization was denied")],
    )
    def test_oauth_errors(self, error, match):
        with (
            patch("huggingface_hub._oauth_device.get_session") as mock_session,
            patch("huggingface_hub._oauth_device.time.sleep"),
        ):
            mock_session.return_value.post.return_value = _mock_response({"error": error})
            with pytest.raises(DeviceCodeError, match=match) as exc_info:
                poll_device_token(_device_info())
        assert exc_info.value.error_code == error


class TestRefreshAccessToken:
    def test_success(self):
        payload = {"access_token": "hf_oauth_new", "refresh_token": "rt_new", "expires_in": 2592000}
        with patch("huggingface_hub._oauth_device.get_session") as mock_session:
            mock_session.return_value.post.return_value = _mock_response(payload)
            assert refresh_access_token("rt_old") == payload

    def test_invalid_grant(self):
        response = _mock_response({"error": "invalid_grant", "error_description": "revoked"}, status_code=400)
        with patch("huggingface_hub._oauth_device.get_session") as mock_session:
            mock_session.return_value.post.return_value = response
            with pytest.raises(DeviceCodeError, match="Failed to refresh") as exc_info:
                refresh_access_token("rt_old")
        assert exc_info.value.error_code == "invalid_grant"


class TestGetTokenAutoRefresh:
    def _login_with_oauth_token(self, expires_in: int) -> None:
        _save_token(TOKEN, "oauth-user", refresh_token="rt_old", expires_at=int(time.time()) + expires_in)
        _set_active_token("oauth-user", add_to_git_credential=False)

    def test_refreshes_near_expiry(self):
        self._login_with_oauth_token(expires_in=3600)  # < 1 day left
        refreshed = {"access_token": "hf_oauth_new", "refresh_token": "rt_new", "expires_in": 2592000}
        with patch("huggingface_hub.utils._auth.refresh_access_token", return_value=refreshed) as mock_refresh:
            assert get_token() == "hf_oauth_new"
            assert get_token() == "hf_oauth_new"  # second call served from cache
        mock_refresh.assert_called_once_with("rt_old")
        # New token + rotated refresh token are persisted, active token file updated
        fields = _read_stored_tokens_full()["oauth-user"]
        assert fields["hf_token"] == "hf_oauth_new"
        assert fields["refresh_token"] == "rt_new"
        assert _get_token_from_file() == "hf_oauth_new"

    def test_no_refresh_when_far_from_expiry(self):
        self._login_with_oauth_token(expires_in=30 * 24 * 3600)
        with patch("huggingface_hub.utils._auth.refresh_access_token") as mock_refresh:
            assert get_token() == TOKEN
        mock_refresh.assert_not_called()

    def test_no_refresh_without_metadata(self):
        _save_token(TOKEN, "classic-token")
        _set_active_token("classic-token", add_to_git_credential=False)
        with patch("huggingface_hub.utils._auth.refresh_access_token") as mock_refresh:
            assert get_token() == TOKEN
        mock_refresh.assert_not_called()

    def test_non_rotated_refresh_token_is_kept(self):
        self._login_with_oauth_token(expires_in=3600)
        refreshed = {"access_token": "hf_oauth_new", "expires_in": 2592000}  # no refresh_token in response
        with patch("huggingface_hub.utils._auth.refresh_access_token", return_value=refreshed):
            assert get_token() == "hf_oauth_new"
        assert _read_stored_tokens_full()["oauth-user"]["refresh_token"] == "rt_old"

    def test_transient_failure_returns_stale_token(self):
        self._login_with_oauth_token(expires_in=3600)
        with patch(
            "huggingface_hub.utils._auth.refresh_access_token", side_effect=RuntimeError("offline")
        ) as mock_refresh:
            assert get_token() == TOKEN
            assert get_token() == TOKEN  # failure is cached: no retry storm
        mock_refresh.assert_called_once()

    def test_invalid_grant_returns_stale_token_and_stops_retrying(self):
        self._login_with_oauth_token(expires_in=3600)
        error = DeviceCodeError("revoked", error_code="invalid_grant")
        with patch("huggingface_hub.utils._auth.refresh_access_token", side_effect=error) as mock_refresh:
            assert get_token() == TOKEN
            assert get_token() == TOKEN
        mock_refresh.assert_called_once()

    def test_env_token_takes_precedence_without_refresh(self, monkeypatch):
        self._login_with_oauth_token(expires_in=3600)
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        with patch("huggingface_hub.utils._auth.refresh_access_token") as mock_refresh:
            assert get_token() == "hf_from_env"
        mock_refresh.assert_not_called()

    def test_short_lived_token_does_not_refresh_on_every_call(self):
        """A token lifetime shorter than the refresh margin must not trigger a refresh per call."""
        self._login_with_oauth_token(expires_in=3600)
        refreshed = {"access_token": "hf_oauth_new", "refresh_token": "rt_new", "expires_in": 60}
        with patch("huggingface_hub.utils._auth.refresh_access_token", return_value=refreshed) as mock_refresh:
            assert get_token() == "hf_oauth_new"
            assert get_token() == "hf_oauth_new"
        mock_refresh.assert_called_once()

    def test_concurrent_refresh_by_other_process_is_adopted(self):
        """If another process refreshed while we waited for the file lock, adopt its token."""
        self._login_with_oauth_token(expires_in=3600)

        @contextmanager
        def lock_after_other_process_refreshed(*args, **kwargs):
            _save_token(
                "hf_oauth_other", "oauth-user", refresh_token="rt_other", expires_at=int(time.time()) + 2592000
            )
            yield

        with (
            patch("huggingface_hub.utils._auth.WeakFileLock", lock_after_other_process_refreshed),
            patch("huggingface_hub.utils._auth.refresh_access_token") as mock_refresh,
        ):
            assert get_token() == "hf_oauth_other"
        mock_refresh.assert_not_called()


class TestFormatExpiration:
    """Drives the `expires` column of `hf auth list`."""

    def test_missing_or_corrupt_is_blank(self):
        assert _format_expiration(None) == ""
        assert _format_expiration("garbage") == ""

    def test_future_and_expired(self):
        future = str(int(time.time()) + 86400)
        past = str(int(time.time()) - 86400)
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", _format_expiration(future))
        assert _format_expiration(past).endswith(" (expired)")


class TestDeviceCodeLogin:
    @patch(
        "huggingface_hub.hf_api.whoami",
        return_value={"name": "testuser", "auth": {}},
    )
    @patch(
        "huggingface_hub._login.poll_device_token",
        return_value={"access_token": "hf_oauth_123", "refresh_token": "rt_123", "expires_in": 2592000},
    )
    @patch("huggingface_hub._login.request_device_code", return_value=_device_info())
    def test_device_code_login_success(self, mock_request, mock_poll, mock_whoami):
        _device_code_login()

        assert _get_token_from_file() == "hf_oauth_123"
        fields = _read_stored_tokens_full()["oauth-testuser"]
        assert fields["hf_token"] == "hf_oauth_123"
        assert fields["refresh_token"] == "rt_123"
        assert int(fields["expires_at"]) == pytest.approx(time.time() + 2592000, abs=60)
