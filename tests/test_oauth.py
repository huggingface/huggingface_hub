# coding=utf-8
# Copyright 2024-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains tests for the OAuth integration."""

import time
from dataclasses import asdict
from unittest.mock import patch

import httpx
import pytest
import starlette.datastructures
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from huggingface_hub._oauth import (
    _get_mocked_oauth_info,
    _get_oauth_uris,
    attach_huggingface_oauth,
    parse_huggingface_oauth,
)

from .testing_constants import TOKEN, USER


@pytest.fixture
def oauth_app(monkeypatch):
    """Defines a simple FastAPI app with OAuth integration."""
    # Mock the fact the FastAPI app is running inside a Space
    monkeypatch.setenv("SPACE_ID", "test_space/test_repo")

    app = FastAPI()

    @app.get("/")
    def greet_json(request: Request):
        oauth_info = parse_huggingface_oauth(request)
        if oauth_info is None:
            return {"msg": "Not logged in!"}
        return {
            "msg": f"Hello, {oauth_info.user_info.preferred_username}!",
            "oauth_info": asdict(oauth_info),
        }

    # path constants.OAUTH_LOGIN_PATH
    with patch.multiple(
        "huggingface_hub.constants",
        OAUTH_CLIENT_ID="dummy-app",
        OAUTH_CLIENT_SECRET="dummy-secret",
        OAUTH_SCOPES="openid profile email",
        OPENID_PROVIDER_URL="https://hub-ci.huggingface.co",
    ):
        attach_huggingface_oauth(app)

    # Little hack for the tests to work
    # On staging, the redirect_url can be only "http://localhost:3000". Here I'm simply mocking the URL to work and
    # remove any query parameters from the URL. In the test after the Hub call, we will replace back the URL with the
    # correct one.
    monkeypatch.setattr(
        "starlette.requests.Request.url_for", lambda *args: starlette.datastructures.URL("http://localhost:3000")
    )
    monkeypatch.setattr("starlette.datastructures.URL.include_query_params", lambda *args, **kwargs: args[0])
    return app


@pytest.fixture
def client(oauth_app):
    return TestClient(oauth_app)


def test_oauth_not_logged_in(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Not logged in!"}


def test_oauth_workflow(client: TestClient):
    # Make call to login
    response = client.get("/oauth/huggingface/login", follow_redirects=False)
    assert response.status_code == 302
    assert "location" in response.headers
    assert "set-cookie" in response.headers
    location = response.headers["location"]
    session_cookie = response.headers["set-cookie"]
    assert session_cookie.startswith("session=")
    session_cookie = session_cookie.split(";")[0]

    # Make call to HF Hub
    assert location.startswith("https://hub-ci.huggingface.co/oauth/authorize")
    location_authorize = location
    response_authorize = httpx.get(
        location_authorize, headers={"cookie": "token=huggingface-hub.js-cookie"}, follow_redirects=False
    )
    assert response_authorize.status_code == 303
    assert "location" in response_authorize.headers

    # Make call to callback
    location_callback = response_authorize.headers["location"].replace(
        "http://localhost:3000/", "http://testserver/oauth/huggingface/callback"
    )
    response_callback = client.get(location_callback, headers={"cookie": session_cookie}, follow_redirects=False)
    assert response_callback.status_code == 307
    assert response_callback.headers["location"] == "/"
    assert "set-cookie" in response_callback.headers
    new_session_cookie = response_callback.headers["set-cookie"].split(";")[0]
    assert len(session_cookie) < len(new_session_cookie)  # oauth data has been added to the cookie

    # Finally make a call to the root
    response_hello = client.get("/", headers={"cookie": new_session_cookie}, follow_redirects=False)
    assert response_hello.status_code == 200
    data = response_hello.json()
    assert data["msg"] == "Hello, hub.js!"
    assert data["oauth_info"]["access_token"] is not None
    assert data["oauth_info"]["scope"] == "openid profile email"
    assert data["oauth_info"]["user_info"] == {
        "sub": "62f264b9f3c90f4b6514a269",
        "name": "@huggingface/hub CI bot",
        "preferred_username": "hub.js",
        "email_verified": True,
        "email": "eliott@huggingface.co",
        "picture": "https://hub-ci.huggingface.co/avatars/934b830e9fdaa879487852f79eef7165.svg",
        "profile": "https://hub-ci.huggingface.co/hub.js",
        "website": "https://github.com/huggingface/hub.js",
        "is_pro": None,
        "can_pay": None,
        "orgs": None,
    }


def test_get_oauth_uris_default():
    login_uri, callback_uri, logout_uri = _get_oauth_uris()
    assert login_uri == "/oauth/huggingface/login"
    assert callback_uri == "/oauth/huggingface/callback"
    assert logout_uri == "/oauth/huggingface/logout"


def test_get_oauth_uris_with_prefix_stripped():
    login_uri, callback_uri, logout_uri = _get_oauth_uris("my/custom/router")
    assert login_uri == "/my/custom/router/oauth/huggingface/login"
    assert callback_uri == "/my/custom/router/oauth/huggingface/callback"
    assert logout_uri == "/my/custom/router/oauth/huggingface/logout"


def test_get_oauth_uris_with_prefix_not_stripped():
    login_uri, callback_uri, logout_uri = _get_oauth_uris("/my/custom/router/")
    assert login_uri == "/my/custom/router/oauth/huggingface/login"
    assert callback_uri == "/my/custom/router/oauth/huggingface/callback"
    assert logout_uri == "/my/custom/router/oauth/huggingface/logout"


def test_get_mocked_oauth_info(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    oauth_info = _get_mocked_oauth_info()

    # Test mock data with logged in user/token
    assert oauth_info["access_token"] == TOKEN
    assert oauth_info["userinfo"]["preferred_username"] == USER
    assert oauth_info["expires_in"] == 28800  # 8 hours
    assert oauth_info["expires_at"] <= time.time() + oauth_info["expires_in"]
    assert oauth_info["expires_at"] + 2 > time.time() + oauth_info["expires_in"]  # 2 seconds of margin
