# Copyright 2020 The HuggingFace Team. All rights reserved.
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
from argparse import _SubParsersAction

from requests.exceptions import HTTPError

from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.constants import (
    ENDPOINT,
)
from huggingface_hub.hf_api import HfApi

from .._login import (  # noqa: F401 # for backward compatibility  # noqa: F401 # for backward compatibility
    NOTEBOOK_LOGIN_PASSWORD_HTML,
    NOTEBOOK_LOGIN_TOKEN_HTML_END,
    NOTEBOOK_LOGIN_TOKEN_HTML_START,
    login,
    logout,
    notebook_login,
)
from ..utils import HfFolder
from ._cli_utils import ANSI


class UserCommands(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        login_parser = parser.add_parser("login", help="Log in using a token from huggingface.co/settings/tokens")
        login_parser.add_argument(
            "--token",
            type=str,
            help="Token generated from https://huggingface.co/settings/tokens",
        )
        login_parser.add_argument(
            "--add-to-git-credential",
            action="store_true",
            help="Optional: Save token to git credential helper.",
        )
        login_parser.set_defaults(func=lambda args: LoginCommand(args))
        whoami_parser = parser.add_parser("whoami", help="Find out which huggingface.co account you are logged in as.")
        whoami_parser.set_defaults(func=lambda args: WhoamiCommand(args))
        logout_parser = parser.add_parser("logout", help="Log out")
        logout_parser.set_defaults(func=lambda args: LogoutCommand(args))


class BaseUserCommand:
    def __init__(self, args):
        self.args = args
        self._api = HfApi()


class LoginCommand(BaseUserCommand):
    def run(self):
        login(token=self.args.token, add_to_git_credential=self.args.add_to_git_credential)


class LogoutCommand(BaseUserCommand):
    def run(self):
        logout()


class WhoamiCommand(BaseUserCommand):
    def run(self):
        token = HfFolder.get_token()
        if token is None:
            print("Not logged in")
            exit()
        try:
            info = self._api.whoami(token)
            print(info["name"])
            orgs = [org["name"] for org in info["orgs"]]
            if orgs:
                print(ANSI.bold("orgs: "), ",".join(orgs))

            if ENDPOINT != "https://huggingface.co":
                print(f"Authenticated through private endpoint: {ENDPOINT}")
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
