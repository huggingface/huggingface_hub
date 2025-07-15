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
"""Contains commands for authentication (login, logout, switch, list).

Usage:
    hf auth login --token=hf_*** --add-to-git-credential
    hf auth logout --token-name=your_token_name
    hf auth switch --token-name=your_token_name
    hf auth list
"""

from argparse import _SubParsersAction

from huggingface_hub.cli import BaseHfCLICommand
from huggingface_hub.commands._cli_utils import ANSI
from huggingface_hub.commands.user import AuthListCommand as OldAuthListCommand
from huggingface_hub.commands.user import AuthSwitchCommand as OldAuthSwitchCommand
from huggingface_hub.commands.user import LoginCommand as OldLoginCommand
from huggingface_hub.commands.user import LogoutCommand as OldLogoutCommand
from huggingface_hub.commands.user import WhoamiCommand as OldWhoamiCommand
from huggingface_hub.constants import ENDPOINT
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import get_token


class AuthCommand(BaseHfCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        auth_parser = parser.add_parser("auth", help="Authentication commands")
        auth_subparsers = auth_parser.add_subparsers(help="Auth subcommands")

        # Login command
        login_parser = auth_subparsers.add_parser("login", help="Log in using a token from huggingface.co/settings/tokens")
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
        login_parser.set_defaults(func=lambda args: AuthLoginCommand(args))

        # Logout command
        logout_parser = auth_subparsers.add_parser("logout", help="Log out")
        logout_parser.add_argument(
            "--token-name",
            type=str,
            help="Optional: Name of the access token to log out from.",
        )
        logout_parser.set_defaults(func=lambda args: AuthLogoutCommand(args))

        # Switch command
        switch_parser = auth_subparsers.add_parser("switch", help="Switch between access tokens")
        switch_parser.add_argument(
            "--token-name",
            type=str,
            help="Optional: Name of the access token to switch to.",
        )
        switch_parser.add_argument(
            "--add-to-git-credential",
            action="store_true",
            help="Optional: Save token to git credential helper.",
        )
        switch_parser.set_defaults(func=lambda args: AuthSwitchCommand(args))

        # List command
        list_parser = auth_subparsers.add_parser("list", help="List all stored access tokens")
        list_parser.set_defaults(func=lambda args: AuthListCommand(args))

        # Whoami command (for backward compatibility)
        whoami_parser = auth_subparsers.add_parser("whoami", help="Find out which huggingface.co account you are logged in as")
        whoami_parser.set_defaults(func=lambda args: AuthWhoamiCommand(args))

    def run(self):
        # This should never be called since we have subcommands
        raise NotImplementedError("Auth command requires a subcommand")


class AuthLoginCommand(BaseHfCLICommand):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        # This is handled by the parent AuthCommand
        pass

    def run(self):
        # Reuse the existing implementation
        old_command = OldLoginCommand(self.args)
        old_command.run()


class AuthLogoutCommand(BaseHfCLICommand):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        # This is handled by the parent AuthCommand
        pass

    def run(self):
        # Reuse the existing implementation
        old_command = OldLogoutCommand(self.args)
        old_command.run()


class AuthSwitchCommand(BaseHfCLICommand):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        # This is handled by the parent AuthCommand
        pass

    def run(self):
        # Reuse the existing implementation
        old_command = OldAuthSwitchCommand(self.args)
        old_command.run()


class AuthListCommand(BaseHfCLICommand):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        # This is handled by the parent AuthCommand
        pass

    def run(self):
        # Reuse the existing implementation
        old_command = OldAuthListCommand(self.args)
        old_command.run()


class AuthWhoamiCommand(BaseHfCLICommand):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        # This is handled by the parent AuthCommand
        pass

    def run(self):
        # Reuse the existing implementation but without deprecation warning
        from requests.exceptions import HTTPError
        
        api = HfApi()
        token = get_token()
        if token is None:
            print("Not logged in")
            exit()
        try:
            info = api.whoami(token)
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