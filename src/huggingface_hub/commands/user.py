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

import subprocess
from argparse import ArgumentParser
from getpass import getpass
from typing import List, Union

from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.constants import (
    REPO_TYPES,
    REPO_TYPES_URL_PREFIXES,
    SPACES_SDK_TYPES,
)
from huggingface_hub.hf_api import HfApi, HfFolder
from requests.exceptions import HTTPError


class UserCommands(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        login_parser = parser.add_parser(
            "login", help="Log in using the same credentials as on huggingface.co"
        )
        login_parser.set_defaults(func=lambda args: LoginCommand(args))
        whoami_parser = parser.add_parser(
            "whoami", help="Find out which huggingface.co account you are logged in as."
        )
        whoami_parser.set_defaults(func=lambda args: WhoamiCommand(args))
        logout_parser = parser.add_parser("logout", help="Log out")
        logout_parser.set_defaults(func=lambda args: LogoutCommand(args))

        # new system: git-based repo system
        repo_parser = parser.add_parser(
            "repo",
            help="{create, ls-files} Commands to interact with your huggingface.co repos.",
        )
        repo_subparsers = repo_parser.add_subparsers(
            help="huggingface.co repos related commands"
        )
        ls_parser = repo_subparsers.add_parser(
            "ls-files", help="List all your files on huggingface.co"
        )
        ls_parser.add_argument(
            "--organization", type=str, help="Optional: organization namespace."
        )
        ls_parser.set_defaults(func=lambda args: ListReposObjsCommand(args))
        repo_create_parser = repo_subparsers.add_parser(
            "create", help="Create a new repo on huggingface.co"
        )
        repo_create_parser.add_argument(
            "name",
            type=str,
            help="Name for your repo. Will be namespaced under your username to build the repo id.",
        )
        repo_create_parser.add_argument(
            "--type",
            type=str,
            help='Optional: repo_type: set to "dataset" or "space" if creating a dataset or space, default is model.',
        )
        repo_create_parser.add_argument(
            "--organization", type=str, help="Optional: organization namespace."
        )
        repo_create_parser.add_argument(
            "--space_sdk",
            type=str,
            help='Optional: Hugging Face Spaces SDK type. Required when --type is set to "space".',
            choices=SPACES_SDK_TYPES,
        )
        repo_create_parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Optional: answer Yes to the prompt",
        )
        repo_create_parser.set_defaults(func=lambda args: RepoCreateCommand(args))


class ANSI:
    """
    Helper for en.wikipedia.org/wiki/ANSI_escape_code
    """

    _bold = "\u001b[1m"
    _red = "\u001b[31m"
    _gray = "\u001b[90m"
    _reset = "\u001b[0m"

    @classmethod
    def bold(cls, s):
        return f"{cls._bold}{s}{cls._reset}"

    @classmethod
    def red(cls, s):
        return f"{cls._bold + cls._red}{s}{cls._reset}"

    @classmethod
    def gray(cls, s):
        return f"{cls._gray}{s}{cls._reset}"


def tabulate(rows: List[List[Union[str, int]]], headers: List[str]) -> str:
    """
    Inspired by:

    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    """
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    row_format = ("{{:{}}} " * len(headers)).format(*col_widths)
    lines = []
    lines.append(row_format.format(*headers))
    lines.append(row_format.format(*["-" * w for w in col_widths]))
    for row in rows:
        lines.append(row_format.format(*row))
    return "\n".join(lines)


def currently_setup_credential_helpers(directory=None) -> List[str]:
    try:
        output = subprocess.run(
            "git config --list".split(),
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding="utf-8",
            check=True,
            cwd=directory,
        ).stdout.split("\n")

        current_credential_helpers = []
        for line in output:
            if "credential.helper" in line:
                current_credential_helpers.append(line.split("=")[-1])
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)

    return current_credential_helpers


class BaseUserCommand:
    def __init__(self, args):
        self.args = args
        self._api = HfApi()


class LoginCommand(BaseUserCommand):
    def run(self):
        print(  # docstyle-ignore
            """
        _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
        _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
        _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
        _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
        _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

        To login, `huggingface_hub` now requires a token generated from https://huggingface.co/settings/token.
        (Deprecated, will be removed in v0.3.0) To login with username and password instead, interrupt with Ctrl+C.
        """
        )

        try:
            token = getpass("Token: ")
            _login(self._api, token=token)

        except KeyboardInterrupt:
            username = input("\rUsername: ")
            password = getpass()
            _login(self._api, username, password)


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
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)


class LogoutCommand(BaseUserCommand):
    def run(self):
        token = HfFolder.get_token()
        if token is None:
            print("Not logged in")
            exit()
        HfFolder.delete_token()
        HfApi.unset_access_token()
        try:
            self._api.logout(token)
        except HTTPError as e:
            # Logging out with an access token will return a client error.
            if not e.response.status_code == 400:
                raise e
        print("Successfully logged out.")


class ListReposObjsCommand(BaseUserCommand):
    def run(self):
        token = HfFolder.get_token()
        if token is None:
            print("Not logged in")
            exit(1)
        try:
            objs = self._api.list_repos_objs(token, organization=self.args.organization)
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        if len(objs) == 0:
            print("No shared file yet")
            exit()
        rows = [[obj.filename, obj.lastModified, obj.commit, obj.size] for obj in objs]
        print(
            tabulate(rows, headers=["Filename", "LastModified", "Commit-Sha", "Size"])
        )


class RepoCreateCommand(BaseUserCommand):
    def run(self):
        token = HfFolder.get_token()
        if token is None:
            print("Not logged in")
            exit(1)
        try:
            stdout = subprocess.check_output(["git", "--version"]).decode("utf-8")
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            print("Looks like you do not have git installed, please install.")

        try:
            stdout = subprocess.check_output(["git-lfs", "--version"]).decode("utf-8")
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            print(
                ANSI.red(
                    "Looks like you do not have git-lfs installed, please install."
                    " You can install from https://git-lfs.github.com/."
                    " Then run `git lfs install` (you only have to do this once)."
                )
            )
        print("")

        user = self._api.whoami(token)["name"]
        namespace = (
            self.args.organization if self.args.organization is not None else user
        )

        repo_id = f"{namespace}/{self.args.name}"

        if self.args.type not in REPO_TYPES:
            print("Invalid repo --type")
            exit(1)

        if self.args.type in REPO_TYPES_URL_PREFIXES:
            repo_id = REPO_TYPES_URL_PREFIXES[self.args.type] + repo_id

        print(f"You are about to create {ANSI.bold(repo_id)}")

        if not self.args.yes:
            choice = input("Proceed? [Y/n] ").lower()
            if not (choice == "" or choice == "y" or choice == "yes"):
                print("Abort")
                exit()
        try:
            url = self._api.create_repo(
                self.args.name,
                token=token,
                organization=self.args.organization,
                repo_type=self.args.type,
                space_sdk=self.args.space_sdk,
            )
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        print("\nYour repo now lives at:")
        print(f"  {ANSI.bold(url)}")
        print(
            "\nYou can clone it locally with the command below,"
            " and commit/push as usual."
        )
        print(f"\n  git clone {url}")
        print("")


NOTEBOOK_LOGIN_PASSWORD_HTML = """<center>
<img src=https://huggingface.co/front/assets/huggingface_logo-noborder.svg alt='Hugging Face'>
<br>
Immediately click login after typing your password or it might be stored in plain text in this notebook file.
</center>"""


NOTEBOOK_LOGIN_TOKEN_HTML_START = """<center>
<img src=https://huggingface.co/front/assets/huggingface_logo-noborder.svg alt='Hugging Face'>
<br>
Copy a token from <a href="https://huggingface.co/settings/token" target="_blank">your Hugging Face tokens page</a> and paste it below.
<br>
Immediately click login after copying your token or it might be stored in plain text in this notebook file.
</center>"""


NOTEBOOK_LOGIN_TOKEN_HTML_END = """
<b>Pro Tip:</b> If you don't already have one, you can create a dedicated 'notebooks' token with 'write' access, that you can then easily reuse for all notebooks.
<br>
<i>Logging in with your username and password is deprecated and won't be possible anymore in the near future. You can still use them for now by clicking below.</i>
</center>"""


def notebook_login():
    """
    Displays a widget to login to the HF website and store the token.
    """
    try:
        import ipywidgets.widgets as widgets
        from IPython.display import clear_output, display
    except ImportError:
        raise ImportError(
            "The `notebook_login` function can only be used in a notebook (Jupyter or Colab) and you need the "
            "`ipywdidgets` module: `pip install ipywidgets`."
        )

    box_layout = widgets.Layout(
        display="flex", flex_flow="column", align_items="center", width="50%"
    )

    token_widget = widgets.Password(description="Token:")
    token_finish_button = widgets.Button(description="Login")
    switch_button = widgets.Button(description="Use password")

    login_token_widget = widgets.VBox(
        [
            widgets.HTML(NOTEBOOK_LOGIN_TOKEN_HTML_START),
            token_widget,
            token_finish_button,
            widgets.HTML(NOTEBOOK_LOGIN_TOKEN_HTML_END),
            switch_button,
        ],
        layout=box_layout,
    )
    display(login_token_widget)

    # Deprecated page for login
    input_widget = widgets.Text(description="Username:")
    password_widget = widgets.Password(description="Password:")
    password_finish_button = widgets.Button(description="Login")

    login_password_widget = widgets.VBox(
        [
            widgets.HTML(value=NOTEBOOK_LOGIN_PASSWORD_HTML),
            widgets.HBox([input_widget, password_widget]),
            password_finish_button,
        ],
        layout=box_layout,
    )

    # On click events
    def login_token_event(t):
        token = token_widget.value
        # Erase token and clear value to make sure it's not saved in the notebook.
        token_widget.value = ""
        clear_output()
        _login(HfApi(), token=token)

    token_finish_button.on_click(login_token_event)

    def login_password_event(t):
        username = input_widget.value
        password = password_widget.value
        # Erase password and clear value to make sure it's not saved in the notebook.
        password_widget.value = ""
        clear_output()
        _login(HfApi(), username=username, password=password)

    password_finish_button.on_click(login_password_event)

    def switch_event(t):
        clear_output()
        display(login_password_widget)

    switch_button.on_click(switch_event)


def _login(hf_api, username=None, password=None, token=None):
    if token is None:
        try:
            token = hf_api.login(username, password)
        except HTTPError as e:
            # probably invalid credentials, display error message.
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
    elif not hf_api._is_valid_token(token):
        raise ValueError("Invalid token passed.")

    hf_api.set_access_token(token)
    HfFolder.save_token(token)
    print("Login successful")
    print("Your token has been saved to", HfFolder.path_token)
    helpers = currently_setup_credential_helpers()

    if "store" not in helpers:
        print(
            ANSI.red(
                "Authenticated through git-credential store but this isn't the helper defined on your machine.\nYou "
                "might have to re-authenticate when pushing to the Hugging Face Hub. Run the following command in your "
                "terminal in case you want to set this credential helper as the default\n\ngit config --global credential.helper store"
            )
        )
