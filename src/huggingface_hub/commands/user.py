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
from typing import List

from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.constants import (
    ENDPOINT,
    REPO_TYPES,
    REPO_TYPES_URL_PREFIXES,
    SPACES_SDK_TYPES,
)
from huggingface_hub.hf_api import HfApi
from requests.exceptions import HTTPError

from ..utils import HfFolder, run_subprocess
from ._cli_utils import ANSI


try:
    # Set to `True` if script is running in a Google Colab notebook.
    # If running in Google Colab, git credential store is set globally which makes the
    # warning disappear. See https://github.com/huggingface/huggingface_hub/issues/1043
    #
    # Taken from https://stackoverflow.com/a/63519730.
    # Got some trouble to make it work inside `login_token_event` callback so now set as
    # global variable.
    _is_google_colab = "google.colab" in str(get_ipython())  # noqa: F821
except NameError:
    _is_google_colab = False


class UserCommands(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        login_parser = parser.add_parser(
            "login", help="Log in using a token from huggingface.co/settings/tokens"
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
            help=(
                "{create, ls-files} Commands to interact with your huggingface.co"
                " repos."
            ),
        )
        repo_subparsers = repo_parser.add_subparsers(
            help="huggingface.co repos related commands"
        )
        repo_create_parser = repo_subparsers.add_parser(
            "create", help="Create a new repo on huggingface.co"
        )
        repo_create_parser.add_argument(
            "name",
            type=str,
            help=(
                "Name for your repo. Will be namespaced under your username to build"
                " the repo id."
            ),
        )
        repo_create_parser.add_argument(
            "--type",
            type=str,
            help=(
                'Optional: repo_type: set to "dataset" or "space" if creating a dataset'
                " or space, default is model."
            ),
        )
        repo_create_parser.add_argument(
            "--organization", type=str, help="Optional: organization namespace."
        )
        repo_create_parser.add_argument(
            "--space_sdk",
            type=str,
            help=(
                "Optional: Hugging Face Spaces SDK type. Required when --type is set to"
                ' "space".'
            ),
            choices=SPACES_SDK_TYPES,
        )
        repo_create_parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Optional: answer Yes to the prompt",
        )
        repo_create_parser.set_defaults(func=lambda args: RepoCreateCommand(args))


def currently_setup_credential_helpers(directory=None) -> List[str]:
    try:
        output = run_subprocess(
            "git config --list".split(),
            directory,
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

        To login, `huggingface_hub` now requires a token generated from https://huggingface.co/settings/tokens .
        """
        )

        token = getpass("Token: ")
        _login(self._api, token=token)


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


class LogoutCommand(BaseUserCommand):
    def run(self):
        token = HfFolder.get_token()
        if token is None:
            print("Not logged in")
            exit()
        HfFolder.delete_token()
        HfApi.unset_access_token()
        print("Successfully logged out.")


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
            prefixed_repo_id = REPO_TYPES_URL_PREFIXES[self.args.type] + repo_id
        else:
            prefixed_repo_id = repo_id

        print(f"You are about to create {ANSI.bold(prefixed_repo_id)}")

        if not self.args.yes:
            choice = input("Proceed? [Y/n] ").lower()
            if not (choice == "" or choice == "y" or choice == "yes"):
                print("Abort")
                exit()
        try:
            url = self._api.create_repo(
                repo_id=repo_id,
                token=token,
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


NOTEBOOK_LOGIN_PASSWORD_HTML = """<center> <img
src=https://huggingface.co/front/assets/huggingface_logo-noborder.svg
alt='Hugging Face'> <br> Immediately click login after typing your password or
it might be stored in plain text in this notebook file. </center>"""


NOTEBOOK_LOGIN_TOKEN_HTML_START = """<center> <img
src=https://huggingface.co/front/assets/huggingface_logo-noborder.svg
alt='Hugging Face'> <br> Copy a token from <a
href="https://huggingface.co/settings/tokens" target="_blank">your Hugging Face
tokens page</a> and paste it below. <br> Immediately click login after copying
your token or it might be stored in plain text in this notebook file. </center>"""


NOTEBOOK_LOGIN_TOKEN_HTML_END = """
<b>Pro Tip:</b> If you don't already have one, you can create a dedicated
'notebooks' token with 'write' access, that you can then easily reuse for all
notebooks. </center>"""


def notebook_login():
    """
    Displays a widget to login to the HF website and store the token.
    """
    try:
        import ipywidgets.widgets as widgets
        from IPython.display import clear_output, display
    except ImportError:
        raise ImportError(
            "The `notebook_login` function can only be used in a notebook (Jupyter or"
            " Colab) and you need the `ipywidgets` module: `pip install ipywidgets`."
        )

    box_layout = widgets.Layout(
        display="flex", flex_flow="column", align_items="center", width="50%"
    )

    token_widget = widgets.Password(description="Token:")
    token_finish_button = widgets.Button(description="Login")

    login_token_widget = widgets.VBox(
        [
            widgets.HTML(NOTEBOOK_LOGIN_TOKEN_HTML_START),
            token_widget,
            token_finish_button,
            widgets.HTML(NOTEBOOK_LOGIN_TOKEN_HTML_END),
        ],
        layout=box_layout,
    )
    display(login_token_widget)

    # On click events
    def login_token_event(t):
        token = token_widget.value
        # Erase token and clear value to make sure it's not saved in the notebook.
        token_widget.value = ""
        clear_output()
        _login(hf_api=HfApi(), token=token)

    token_finish_button.on_click(login_token_event)


def _login(hf_api: HfApi, token: str) -> None:
    if token.startswith("api_org"):
        raise ValueError("You must use your personal account token.")
    if not hf_api._is_valid_token(token=token):
        raise ValueError("Invalid token passed!")
    hf_api.set_access_token(token)
    HfFolder.save_token(token)
    print("Login successful")
    print("Your token has been saved to", HfFolder.path_token)

    # Only in Google Colab to avoid the warning message
    # See https://github.com/huggingface/huggingface_hub/issues/1043#issuecomment-1247010710
    if _is_google_colab:
        _set_store_as_git_credential_helper_globally()

    helpers = currently_setup_credential_helpers()

    if "store" not in helpers:
        print(
            ANSI.red(
                "Authenticated through git-credential store but this isn't the helper"
                " defined on your machine.\nYou might have to re-authenticate when"
                " pushing to the Hugging Face Hub. Run the following command in your"
                " terminal in case you want to set this credential helper as the"
                " default\n\ngit config --global credential.helper store"
            )
        )


def _set_store_as_git_credential_helper_globally() -> None:
    """Set globally the credential.helper to `store`.

    To be used only in Google Colab as we assume the user doesn't care about the git
    credential config. It is the only particular case where we don't want to display the
    warning message in `notebook_login()`.

    Related:
    - https://github.com/huggingface/huggingface_hub/issues/1043
    - https://github.com/huggingface/huggingface_hub/issues/1051
    - https://git-scm.com/docs/git-credential-store
    """
    try:
        run_subprocess("git config --global credential.helper store")
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)
