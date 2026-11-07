import time
from pathlib import Path

import pytest

from huggingface_hub.constants import ENDPOINT
from huggingface_hub.utils import run_interactive_subprocess, run_subprocess
from huggingface_hub.utils._git_credential import (
    _parse_credential_output,
    list_credential_helpers,
    set_git_credential,
    unset_git_credential,
)


STORE_AND_CACHE_HELPERS_CONFIG = """
[credential]
    helper = store
    helper = cache --timeout 30000
    helper = git-credential-manager
    helper = /usr/libexec/git-core/git-credential-libsecret
"""


class TestGitCredentials:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Initialize and configure a local repo.

        Avoid to configure git helpers globally on a contributor's machine.
        """
        run_subprocess("git init", folder=tmp_path)
        with (tmp_path / ".git" / "config").open("w") as f:
            f.write(STORE_AND_CACHE_HELPERS_CONFIG)
        yield

    def test_list_credential_helpers(self, tmp_path: Path) -> None:
        helpers = list_credential_helpers(folder=tmp_path)
        assert "cache" in helpers
        assert "store" in helpers
        assert "git-credential-manager" in helpers
        assert "/usr/libexec/git-core/git-credential-libsecret" in helpers

    def test_set_and_unset_git_credential(self, tmp_path: Path) -> None:
        username = "hf_test_user_" + str(round(time.time()))  # make username unique

        # Set credentials
        set_git_credential(token="hf_test_token", username=username, folder=tmp_path)

        # Check credentials are stored
        with run_interactive_subprocess("git credential fill", folder=tmp_path) as (stdin, stdout):
            stdin.write(f"url={ENDPOINT}\nusername={username}\n\n")
            stdin.flush()
            output = stdout.read()
        assert "password=hf_test_token" in output

        # Unset credentials
        unset_git_credential(username=username, folder=tmp_path)

        # Check credentials are NOT stored
        # Cannot check with `git credential fill` as it would hang forever: only
        # checking `store` helper instead.
        with run_interactive_subprocess("git credential-store get", folder=tmp_path) as (stdin, stdout):
            stdin.write(f"url={ENDPOINT}\nusername={username}\n\n")
            stdin.flush()
            output = stdout.read()
        assert output == ""

    def test_git_credential_parsing_regex(self) -> None:
        output = """
            credential.helper = store
            credential.helper = cache --timeout 30000
        credential.helper = osxkeychain"""
        assert _parse_credential_output(output) == ["cache", "osxkeychain", "store"]
