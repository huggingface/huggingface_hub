import subprocess
from typing import Optional

import pytest

from huggingface_hub._login import _set_store_as_git_credential_helper_globally
from huggingface_hub.utils import run_subprocess


class TestSetGlobalStore:
    previous_config: Optional[str]

    @pytest.fixture(autouse=True)
    def setup(self):
        """Get current global config value."""
        try:
            self.previous_config = run_subprocess("git config --global credential.helper").stdout
        except subprocess.CalledProcessError:
            self.previous_config = None  # Means global credential.helper value not set

        run_subprocess("git config --global credential.helper store")

        yield

        # Reset global config value.
        if self.previous_config is None:
            run_subprocess("git config --global --unset credential.helper")
        else:
            run_subprocess(f"git config --global credential.helper {self.previous_config}")

    def test_set_store_as_git_credential_helper_globally(self) -> None:
        """Test `_set_store_as_git_credential_helper_globally` works as expected.

        Previous value from the machine is restored after the test.
        """
        _set_store_as_git_credential_helper_globally()
        new_config = run_subprocess("git config --global credential.helper").stdout
        assert new_config == "store\n"
