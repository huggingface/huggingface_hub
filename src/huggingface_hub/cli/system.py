# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""Contains commands to print information about the environment and version."""

import shutil
import subprocess
import sys

import click

from huggingface_hub import __version__, constants

from ..utils import dump_environment_info, installation_method
from ._cli_utils import _fetch_latest_pypi_version, run_update
from ._output import out
from ._skills import DEFAULT_SKILL_ID, _installed_hf_cli_dirs


def env() -> None:
    """Print information about the environment."""
    dump_environment_info()


def version() -> None:
    """Print information about the hf version."""
    out.result("hf version", version=__version__)


def update() -> None:
    """Update the `hf` CLI to the latest version."""
    out.text(f"Current version: {__version__}")
    out.text("Checking for updates to latest version...")
    latest_version = _fetch_latest_pypi_version("huggingface_hub")
    if latest_version is not None and __version__ == latest_version:
        out.text(f"hf is up to date ({__version__})")
        return

    # The standalone installer installs the `hf-cli` skill by default. If it's not installed at this
    # point, the user opted out (or removed it): tell the installer to leave it alone instead of
    # silently undoing that choice.
    skill_installed = bool(
        _installed_hf_cli_dirs(constants.AGENTS_SKILLS_GLOBAL_PATH, constants.CLAUDE_SKILLS_GLOBAL_PATH)
    )

    returncode = run_update(exclude_skill=not skill_installed)
    if returncode != 0:
        raise click.exceptions.Exit(code=returncode)

    if not skill_installed:
        out.hint("Run `hf skills add -g --claude` to teach your AI agents how to use the `hf` CLI.")
        return

    # Refresh the globally installed skill so agents see the new command surface. Runs in a
    # subprocess: the skill is generated from the CLI code, which has just been replaced on disk
    # while this process still runs the previous version.
    out.text(f"Updating the `{DEFAULT_SKILL_ID}` skill...")
    subprocess.call([*_hf_argv(), "skills", "update", DEFAULT_SKILL_ID, "-g", "--claude"])


def _hf_argv() -> list[str]:
    """argv prefix to invoke the freshly updated `hf` CLI."""
    if installation_method() == "brew":
        # Homebrew installs the new version in a new prefix: the current interpreter is the old one.
        hf_bin = shutil.which("hf")
        if hf_bin is not None:
            return [hf_bin]
    return [sys.executable, "-m", "huggingface_hub.cli.hf"]
