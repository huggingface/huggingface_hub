# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import sys
from unittest.mock import patch

import pytest

from huggingface_hub.cli import system


@pytest.mark.parametrize("argv0", [r"C:\\venv\\Scripts\\hf.exe", r"C:\\venv\\Scripts\\hf"])
def test_windows_pip_hf_launcher_update_is_deferred_until_hf_exits(argv0: str) -> None:
    expected_command = subprocess.list2cmdline([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])

    with (
        patch("huggingface_hub.cli.system.sys.platform", "win32"),
        patch("huggingface_hub.cli.system.sys.argv", [argv0, "update"]),
        patch("huggingface_hub.cli.system.installation_method", return_value="pip"),
        patch("huggingface_hub.cli.system._fetch_latest_pypi_version", return_value="999.0.0"),
        patch("huggingface_hub.cli.system.run_update") as mock_run_update,
        patch("huggingface_hub.cli.system.out.warning") as mock_warning,
        patch("huggingface_hub.cli.system.out.hint") as mock_hint,
    ):
        system.update()

    mock_run_update.assert_not_called()
    mock_warning.assert_called_once_with(
        "Cannot safely update a pip-installed `hf` CLI while the `hf` launcher is running on Windows."
    )
    mock_hint.assert_called_once_with(f"After this command exits, run: {expected_command}")


def test_windows_pip_module_update_still_runs_in_process() -> None:
    with (
        patch("huggingface_hub.cli.system.sys.platform", "win32"),
        patch("huggingface_hub.cli.system.sys.argv", ["huggingface_hub.cli.hf", "update"]),
        patch("huggingface_hub.cli.system.installation_method", return_value="pip"),
        patch("huggingface_hub.cli.system._fetch_latest_pypi_version", return_value="999.0.0"),
        patch("huggingface_hub.cli.system._installed_hf_cli_dirs", return_value=[]),
        patch("huggingface_hub.cli.system.run_update", return_value=0) as mock_run_update,
    ):
        system.update()

    mock_run_update.assert_called_once_with(exclude_skill=True)


def test_non_windows_pip_update_still_runs_in_process() -> None:
    with (
        patch("huggingface_hub.cli.system.sys.platform", "linux"),
        patch("huggingface_hub.cli.system.installation_method", return_value="pip"),
        patch("huggingface_hub.cli.system._fetch_latest_pypi_version", return_value="999.0.0"),
        patch("huggingface_hub.cli.system._installed_hf_cli_dirs", return_value=[]),
        patch("huggingface_hub.cli.system.run_update", return_value=0) as mock_run_update,
    ):
        system.update()

    mock_run_update.assert_called_once_with(exclude_skill=True)
