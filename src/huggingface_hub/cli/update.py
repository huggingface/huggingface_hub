# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""Contains command to update the CLI to the latest version."""

import importlib.metadata
import os
import subprocess
import sys

from huggingface_hub.utils import ANSI, get_session, hf_raise_for_status, installation_method


def update() -> None:
    """Update the huggingface_hub CLI to the latest version."""
    current_version = importlib.metadata.version("huggingface_hub")

    # Fetch latest version from PyPI
    print("Checking for updates...")
    response = get_session().get("https://pypi.org/pypi/huggingface_hub/json", timeout=10)
    hf_raise_for_status(response)
    latest_version = response.json()["info"]["version"]

    if current_version == latest_version:
        print(f"Already up to date ({ANSI.bold(current_version)}).")
        return

    print(f"Updating huggingface_hub: {ANSI.bold(current_version)} -> {ANSI.bold(latest_version)}\n")

    method = installation_method()
    if method == "brew":
        cmd = ["brew", "upgrade", "hf"]
    elif method == "hf_installer":
        if os.name == "nt":
            cmd = ["powershell", "-NoProfile", "-Command", "iwr -useb https://hf.co/cli/install.ps1 | iex"]
        else:
            cmd = ["bash", "-c", "curl -LsSf https://hf.co/cli/install.sh | bash -"]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"]

    print(f"Running: {ANSI.bold(' '.join(cmd))}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n{ANSI.bold('Successfully updated to ' + latest_version)}.")
    else:
        print(f"\nUpdate failed (exit code {result.returncode}).", file=sys.stderr)
        sys.exit(result.returncode)
