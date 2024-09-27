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
"""Contains helpers used by the scripts in `./utils`."""

import subprocess
import tempfile
from pathlib import Path

from ruff.__main__ import find_ruff_bin


def check_and_update_file_content(file: Path, expected_content: str, update: bool):
    # Ensure the expected content ends with a newline to satisfy end-of-file-fixer hook
    expected_content = expected_content.rstrip("\n") + "\n"
    content = file.read_text() if file.exists() else None
    if content != expected_content:
        if update:
            file.write_text(expected_content)
            print(f"  {file} has been updated. Please make sure the changes are accurate and commit them.")
        else:
            print(f"❌ Expected content mismatch in {file}.")
            exit(1)


def format_source_code(code: str) -> str:
    """Format the generated source code using Ruff."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "tmp.py"
        filepath.write_text(code)
        ruff_bin = find_ruff_bin()
        if not ruff_bin:
            raise FileNotFoundError("Ruff executable not found.")
        try:
            subprocess.run([ruff_bin, "check", str(filepath), "--fix", "--quiet"], check=True)
            subprocess.run([ruff_bin, "format", str(filepath), "--quiet"], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error running Ruff: {e}")
        return filepath.read_text()
