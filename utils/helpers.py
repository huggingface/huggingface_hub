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
from pathlib import Path


def check_and_update_file_content(file: Path, expected_content: str, update: bool):
    content = file.read_text() if file.exists() else None
    if content != expected_content:
        if update:
            file.write_text(expected_content)
            print(f"  {file} has been updated. Please make sure the changes are accurate and commit them.")
        else:
            print(f"❌ Expected content mismatch in {file}.")
            exit(1)
