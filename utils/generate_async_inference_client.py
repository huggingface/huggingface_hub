# coding=utf-8
# Copyright 2023-present, the HuggingFace Inc. team.
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
"""Contains a tool to generate `src/huggingface_hub/inference/_async_client.py`."""
import argparse
import os
import tempfile
from pathlib import Path
from typing import NoReturn

from ruff.__main__ import find_ruff_bin


ASYNC_CLIENT_FILE_PATH = Path(__file__).parents[1] / "src" / "huggingface_hub" / "inference" / "_async_client.py"
SYNC_CLIENT_FILE_PATH = Path(__file__).parents[1] / "src" / "huggingface_hub" / "inference" / "_client.py"


def generate_async_client_code(sync_client_code: str) -> str:
    """Generate AsyncInferenceClient source code."""
    return ASYNC_CLIENT_FILE_PATH.read_text()


def format_source_code(code: str) -> str:
    """Apply formatter on a generated source code."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "async_client.py"
        filepath.write_text(code)
        ruff_bin = find_ruff_bin()
        os.spawnv(os.P_WAIT, ruff_bin, ["ruff", str(filepath), "--fix", "--quiet"])
        return filepath.read_text()


def check_async_client(update: bool) -> NoReturn:
    """Check AsyncInferenceClient is correctly defined and consistent with InferenceClient.

    This script is used in the `make style` and `make quality` checks.
    """
    sync_client_code = SYNC_CLIENT_FILE_PATH.read_text()
    current_async_client_code = ASYNC_CLIENT_FILE_PATH.read_text()

    raw_async_client_code = generate_async_client_code(sync_client_code)

    formatted_async_client_code = format_source_code(raw_async_client_code)

    # If expected `__init__.py` content is different, test fails. If '--update-init-file'
    # is used, `__init__.py` file is updated before the test fails.
    if current_async_client_code != formatted_async_client_code:
        if update:
            ASYNC_CLIENT_FILE_PATH.write_text(formatted_async_client_code)

            print(
                "✅ AsyncInferenceClient source code has been updated in"
                " `./src/huggingface_hub/inference/_async_client.py`.\n   Please make sure the changes are accurate"
                " and commit them."
            )
            exit(0)
        else:
            print(
                "❌ Expected content mismatch in `./src/huggingface_hub/inference/_async_client.py`.\n   It is most"
                " likely that you modified some InferenceClient code and did not update the the AsyncInferenceClient"
                " one.\n   Please run `make style` or `python utils/generate_async_inference_client.py --update`."
            )
            exit(1)

    print("✅ All good! (AsyncInferenceClient)")
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update",
        action="store_true",
        help="Whether to re-generate `./src/huggingface_hub/inference/_async_client.py` if a change is detected.",
    )
    args = parser.parse_args()

    check_async_client(update=args.update)
