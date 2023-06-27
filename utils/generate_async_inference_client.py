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
import re
import tempfile
from pathlib import Path
from typing import NoReturn

from ruff.__main__ import find_ruff_bin


ASYNC_CLIENT_FILE_PATH = Path(__file__).parents[1] / "src" / "huggingface_hub" / "inference" / "_async_client.py"
SYNC_CLIENT_FILE_PATH = Path(__file__).parents[1] / "src" / "huggingface_hub" / "inference" / "_client.py"


def _add_warning_to_file_header(code: str) -> str:
    warning_message = (
        "#\n# WARNING\n# This entire file has been generated automatically based on"
        " `src/huggingface_hub/inference/_client.py`.\n# To re-generate it, run `make style` or `python"
        " ./utils/generate_async_inference_client.py --update`.\n# WARNING"
    )
    return re.sub(
        r"""
        ( # Group1: license (end)
            \n
            \#\ limitations\ under\ the\ License.
            \n
        )
        (.*?) # Group2 : all notes and comments (to be replaced)
        (\nimport[ ]) # Group3: import section (start)
        """,
        repl=rf"\1{warning_message}\3",
        string=code,
        count=1,
        flags=re.DOTALL | re.VERBOSE,
    )


def _add_aiohttp_import(code: str) -> str:
    return re.sub(
        r"(\nimport .*?\n)",
        repl=r"\1import aiohttp\n",
        string=code,
        count=1,
        flags=re.DOTALL,
    )


def _rename_to_AsyncInferenceClient(code: str) -> str:
    return code.replace("class InferenceClient:", "class AsyncInferenceClient:", 1)


ASYNC_POST_CODE = """
        url = self._resolve_url(model, task)

        if data is not None and json is not None:
            warnings.warn("Ignoring `json` as `data` is passed as binary.")

        t0 = time.time()
        timeout = self.timeout
        while True:
            with _open_as_binary(data) as data_as_binary:
                async with aiohttp.ClientSession(
                    headers=self.headers, cookies=self.cookies, timeout=aiohttp.ClientTimeout(self.timeout)
                ) as client:
                    try:
                        async with client.post(
                            url,
                            headers=build_hf_headers(),
                            json=json,
                            data=data_as_binary,
                        ) as response:
                            response.raise_for_status()
                            return await response.read()
                    except TimeoutError as error:
                        # Convert any `TimeoutError` to a `InferenceTimeoutError`
                        raise InferenceTimeoutError(f"Inference call timed out: {url}") from error
                    except aiohttp.ClientResponseError as error:
                        if response.status == 503:
                            # If Model is unavailable, either raise a TimeoutError...
                            if timeout is not None and time.time() - t0 > timeout:
                                raise InferenceTimeoutError(
                                    f"Model not loaded on the server: {url}. Please retry with a higher timeout"
                                    f" (current: {self.timeout})."
                                ) from error
                            # ...or wait 1s and retry
                            logger.info(f"Waiting for model to be loaded on the server: {error}")
                            time.sleep(1)
                            if timeout is not None:
                                timeout = max(self.timeout - (time.time() - t0), 1)  # type: ignore
                            continue
                        raise error"""


def _make_post_async(code: str) -> str:
    # Update AsyncInferenceClient.post() implementation (use aiohttp instead of requests)
    return re.sub(
        r"def post\((\n.*?\"\"\".*?\"\"\"\n).*?(\n\W*def )",
        repl=rf"async def post(\1{ASYNC_POST_CODE}\2",
        string=code,
        count=1,
        flags=re.DOTALL,
    )


def _rename_HTTPError_to_ClientResponseError_in_docstring(code: str) -> str:
    # Update `raises`-part in docstrings
    return code.replace("`HTTPError`:", "`aiohttp.ClientResponseError`:")


def _make_public_methods_async(code: str) -> str:
    # Add `async` keyword in front of public methods (of AsyncClientInference)
    return re.sub(
        r"""
        # Group 1: newline  + 4-spaces indent
        (\n\ {4})
        # Group 2: def + method name + parenthesis + optionally type: ignore + self
        (
            def[ ] # def
            [a-z]\w*? # method name (not starting by _)
            \( # parenthesis
            (\s*\#[ ]type:[ ]ignore)? # optionally type: ignore
            \s*self, # expect self, as first arg
        )""",
        repl=r"\1async \2",  # insert "async" keyword
        string=code,
        flags=re.DOTALL | re.VERBOSE,
    )


def _await_post_method_call(code: str) -> str:
    return code.replace("self.post(", "await self.post(")


def _remove_examples_from_public_methods(code: str) -> str:
    # "Example" sections are not valid in async methods. Let's remove them.
    return re.sub(
        r"""
        \n\s*
        Example:\n\s* # example section
        ```py # start
        .*? # anything
        ``` # end
        \n
        """,
        repl="\n",
        string=code,
        flags=re.DOTALL | re.VERBOSE,
    )


def generate_async_client_code(code: str) -> str:
    """Generate AsyncInferenceClient source code."""
    code = _add_warning_to_file_header(code)
    code = _add_aiohttp_import(code)
    code = _rename_to_AsyncInferenceClient(code)
    code = _make_post_async(code)
    code = _rename_HTTPError_to_ClientResponseError_in_docstring(code)
    code = _make_public_methods_async(code)
    code = _await_post_method_call(code)
    code = _remove_examples_from_public_methods(code)
    return code


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
