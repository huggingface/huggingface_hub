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
"""Contains a tool to generate `src/huggingface_hub/inference/_generated/_async_client.py`."""

import argparse
import re
from pathlib import Path
from typing import NoReturn

from helpers import format_source_code


ASYNC_CLIENT_FILE_PATH = (
    Path(__file__).parents[1] / "src" / "huggingface_hub" / "inference" / "_generated" / "_async_client.py"
)
SYNC_CLIENT_FILE_PATH = Path(__file__).parents[1] / "src" / "huggingface_hub" / "inference" / "_client.py"


def generate_async_client_code(code: str) -> str:
    """Generate AsyncInferenceClient source code."""
    # Warning message "this is an automatically generated file"
    code = _add_warning_to_file_header(code)

    # Imports specific to asyncio
    code = _add_imports(code)

    # Define `AsyncInferenceClient`
    code = _rename_to_AsyncInferenceClient(code)

    # Refactor `.post` method to be async + adapt calls
    code = _make_inner_post_async(code)
    code = _await_inner_post_method_call(code)

    # Handle __enter__, __exit__, close
    code = _remove_enter_exit_stack(code)

    # Use _async_stream_text_generation_response
    code = _use_async_streaming_util(code)

    # Make all tasks-method async
    code = _make_tasks_methods_async(code)

    # Adapt text_generation to async
    code = _adapt_text_generation_to_async(code)

    # Adapt chat_completion to async
    code = _adapt_chat_completion_to_async(code)

    # Update some docstrings
    code = _update_examples_in_public_methods(code)

    # Adapt /info and /health endpoints
    code = _adapt_info_and_health_endpoints(code)

    # Adapt the proxy client (for client.chat.completions.create)
    code = _adapt_proxy_client(code)

    return code


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
                " `./src/huggingface_hub/inference/_generated/_async_client.py`.\n   Please make sure the changes are"
                " accurate and commit them."
            )
            exit(0)
        else:
            print(
                "❌ Expected content mismatch in `./src/huggingface_hub/inference/_generated/_async_client.py`.\n   It"
                " is most likely that you modified some InferenceClient code and did not update the"
                " AsyncInferenceClient one.\n   Please run `make style` or `python"
                " utils/generate_async_inference_client.py --update`."
            )
            exit(1)

    print("✅ All good! (AsyncInferenceClient)")
    exit(0)


def _add_warning_to_file_header(code: str) -> str:
    warning_message = (
        "#\n# WARNING\n# This entire file has been adapted from the sync-client code in"
        " `src/huggingface_hub/inference/_client.py`.\n# Any change in InferenceClient will be automatically reflected"
        " in AsyncInferenceClient.\n# To re-generate the code, run `make style` or `python"
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


def _add_imports(code: str) -> str:
    # global imports
    code = re.sub(
        r"(\nimport .*?\n)",
        repl=(
            r"\1"
            + "from .._common import _async_yield_from\n"
            + "from huggingface_hub.utils import get_async_session\n"
            + "from typing import AsyncIterable\n"
            + "from contextlib import AsyncExitStack\n"
            + "from typing import Set\n"
            + "import asyncio\n"
            + "import httpx\n"
        ),
        string=code,
        count=1,
        flags=re.DOTALL,
    )

    # type-checking imports
    code = re.sub(
        r"(\nif TYPE_CHECKING:\n)",
        repl=r"\1    from aiohttp import ClientResponse, ClientSession\n",
        string=code,
        count=1,
        flags=re.DOTALL,
    )

    return code


def _rename_to_AsyncInferenceClient(code: str) -> str:
    return code.replace("class InferenceClient:", "class AsyncInferenceClient:", 1)


ASYNC_INNER_POST_CODE = """
        # TODO: this should be handled in provider helpers directly
        if request_parameters.task in TASKS_EXPECTING_IMAGES and "Accept" not in request_parameters.headers:
            request_parameters.headers["Accept"] = "image/png"

        try:
            client = await self._get_async_client()
            if stream:
                response = await self.exit_stack.enter_async_context(
                    client.stream(
                        "POST",
                        request_parameters.url,
                        json=request_parameters.json,
                        data=request_parameters.data,
                        headers=request_parameters.headers,
                        cookies=self.cookies,
                        timeout=self.timeout,
                    )
                )
                hf_raise_for_status(response)
                return _async_yield_from(client, response)
            else:
                response = await client.post(
                    request_parameters.url,
                    json=request_parameters.json,
                    data=request_parameters.data,
                    headers=request_parameters.headers,
                    cookies=self.cookies,
                    timeout=self.timeout,
                )
                hf_raise_for_status(response)
                return response.content
        except asyncio.TimeoutError as error:
            # Convert any `TimeoutError` to a `InferenceTimeoutError`
            raise InferenceTimeoutError(f"Inference call timed out: {request_parameters.url}") from error  # type: ignore
        except HfHubHTTPError as error:
            if error.response.status_code == 422 and request_parameters.task != "unknown":
                msg = str(error.args[0])
                if len(error.response.text) > 0:
                    msg += f"{os.linesep}{error.response.text}{os.linesep}"
                error.args = (msg,) + error.args[1:]
            raise
            """


def _make_inner_post_async(code: str) -> str:
    # Update AsyncInferenceClient._inner_post() implementation
    code = re.sub(
        r"""
        def[ ]_inner_post\( # definition
            (\n.*?\"\"\".*?\"\"\"\n) # Group1: docstring
            .*? # implementation (to be overwritten)
        (\n\W*def ) # Group2: next method
        """,
        repl=rf"async def _inner_post(\1{ASYNC_INNER_POST_CODE}\2",
        string=code,
        count=1,
        flags=re.DOTALL | re.VERBOSE,
    )
    # Update `post`'s type annotations
    code = code.replace("    def _inner_post(", "    async def _inner_post(")
    return code


ENTER_EXIT_STACK_SYNC_CODE = """
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_stack.close()

    def close(self):
        self.exit_stack.close()"""

ENTER_EXIT_STACK_ASYNC_CODE = """
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    async def close(self):
        \"""Close the client.

        This method is automatically called when using the client as a context manager.
        \"""
        await self.exit_stack.aclose()

    async def _get_async_client(self):
        \"""Get a unique async client for this AsyncInferenceClient instance.

        Returns the same client instance on subsequent calls, ensuring proper
        connection reuse and resource management through the exit stack.
        \"""
        if self._async_client is None:
            self._async_client = await self.exit_stack.enter_async_context(get_async_session())
        return self._async_client
"""


def _remove_enter_exit_stack(code: str) -> str:
    code = code.replace(
        "exit_stack = ExitStack()",
        "exit_stack = AsyncExitStack()\n        self._async_client: Optional[httpx.AsyncClient] = None",
    )
    code = code.replace(ENTER_EXIT_STACK_SYNC_CODE, ENTER_EXIT_STACK_ASYNC_CODE)
    return code


def _make_tasks_methods_async(code: str) -> str:
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
            (\s*\#[ ]type:[ ]ignore(\[misc\])?)? # optionally 'type: ignore' or 'type: ignore[misc]'
            \s*self, # expect self, as first arg
        )""",
        repl=r"\1async \2",  # insert "async" keyword
        string=code,
        flags=re.DOTALL | re.VERBOSE,
    )


def _adapt_text_generation_to_async(code: str) -> str:
    # Text-generation task has to be handled specifically since it has a recursive call mechanism (to retry on non-tgi servers)

    # Await recursive call
    code = code.replace(
        "return self.text_generation",
        "return await self.text_generation",
    )
    code = code.replace(
        "return self.chat_completion",
        "return await self.chat_completion",
    )

    # Update return types: Iterable -> AsyncIterable
    code = code.replace(
        "Iterable[",
        "AsyncIterable[",
    )

    return code


def _adapt_chat_completion_to_async(code: str) -> str:
    # Await text-generation call
    code = code.replace(
        "text_generation_output = self.text_generation(",
        "text_generation_output = await self.text_generation(",
    )

    return code


def _await_inner_post_method_call(code: str) -> str:
    return code.replace("self._inner_post(", "await self._inner_post(")


def _update_example_code_block(code_block: str) -> str:
    """Update an atomic code block example from a docstring."""
    code_block = "\n        # Must be run in an async context" + code_block
    code_block = code_block.replace("InferenceClient", "AsyncInferenceClient")
    code_block = code_block.replace("client.", "await client.")
    code_block = code_block.replace(">>> for ", ">>> async for ")
    return code_block


def _update_examples_in_public_methods(code: str) -> str:
    for match in re.finditer(
        r"""
        \n\s*
        Example.*?:\n\s* # example section
        ```py # start
        (.*?) # code block
        ``` # end
        \n
        """,
        string=code,
        flags=re.DOTALL | re.VERBOSE,
    ):
        # Example, including code block
        full_match = match.group()

        # Code block alone
        code_block = match.group(1)

        # Update code block in example
        updated_match = full_match.replace(code_block, _update_example_code_block(code_block))

        # Update example in full script
        code = code.replace(full_match, updated_match)

    return code


def _use_async_streaming_util(code: str) -> str:
    code = code.replace(
        "_stream_text_generation_response",
        "_async_stream_text_generation_response",
    )
    code = code.replace("_stream_chat_completion_response", "_async_stream_chat_completion_response")
    return code


def _adapt_info_and_health_endpoints(code: str) -> str:
    get_url_sync_snippet = """
        response = get_session().get(url, headers=build_hf_headers(token=self.token))"""

    get_url_async_snippet = """
        client = await self._get_async_client()
        response = await client.get(url, headers=build_hf_headers(token=self.token))"""

    return code.replace(get_url_sync_snippet, get_url_async_snippet)


def _adapt_proxy_client(code: str) -> str:
    return code.replace(
        "def __init__(self, client: InferenceClient):",
        "def __init__(self, client: AsyncInferenceClient):",
    )


def _add_before(code: str, pattern: str, addition: str) -> str:
    index = code.find(pattern)
    assert index != -1, f"Pattern '{pattern}' not found in code."
    return code[:index] + addition + code[index:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update",
        action="store_true",
        help=(
            "Whether to re-generate `./src/huggingface_hub/inference/_generated/_async_client.py` if a change is"
            " detected."
        ),
    )
    args = parser.parse_args()

    check_async_client(update=args.update)
