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
import os
import re
import tempfile
from pathlib import Path
from typing import NoReturn

from ruff.__main__ import find_ruff_bin


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
    code = _make_post_async(code)
    code = _await_post_method_call(code)
    code = _use_async_streaming_util(code)

    # Make all tasks-method async
    code = _make_tasks_methods_async(code)

    # Adapt text_generation to async
    code = _adapt_text_generation_to_async(code)

    # Adapt chat_completion to async
    code = _adapt_chat_completion_to_async(code)

    # Update some docstrings
    code = _rename_HTTPError_to_ClientResponseError_in_docstring(code)
    code = _update_examples_in_public_methods(code)

    # Adapt get_model_status
    code = _adapt_get_model_status(code)

    # Adapt list_deployed_models
    code = _adapt_list_deployed_models(code)

    # Adapt /info and /health endpoints
    code = _adapt_info_and_health_endpoints(code)

    return code


def format_source_code(code: str) -> str:
    """Apply formatter on a generated source code."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "async_client.py"
        filepath.write_text(code)
        ruff_bin = find_ruff_bin()
        os.spawnv(os.P_WAIT, ruff_bin, ["ruff", str(filepath), "--fix", "--quiet"])
        os.spawnv(os.P_WAIT, ruff_bin, ["ruff", "format", str(filepath), "--quiet"])
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
                " `./src/huggingface_hub/inference/_generated/_async_client.py`.\n   Please make sure the changes are"
                " accurate and commit them."
            )
            exit(0)
        else:
            print(
                "❌ Expected content mismatch in `./src/huggingface_hub/inference/_generated/_async_client.py`.\n   It"
                " is most likely that you modified some InferenceClient code and did not update the the"
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
            + "from .._common import _async_yield_from, _import_aiohttp\n"
            + "from typing import AsyncIterable\n"
            + "import asyncio\n"
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


ASYNC_POST_CODE = """
        aiohttp = _import_aiohttp()

        url = self._resolve_url(model, task)

        if data is not None and json is not None:
            warnings.warn("Ignoring `json` as `data` is passed as binary.")

        # Set Accept header if relevant
        headers = self.headers.copy()
        if task in TASKS_EXPECTING_IMAGES and "Accept" not in headers:
            headers["Accept"] = "image/png"

        t0 = time.time()
        timeout = self.timeout
        while True:
            with _open_as_binary(data) as data_as_binary:
                # Do not use context manager as we don't want to close the connection immediately when returning
                # a stream
                client = aiohttp.ClientSession(
                    headers=headers, cookies=self.cookies, timeout=aiohttp.ClientTimeout(self.timeout)
                )

                try:
                    response = await client.post(url, json=json, data=data_as_binary)
                    response_error_payload = None
                    if response.status != 200:
                        try:
                            response_error_payload = await response.json()  # get payload before connection closed
                        except Exception:
                            pass
                    response.raise_for_status()
                    if stream:
                        return _async_yield_from(client, response)
                    else:
                        content = await response.read()
                        await client.close()
                        return content
                except asyncio.TimeoutError as error:
                    await client.close()
                    # Convert any `TimeoutError` to a `InferenceTimeoutError`
                    raise InferenceTimeoutError(f"Inference call timed out: {url}") from error  # type: ignore
                except aiohttp.ClientResponseError as error:
                    error.response_error_payload = response_error_payload
                    await client.close()
                    if response.status == 422 and task is not None:
                        error.message += f". Make sure '{task}' task is supported by the model."
                    if response.status == 503:
                        # If Model is unavailable, either raise a TimeoutError...
                        if timeout is not None and time.time() - t0 > timeout:
                            raise InferenceTimeoutError(
                                f"Model not loaded on the server: {url}. Please retry with a higher timeout"
                                f" (current: {self.timeout}).",
                                request=error.request,
                                response=error.response,
                            ) from error
                        # ...or wait 1s and retry
                        logger.info(f"Waiting for model to be loaded on the server: {error}")
                        time.sleep(1)
                        if timeout is not None:
                            timeout = max(self.timeout - (time.time() - t0), 1)  # type: ignore
                        continue
                    raise error
                except Exception:
                    await client.close()
                    raise"""


def _make_post_async(code: str) -> str:
    # Update AsyncInferenceClient.post() implementation (use aiohttp instead of requests)
    code = re.sub(
        r"""
        def[ ]post\( # definition
            (\n.*?\"\"\".*?\"\"\"\n) # Group1: docstring
            .*? # implementation (to be overwritten)
        (\n\W*def ) # Group2: next method
        """,
        repl=rf"async def post(\1{ASYNC_POST_CODE}\2",
        string=code,
        count=1,
        flags=re.DOTALL | re.VERBOSE,
    )
    # Update `post`'s type annotations
    return code.replace("Iterable[bytes]", "AsyncIterable[bytes]", 2)


def _rename_HTTPError_to_ClientResponseError_in_docstring(code: str) -> str:
    # Update `raises`-part in docstrings
    return code.replace("`HTTPError`:", "`aiohttp.ClientResponseError`:")


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
    # Text-generation task has to be handled specifically since it has a recursive call mechanism (to retry on non-tgi
    # servers)

    # Catch `aiohttp` error instead of `requests` error
    code = code.replace(
        """
        except HTTPError as e:
            match = MODEL_KWARGS_NOT_USED_REGEX.search(str(e))
            if isinstance(e, BadRequestError) and match:
    """,
        """
        except _import_aiohttp().ClientResponseError as e:
            match = MODEL_KWARGS_NOT_USED_REGEX.search(e.response_error_payload["error"])
            if e.status == 400 and match:
    """,
    )

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
        ") -> Iterable[str]:",
        ") -> AsyncIterable[str]:",
    )
    code = code.replace(
        ") -> Union[bytes, Iterable[bytes]]:",
        ") -> Union[bytes, AsyncIterable[bytes]]:",
    )
    code = code.replace(
        ") -> Iterable[TextGenerationStreamOutput]:",
        ") -> AsyncIterable[TextGenerationStreamOutput]:",
    )
    code = code.replace(
        ") -> Union[TextGenerationOutput, Iterable[TextGenerationStreamOutput]]:",
        ") -> Union[TextGenerationOutput, AsyncIterable[TextGenerationStreamOutput]]:",
    )
    code = code.replace(
        ") -> Union[str, TextGenerationOutput, Iterable[str], Iterable[TextGenerationStreamOutput]]:",
        ") -> Union[str, TextGenerationOutput, AsyncIterable[str], AsyncIterable[TextGenerationStreamOutput]]:",
    )

    return code


def _adapt_chat_completion_to_async(code: str) -> str:
    # Catch `aiohttp` error instead of `requests` error
    code = code.replace(
        """            except HTTPError as e:
                if e.response.status_code in (400, 404, 500):""",
        """            except _import_aiohttp().ClientResponseError as e:
                if e.status in (400, 404, 500):""",
    )

    # Await text-generation call
    code = code.replace(
        "text_generation_output = self.text_generation(",
        "text_generation_output = await self.text_generation(",
    )

    # Update return types: Iterable -> AsyncIterable
    code = code.replace(
        ") -> Iterable[ChatCompletionStreamOutput]:",
        ") -> AsyncIterable[ChatCompletionStreamOutput]:",
    )
    code = code.replace(
        ") -> Union[ChatCompletionOutput, Iterable[ChatCompletionStreamOutput]]:",
        ") -> Union[ChatCompletionOutput, AsyncIterable[ChatCompletionStreamOutput]]:",
    )

    return code


def _await_post_method_call(code: str) -> str:
    return code.replace("self.post(", "await self.post(")


def _update_example_code_block(code_block: str) -> str:
    """Update an atomic code block example from a docstring."""
    code_block = "\n        # Must be run in an async context" + code_block
    code_block = code_block.replace("InferenceClient", "AsyncInferenceClient")
    code_block = code_block.replace("client.", "await client.")
    code_block = code_block.replace(" for ", " async for ")
    return code_block


def _update_examples_in_public_methods(code: str) -> str:
    for match in re.finditer(
        r"""
        \n\s*
        Example:\n\s* # example section
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
    code = code.replace(
        "_stream_chat_completion_response_from_bytes", "_async_stream_chat_completion_response_from_bytes"
    )
    return code


def _adapt_get_model_status(code: str) -> str:
    sync_snippet = """
        response = get_session().get(url, headers=self.headers)
        hf_raise_for_status(response)
        response_data = response.json()"""

    async_snippet = """
        async with _import_aiohttp().ClientSession(headers=self.headers) as client:
            response = await client.get(url)
            response.raise_for_status()
            response_data = await response.json()"""

    return code.replace(sync_snippet, async_snippet)


def _adapt_list_deployed_models(code: str) -> str:
    sync_snippet = """
        for framework in frameworks:
            response = get_session().get(f"{INFERENCE_ENDPOINT}/framework/{framework}", headers=self.headers)
            hf_raise_for_status(response)
            _unpack_response(framework, response.json())""".strip()

    async_snippet = """
        async def _fetch_framework(framework: str) -> None:
            async with _import_aiohttp().ClientSession(headers=self.headers) as client:
                response = await client.get(f"{INFERENCE_ENDPOINT}/framework/{framework}")
                response.raise_for_status()
                _unpack_response(framework, await response.json())

        import asyncio

        await asyncio.gather(*[_fetch_framework(framework) for framework in frameworks])""".strip()

    return code.replace(sync_snippet, async_snippet)


def _adapt_info_and_health_endpoints(code: str) -> str:
    info_sync_snippet = """
        response = get_session().get(url, headers=self.headers)
        hf_raise_for_status(response)
        return response.json()"""

    info_async_snippet = """
        async with _import_aiohttp().ClientSession(headers=self.headers) as client:
            response = await client.get(url)
            response.raise_for_status()
            return await response.json()"""

    code = code.replace(info_sync_snippet, info_async_snippet)

    health_sync_snippet = """
        response = get_session().get(url, headers=self.headers)
        return response.status_code == 200"""

    health_async_snippet = """
        async with _import_aiohttp().ClientSession(headers=self.headers) as client:
            response = await client.get(url)
            return response.status == 200"""

    return code.replace(health_sync_snippet, health_async_snippet)


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
