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

    # Add _get_client_session
    code = _add_get_client_session(code)

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
            + "from .._common import _async_yield_from, _import_aiohttp\n"
            + "from typing import AsyncIterable\n"
            + "from typing import Set\n"
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


ASYNC_INNER_POST_CODE = """
        aiohttp = _import_aiohttp()

        # TODO: this should be handled in provider helpers directly
        if request_parameters.task in TASKS_EXPECTING_IMAGES and "Accept" not in request_parameters.headers:
            request_parameters.headers["Accept"] = "image/png"

        while True:
            with _open_as_binary(request_parameters.data) as data_as_binary:
                # Do not use context manager as we don't want to close the connection immediately when returning
                # a stream
                session = self._get_client_session(headers=request_parameters.headers)

                try:
                    response = await session.post(request_parameters.url, json=request_parameters.json, data=data_as_binary, proxy=self.proxies)
                    response_error_payload = None
                    if response.status != 200:
                        try:
                            response_error_payload = await response.json()  # get payload before connection closed
                        except Exception:
                            pass
                    response.raise_for_status()
                    if stream:
                        return _async_yield_from(session, response)
                    else:
                        content = await response.read()
                        await session.close()
                        return content
                except asyncio.TimeoutError as error:
                    await session.close()
                    # Convert any `TimeoutError` to a `InferenceTimeoutError`
                    raise InferenceTimeoutError(f"Inference call timed out: {request_parameters.url}") from error  # type: ignore
                except aiohttp.ClientResponseError as error:
                    error.response_error_payload = response_error_payload
                    await session.close()
                    raise error
                except Exception:
                    await session.close()
                    raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    def __del__(self):
        if len(self._sessions) > 0:
            warnings.warn(
                "Deleting 'AsyncInferenceClient' client but some sessions are still open. "
                "This can happen if you've stopped streaming data from the server before the stream was complete. "
                "To close the client properly, you must call `await client.close()` "
                "or use an async context (e.g. `async with AsyncInferenceClient(): ...`."
            )

    async def close(self):
        \"""Close all open sessions.

        By default, 'aiohttp.ClientSession' objects are closed automatically when a call is completed. However, if you
        are streaming data from the server and you stop before the stream is complete, you must call this method to
        close the session properly.

        Another possibility is to use an async context (e.g. `async with AsyncInferenceClient(): ...`).
        \"""
        await asyncio.gather(*[session.close() for session in self._sessions.keys()])"""


def _make_inner_post_async(code: str) -> str:
    # Update AsyncInferenceClient._inner_post() implementation (use aiohttp instead of requests)
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
    return code.replace("Iterable[bytes]", "AsyncIterable[bytes]")


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


def _adapt_get_model_status(code: str) -> str:
    sync_snippet = """
        response = get_session().get(url, headers=build_hf_headers(token=self.token))
        hf_raise_for_status(response)
        response_data = response.json()"""

    async_snippet = """
        async with self._get_client_session(headers=build_hf_headers(token=self.token)) as client:
            response = await client.get(url, proxy=self.proxies)
            response.raise_for_status()
            response_data = await response.json()"""

    return code.replace(sync_snippet, async_snippet)


def _adapt_list_deployed_models(code: str) -> str:
    sync_snippet = """
        for framework in frameworks:
            response = get_session().get(f"{INFERENCE_ENDPOINT}/framework/{framework}", headers=build_hf_headers(token=self.token))
            hf_raise_for_status(response)
            _unpack_response(framework, response.json())""".strip()

    async_snippet = """
        async def _fetch_framework(framework: str) -> None:
            async with self._get_client_session(headers=build_hf_headers(token=self.token)) as client:
                response = await client.get(f"{INFERENCE_ENDPOINT}/framework/{framework}", proxy=self.proxies)
                response.raise_for_status()
                _unpack_response(framework, await response.json())

        import asyncio

        await asyncio.gather(*[_fetch_framework(framework) for framework in frameworks])""".strip()

    return code.replace(sync_snippet, async_snippet)


def _adapt_info_and_health_endpoints(code: str) -> str:
    info_sync_snippet = """
        response = get_session().get(url, headers=build_hf_headers(token=self.token))
        hf_raise_for_status(response)
        return response.json()"""

    info_async_snippet = """
        async with self._get_client_session(headers=build_hf_headers(token=self.token)) as client:
            response = await client.get(url, proxy=self.proxies)
            response.raise_for_status()
            return await response.json()"""

    code = code.replace(info_sync_snippet, info_async_snippet)

    health_sync_snippet = """
        response = get_session().get(url, headers=build_hf_headers(token=self.token))
        return response.status_code == 200"""

    health_async_snippet = """
        async with self._get_client_session(headers=build_hf_headers(token=self.token)) as client:
            response = await client.get(url, proxy=self.proxies)
            return response.status == 200"""

    return code.replace(health_sync_snippet, health_async_snippet)


def _add_get_client_session(code: str) -> str:
    # Add trust_env as parameter
    code = _add_before(code, "proxies: Optional[Any] = None,", "trust_env: bool = False,")
    code = _add_before(code, "\n        self.proxies = proxies\n", "\n        self.trust_env = trust_env")

    # Document `trust_env` parameter
    code = _add_before(
        code,
        "\n        proxies (`Any`, `optional`):",
        """
        trust_env ('bool', 'optional'):
            Trust environment settings for proxy configuration if the parameter is `True` (`False` by default).""",
    )

    # insert `_get_client_session` before `get_endpoint_info` method
    client_session_code = """

    def _get_client_session(self, headers: Optional[Dict] = None) -> "ClientSession":
        aiohttp = _import_aiohttp()
        client_headers = self.headers.copy()
        if headers is not None:
            client_headers.update(headers)

        # Return a new aiohttp ClientSession with correct settings.
        session = aiohttp.ClientSession(
            headers=client_headers,
            cookies=self.cookies,
            timeout=aiohttp.ClientTimeout(self.timeout),
            trust_env=self.trust_env,
        )

        # Keep track of sessions to close them later
        self._sessions[session] = set()

        # Override the `._request` method to register responses to be closed
        session._wrapped_request = session._request

        async def _request(method, url, **kwargs):
            response = await session._wrapped_request(method, url, **kwargs)
            self._sessions[session].add(response)
            return response

        session._request = _request

        # Override the 'close' method to
        # 1. close ongoing responses
        # 2. deregister the session when closed
        session._close = session.close

        async def close_session():
            for response in self._sessions[session]:
                response.close()
            await session._close()
            self._sessions.pop(session, None)

        session.close = close_session
        return session

"""
    code = _add_before(code, "\n    async def get_endpoint_info(", client_session_code)

    # Add self._sessions attribute in __init__
    code = _add_before(
        code,
        "\n    def __repr__(self):\n",
        "\n        # Keep track of the sessions to close them properly"
        "\n        self._sessions: Dict['ClientSession', Set['ClientResponse']] = dict()",
    )

    return code


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
