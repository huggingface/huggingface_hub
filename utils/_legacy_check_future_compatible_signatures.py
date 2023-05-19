# coding=utf-8
# Copyright 2022-present, the HuggingFace Inc. team.
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
"""Contains a tool to add/check the definition of "async" methods of `HfApi` in `huggingface_hub.hf_api.py`.

WARNING: this is a script kept to help with `@future_compatible` methods of `HfApi` but it is not 100% correct.
Keeping it here for reference but it is not used in the CI/Makefile.

What is done correctly:
1. Add "as_future" as argument to the method signature
2. Set Union[T, Future[T]] as return type to the method signature
3. Document "as_future" argument in the docstring of the method

What is NOT done correctly:
1. Generated stubs are grouped at the top of the `HfApi` class. They must be copy-pasted (overload definition must be
just before the method implementation)
2. `#type: ignore` must be adjusted in the first stub (if multiline definition)
"""
import argparse
import inspect
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, NoReturn

import black
from ruff.__main__ import find_ruff_bin

from huggingface_hub.hf_api import HfApi


STUBS_SECTION_TEMPLATE = """
    ### Stubs section start ###

    # This section contains stubs for the methods that are marked as `@future_compatible`. Those methods have a
    # different return type depending on the `as_future: bool` value. For better integrations with IDEs, we provide
    # stubs for both return types. The actual implementation of those methods is written below.

    # WARNING: this section have been generated automatically. Do not modify it manually. If you modify it manually, your
    # changes will be overwritten. To re-generate this section, run `make style` (or `python utils/check_future_compatible_signatures.py`
    # directly).

    # FAQ:
    # 1. Why should we have these? For better type annotation which helps with IDE features like autocompletion.
    # 2. Why not a separate `hf_api.pyi` file? Would require to re-defined all the existing annotations from `hf_api.py`.
    # 3. Why not at the end of the module? Because `@overload` methods must be defined first.
    # 4. Why not another solution? I'd be glad, but this is the "less worse" I could find.
    # For more details, see https://github.com/huggingface/huggingface_hub/pull/1458


    {stubs}

    # WARNING: this section have been generated automatically. Do not modify it manually. If you modify it manually, your
    # changes will be overwritten. To re-generate this section, run `make style` (or `python utils/check_future_compatible_signatures.py`
    # directly).

    ### Stubs section end ###
"""

STUBS_SECTION_TEMPLATE_REGEX = re.compile(r"### Stubs section start ###.*### Stubs section end ###", re.DOTALL)

AS_FUTURE_SIGNATURE_TEMPLATE = "as_future: bool = False"

AS_FUTURE_DOCSTRING_TEMPLATE = """
            as_future (`bool`, *optional*):
                Whether or not to run this method in the background. Background jobs are run sequentially without
                blocking the main thread. Passing `as_future=True` will return a [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects)
                object. Defaults to `False`."""

ARGS_DOCSTRING_REGEX = re.compile(
    """
^[ ]{8}Args:   # Match args section ...
(.*?)          # ... everything ...
^[ ]{8}\\S      # ... until next section or end of docstring
""",
    re.MULTILINE | re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

SIGNATURE_REGEX_FULL = re.compile(r"^\s*def.*?-> (.*?):", re.DOTALL | re.MULTILINE)
SIGNATURE_REGEX_RETURN_TYPE = re.compile(r"-> (.*?):")
SIGNATURE_REGEX_RETURN_TYPE_WITH_FUTURE = re.compile(r"-> Union\[(.*?), (.*?)\]:")


HF_API_FILE_PATH = Path(__file__).parents[1] / "src" / "huggingface_hub" / "hf_api.py"
HF_API_FILE_CONTENT = HF_API_FILE_PATH.read_text()


def generate_future_compatible_method(method: Callable, method_source: str) -> str:
    # 1. Document `as_future` parameter
    if AS_FUTURE_DOCSTRING_TEMPLATE not in method_source:
        match = ARGS_DOCSTRING_REGEX.search(method_source)
        if match is None:
            raise ValueError(f"Could not find `Args` section in docstring of {method}.")
        args_docs = match.group(1).strip()
        method_source = method_source.replace(args_docs, args_docs + AS_FUTURE_DOCSTRING_TEMPLATE)

    # 2. Update signature
    # 2.a. Add `as_future` parameter
    if AS_FUTURE_SIGNATURE_TEMPLATE not in method_source:
        match = SIGNATURE_REGEX_FULL.search(method_source)
        if match is None:
            raise ValueError(f"Could not find signature of {method} in source.")
        method_source = method_source.replace(
            match.group(), match.group().replace(") ->", f" {AS_FUTURE_SIGNATURE_TEMPLATE}) ->"), 1
        )

    # 2.b. Update return value
    if "Future[" not in method_source:
        match = SIGNATURE_REGEX_RETURN_TYPE.search(method_source)
        if match is None:
            raise ValueError(f"Could not find return type of {method} in source.")
        base_type = match.group(1).strip()
        return_type = f"Union[{base_type}, Future[{base_type}]]"
        return_value_replaced = match.group().replace(match.group(1), return_type)
        method_source = method_source.replace(match.group(), return_value_replaced)

    # 3. Generate @overload stubs
    match = SIGNATURE_REGEX_FULL.search(method_source)
    if match is None:
        raise ValueError(f"Could not find signature of {method} in source.")
    method_sig = match.group()

    match = SIGNATURE_REGEX_RETURN_TYPE_WITH_FUTURE.search(method_sig)
    if match is None:
        raise ValueError(f"Could not find return type (with Future) of {method} in source.")
    no_future_return_type = match.group(1).strip()
    with_future_return_type = match.group(2).strip()

    # 3.a. Stub when `as_future=False`
    no_future_stub = "    @overload\n" + method_sig
    no_future_stub = no_future_stub.replace(AS_FUTURE_SIGNATURE_TEMPLATE, "as_future: Literal[False] = ...")
    no_future_stub = SIGNATURE_REGEX_RETURN_TYPE.sub(rf"-> {no_future_return_type}:", no_future_stub)
    no_future_stub += "  # type: ignore\n        ..."  # only the first stub requires "type: ignore"

    # 3.b. Stub when `as_future=True`
    with_future_stub = "    @overload\n" + method_sig
    with_future_stub = with_future_stub.replace(AS_FUTURE_SIGNATURE_TEMPLATE, "as_future: Literal[True] = ...")
    with_future_stub = SIGNATURE_REGEX_RETURN_TYPE.sub(rf"-> {with_future_return_type}:", with_future_stub)
    with_future_stub += "\n        ..."

    stubs_source = no_future_stub + "\n\n" + with_future_stub + "\n\n"

    # 4. All good!
    return method_source, stubs_source


def generate_hf_api_module() -> str:
    raw_code = HF_API_FILE_CONTENT

    # Process all Future-compatible methods
    all_stubs_source = ""
    for _, method in inspect.getmembers(HfApi, predicate=inspect.isfunction):
        if not getattr(method, "is_future_compatible", False):
            continue
        source = inspect.getsource(method)
        method_source, stubs_source = generate_future_compatible_method(method, source)

        raw_code = raw_code.replace(source, method_source)
        all_stubs_source += "\n\n" + stubs_source

    # Generate code with stubs
    generated_code = STUBS_SECTION_TEMPLATE_REGEX.sub(STUBS_SECTION_TEMPLATE.format(stubs=all_stubs_source), raw_code)

    # Format (black+ruff)
    return format_generated_code(generated_code)


def format_generated_code(code: str) -> str:
    """
    Format some code with black+ruff. Cannot be done "on the fly" so we first save the code in a temporary file.
    """
    # Format with black
    code = black.format_file_contents(code, fast=False, mode=black.FileMode(line_length=119))

    # Format with ruff
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "__init__.py"
        filepath.write_text(code)
        ruff_bin = find_ruff_bin()
        os.spawnv(os.P_WAIT, ruff_bin, ["ruff", str(filepath), "--fix", "--quiet"])
        return filepath.read_text()


def check_future_compatible_hf_api(update: bool) -> NoReturn:
    """Check that the code defining the threaded version of HfApi is up-to-date."""
    # If expected `__init__.py` content is different, test fails. If '--update-init-file'
    # is used, `__init__.py` file is updated before the test fails.
    expected_content = generate_hf_api_module()
    if expected_content != HF_API_FILE_CONTENT:
        if update:
            with HF_API_FILE_PATH.open("w") as f:
                f.write(expected_content)

            print(
                "✅ Signature/docstring/annotations for Future-compatible methods have been updated in"
                " `./src/huggingface_hub/hf_api.py`.\n   Please make sure the changes are accurate and commit them."
            )
            exit(0)
        else:
            print(
                "❌ Expected content mismatch for Future compatible methods in `./src/huggingface_hub/hf_api.py`.\n  "
                " Please run `make style` or `python utils/check_future_compatible_signatures.py --update`."
            )
            exit(1)

    print("✅ All good! (Future-compatible methods)")
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update",
        action="store_true",
        help="Whether to override `./src/huggingface_hub/hf_api.py` if a change is detected.",
    )
    args = parser.parse_args()

    check_future_compatible_hf_api(update=args.update)
