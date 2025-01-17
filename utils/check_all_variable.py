# coding=utf-8
# Copyright 2025-present, the HuggingFace Inc. team.
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

"""Script to check and update the __all__ variable for huggingface_hub/__init__.py."""

import argparse
import re
from pathlib import Path
from typing import Dict, List, NoReturn

from huggingface_hub import _SUBMOD_ATTRS


INIT_FILE_PATH = Path(__file__).parents[1] / "src" / "huggingface_hub" / "__init__.py"


def format_all_definition(submod_attrs: Dict[str, List[str]]) -> str:
    """
    Generate a formatted static __all__ definition with grouped comments.
    """
    all_attrs = sorted(attr for attrs in submod_attrs.values() for attr in attrs)

    lines = ["__all__ = ["]
    lines.extend(f'    "{attr}",' for attr in all_attrs)
    lines.append("]")

    return "\n".join(lines)


def parse_all_definition(content: str) -> List[str]:
    """
    Extract the current __all__ contents from file content.

    This is prefered over "from huggingface_hub import __all__ as current_items" to handle
    case where __all__ is not defined or malformed in the file we want to be able to fix
    such issues rather than crash also, we are interested in the file content.
    """
    match = re.search(r"__all__\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not match:
        return []

    # Extract items while preserving order, properly cleaning whitespace and quotes
    return [
        line.strip().strip("\",'")
        for line in match.group(1).split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]


def check_static_all(update: bool) -> NoReturn:
    """Check if __all__ is aligned with _SUBMOD_ATTRS or update it."""
    content = INIT_FILE_PATH.read_text()
    new_all = format_all_definition(_SUBMOD_ATTRS)

    expected_items = sorted(attr for attrs in _SUBMOD_ATTRS.values() for attr in attrs)

    current_items = list(parse_all_definition(content))

    if current_items == expected_items:
        print("✅ All good! the __all__ variable is up to date")
        exit(0)

    if update:
        all_pattern = re.compile(r"__all__\s*=\s*\[[^\]]*\]", re.MULTILINE | re.DOTALL)
        if all_pattern.search(content):
            new_content = all_pattern.sub(new_all, content)
        else:
            submod_attrs_pattern = re.compile(r"_SUBMOD_ATTRS\s*=\s*{[^}]*}", re.MULTILINE | re.DOTALL)
            match = submod_attrs_pattern.search(content)
            if not match:
                print("Error: _SUBMOD_ATTRS dictionary not found in `./src/huggingface_hub/__init__.py`.")
                exit(1)

            dict_end = match.end()
            new_content = content[:dict_end] + "\n\n\n" + new_all + "\n\n" + content[dict_end:]

        INIT_FILE_PATH.write_text(new_content)
        print(
            "✅ __all__ variable has been updated in `./src/huggingface_hub/__init__.py`."
            "\n   Please make sure the changes are accurate and commit them."
        )
        exit(0)
    else:
        print(
            "❌ Expected content mismatch in"
            " `./src/huggingface_hub/__init__.py`.\n   It is most likely that"
            " a module was added to the `_SUBMOD_ATTRS` mapping and did not update the"
            " '__all__' variable.\n   Please run `make style` or `python"
            " utils/check_all_variable.py --update`."
        )
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update",
        action="store_true",
        help="Whether to fix `./src/huggingface_hub/__init__.py` if a change is detected.",
    )
    args = parser.parse_args()

    check_static_all(update=args.update)
