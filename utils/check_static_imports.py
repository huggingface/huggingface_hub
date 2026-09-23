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
"""Reformat and validate lazy and static imports in package ``__init__.py`` files."""

import argparse
import os
import re
import tempfile
from pathlib import Path

from helpers import read_literal_assignment
from ruff.__main__ import find_ruff_bin


REPO_ROOT = Path(__file__).parents[1]
ROOT_INIT_PATH = REPO_ROOT / "src" / "huggingface_hub" / "__init__.py"
UTILS_INIT_PATH = REPO_ROOT / "src" / "huggingface_hub" / "utils" / "__init__.py"

IF_TYPE_CHECKING_LINE = "\nif TYPE_CHECKING:  # pragma: no cover\n"
SUBMOD_ATTRS_PATTERN = re.compile("_SUBMOD_ATTRS = {[^}]+}")
SUBMODULES_PATTERN = re.compile(r"_SUBMODULES = \{[^}]+\}")


def _format_mapping(mapping: dict[str, list[str]]) -> str:
    return (
        "_SUBMOD_ATTRS = {\n"
        + "\n".join(
            f'    "{module}": [\n'
            + "\n".join(f'        "{attr}",' for attr in sorted(set(mapping[module])))
            + "\n    ],"
            for module in sorted(mapping)
        )
        + "\n}"
    )


def _format_set(values: set[str]) -> str:
    return "_SUBMODULES = {\n" + "\n".join(f'    "{value}",' for value in sorted(values)) + "\n}"


def _format_with_ruff(content: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "__init__.py"
        filepath.write_text(content)
        ruff_bin = find_ruff_bin()
        os.spawnv(os.P_WAIT, ruff_bin, ["ruff", "check", str(filepath), "--fix", "--quiet"])
        os.spawnv(os.P_WAIT, ruff_bin, ["ruff", "format", str(filepath), "--quiet"])
        return filepath.read_text()


def _generate_root_init(content: str) -> str:
    root_submod_attrs = read_literal_assignment(content, "_SUBMOD_ATTRS")
    content_before_static_checks = content.split(IF_TYPE_CHECKING_LINE)[0]
    if SUBMOD_ATTRS_PATTERN.search(content_before_static_checks) is None:
        raise ValueError(f"_SUBMOD_ATTRS dictionary not found in {ROOT_INIT_PATH}")

    content_before_static_checks = SUBMOD_ATTRS_PATTERN.sub(
        _format_mapping(root_submod_attrs), content_before_static_checks
    )
    static_imports = [
        f"    from .{module} import {attr}  # noqa: F401"
        for module, attributes in root_submod_attrs.items()
        for attr in attributes
    ]
    return _format_with_ruff(content_before_static_checks + IF_TYPE_CHECKING_LINE + "\n".join(static_imports) + "\n")


def _generate_utils_init(content: str) -> str:
    utils_submod_attrs = read_literal_assignment(content, "_SUBMOD_ATTRS")
    utils_submodules = read_literal_assignment(content, "_SUBMODULES")
    content_before_static_checks = content.split(IF_TYPE_CHECKING_LINE)[0]
    if SUBMOD_ATTRS_PATTERN.search(content_before_static_checks) is None:
        raise ValueError(f"_SUBMOD_ATTRS dictionary not found in {UTILS_INIT_PATH}")
    if SUBMODULES_PATTERN.search(content_before_static_checks) is None:
        raise ValueError(f"_SUBMODULES set not found in {UTILS_INIT_PATH}")

    content_before_static_checks = SUBMODULES_PATTERN.sub(_format_set(utils_submodules), content_before_static_checks)
    content_before_static_checks = SUBMOD_ATTRS_PATTERN.sub(
        _format_mapping(utils_submod_attrs), content_before_static_checks
    )

    static_imports = ["    import httpx as httpx  # noqa: F401"]
    for module in sorted(utils_submodules):
        imported_module = "tqdm" if module == "_tqdm" else module
        static_imports.append(f"    from . import {imported_module} as {module}  # noqa: F401")
    for module, attributes in utils_submod_attrs.items():
        module_path = module if module.startswith("huggingface_hub.") else f".{module}"
        static_imports.extend(f"    from {module_path} import {attr}  # noqa: F401" for attr in attributes)

    return _format_with_ruff(content_before_static_checks + IF_TYPE_CHECKING_LINE + "\n".join(static_imports) + "\n")


def check_static_imports(update: bool) -> bool:
    """Check that lazy definitions and static imports agree in both package facades."""
    files = {
        ROOT_INIT_PATH: _generate_root_init,
        UTILS_INIT_PATH: _generate_utils_init,
    }
    mismatches = []
    for path, generate in files.items():
        content = path.read_text()
        expected_content = generate(content)
        if content == expected_content:
            continue
        mismatches.append(path)
        if update:
            path.write_text(expected_content)

    if not mismatches:
        print("✅ All good! (static imports)")
        return True

    relative_paths = ", ".join(str(path.relative_to(REPO_ROOT)) for path in mismatches)
    if update:
        print(f"✅ Imports have been updated in {relative_paths}.\n   Please review and commit the changes.")
        return True

    print(
        f"❌ Static imports do not match lazy imports in {relative_paths}.\n"
        "   Run `make style` or `python utils/check_static_imports.py --update`."
    )
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update",
        action="store_true",
        help="Whether to update package __init__.py files when their imports do not match.",
    )
    args = parser.parse_args()
    raise SystemExit(0 if check_static_imports(update=args.update) else 1)
