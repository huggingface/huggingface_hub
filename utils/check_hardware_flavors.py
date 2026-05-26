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
"""Check and update SpaceHardware / JobHardware enums against the Hub API.

Fetches the current hardware flavors from the Hub API (list_spaces_hardware,
list_jobs_hardware) and compares them to the hardcoded enums. New flavors are
added; removed flavors are kept for backward compat and marked with ``# legacy``.

Usage:
    python utils/check_hardware_flavors.py           # check only (CI)
    python utils/check_hardware_flavors.py --update   # update source files
"""

import argparse
import re
import sys
from pathlib import Path
from typing import NoReturn

from huggingface_hub.hf_api import HfApi


SPACE_API_PATH = Path(__file__).parents[1] / "src" / "huggingface_hub" / "_space_api.py"
JOBS_API_PATH = Path(__file__).parents[1] / "src" / "huggingface_hub" / "_jobs_api.py"


def _make_enum_name(value: str) -> str:
    """Convert a flavor slug like 'cpu-basic' to an enum member name like 'CPU_BASIC'."""
    return value.upper().replace("-", "_")


class _EnumEntry:
    """An entry in an enum body: either a member or a comment/blank line."""

    __slots__ = ("kind", "name", "value", "is_legacy", "text")

    def __init__(
        self,
        kind: str,
        name: str = "",
        value: str = "",
        is_legacy: bool = False,
        text: str = "",
    ):
        self.kind = kind  # "member" or "line"
        self.name = name
        self.value = value
        self.is_legacy = is_legacy
        self.text = text  # raw line text for comments/blanks


def _parse_enum_entries(source: str, class_name: str) -> list[_EnumEntry]:
    """Parse enum body from source, preserving members, section comments, and blank lines."""
    pattern = re.compile(
        rf"class {re.escape(class_name)}\(str, Enum\):(.*?)(?=\nclass |\ndef |\n@dataclass)",
        re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        print(f"Error: could not find class {class_name} in source.")
        sys.exit(1)

    body = match.group(1)
    # Skip the docstring: find closing """
    doc_end = body.find('"""', body.find('"""') + 3)
    if doc_end == -1:
        print(f"Error: could not find docstring end for class {class_name}.")
        sys.exit(1)
    body_after_doc = body[doc_end + 3 :]

    entries: list[_EnumEntry] = []
    for line in body_after_doc.splitlines():
        m = re.match(r"^\s{4}([A-Z_][A-Z_0-9]*)\s*=\s*\"([^\"]+)\"(.*)$", line)
        if m:
            name, value, rest = m.group(1), m.group(2), m.group(3)
            is_legacy = "# legacy" in rest
            entries.append(_EnumEntry("member", name=name, value=value, is_legacy=is_legacy))
        elif line.strip():
            entries.append(_EnumEntry("line", text=line))
        else:
            entries.append(_EnumEntry("line", text=""))

    # Strip leading blank lines (we always emit one blank line after the docstring).
    while entries and entries[0].kind == "line" and not entries[0].text.strip():
        entries.pop(0)

    return entries


def _parse_enum_members(source: str, class_name: str) -> list[tuple[str, str, bool]]:
    """Parse enum members from source, returning (name, value, is_legacy) tuples in order."""
    entries = _parse_enum_entries(source, class_name)
    return [(e.name, e.value, e.is_legacy) for e in entries if e.kind == "member"]


def _build_enum_lines(
    existing_entries: list[_EnumEntry],
    api_values: list[str],
) -> list[str]:
    """Build the list of enum body lines, preserving order, section comments, and blank lines.

    - Existing members keep their position and surrounding comments.
    - Members no longer in the API get ``# legacy`` if not already marked.
    - New API values are appended at the end.
    """
    api_set = set(api_values)
    existing_values = {e.value for e in existing_entries if e.kind == "member"}

    lines: list[str] = []
    for entry in existing_entries:
        if entry.kind == "line":
            lines.append(entry.text)
        else:
            name, value, was_legacy = entry.name, entry.value, entry.is_legacy
            if value not in api_set and not was_legacy:
                lines.append(f'    {name} = "{value}"  # legacy')
            elif was_legacy and value in api_set:
                lines.append(f'    {name} = "{value}"')
            else:
                suffix = "  # legacy" if was_legacy else ""
                lines.append(f'    {name} = "{value}"{suffix}')

    # Append new values not already in the enum.
    new_values = [v for v in api_values if v not in existing_values]
    if new_values:
        lines.append("")
        lines.append("    # New (auto-added by utils/check_hardware_flavors.py)")
        for value in new_values:
            lines.append(f'    {_make_enum_name(value)} = "{value}"')

    return lines


def _replace_enum_body(source: str, class_name: str, new_lines: list[str]) -> str:
    """Replace the member lines of an enum class in source code.

    Keeps everything up to and including the closing docstring, then replaces the
    enum member definitions up to the next top-level class/def/decorator.
    """
    # Step 1: find the enum class and its docstring.
    class_pattern = re.compile(rf"^class {re.escape(class_name)}\(str, Enum\):", re.MULTILINE)
    class_match = class_pattern.search(source)
    if not class_match:
        print(f"Error: could not find class {class_name}.")
        sys.exit(1)

    # Find the closing `"""` of the docstring (third occurrence of `"""` after class start).
    pos = class_match.start()
    doc_open = source.index('"""', pos)
    doc_close = source.index('"""', doc_open + 3) + 3
    header = source[:doc_close]

    # Step 2: find where the enum body ends (next top-level class/def/decorator).
    end_pattern = re.compile(r"^\n(?=class |def |@dataclass)", re.MULTILINE)
    end_match = end_pattern.search(source, doc_close)
    if not end_match:
        # Enum is at the end of file
        body_end = len(source)
    else:
        body_end = end_match.start()

    new_body = "\n".join(["", "", *new_lines, ""])
    return header + new_body + source[body_end:]


def check_hardware_flavors(update: bool) -> NoReturn:
    api = HfApi()

    # Fetch current hardware from the API.
    print("Fetching hardware flavors from the Hub API...")
    spaces_hardware = api.list_spaces_hardware()
    jobs_hardware = api.list_jobs_hardware()
    spaces_api_values = [hw.name for hw in spaces_hardware]
    jobs_api_values = [hw.name for hw in jobs_hardware]
    print(f"  Spaces: {len(spaces_api_values)} flavors")
    print(f"  Jobs:   {len(jobs_api_values)} flavors")

    # Read source files.
    space_source = SPACE_API_PATH.read_text()
    jobs_source = JOBS_API_PATH.read_text()

    # Parse existing enums.
    space_entries = _parse_enum_entries(space_source, "SpaceHardware")
    job_entries = _parse_enum_entries(jobs_source, "JobHardware")
    space_existing = {e.value for e in space_entries if e.kind == "member"}
    job_existing = {e.value for e in job_entries if e.kind == "member"}

    # Compute diffs.
    spaces_new = [v for v in spaces_api_values if v not in space_existing]
    jobs_new = [v for v in jobs_api_values if v not in job_existing]
    spaces_removed = [
        e.value
        for e in space_entries
        if e.kind == "member" and e.value not in set(spaces_api_values) and not e.is_legacy
    ]
    jobs_removed = [
        e.value for e in job_entries if e.kind == "member" and e.value not in set(jobs_api_values) and not e.is_legacy
    ]

    has_changes = bool(spaces_new or jobs_new or spaces_removed or jobs_removed)

    if spaces_new:
        print(f"\n  SpaceHardware — new:     {spaces_new}")
    if spaces_removed:
        print(f"  SpaceHardware — removed: {spaces_removed} (will be marked as legacy)")
    if jobs_new:
        print(f"  JobHardware — new:     {jobs_new}")
    if jobs_removed:
        print(f"  JobHardware — removed: {jobs_removed} (will be marked as legacy)")

    if not has_changes:
        print("\n✅ All good! Hardware enums are up to date.")
        sys.exit(0)

    if not update:
        print("\n❌ Hardware enums are out of date.\n   Please run `python utils/check_hardware_flavors.py --update`.")
        sys.exit(1)

    # Update files.
    new_space_lines = _build_enum_lines(space_entries, spaces_api_values)
    new_job_lines = _build_enum_lines(job_entries, jobs_api_values)

    new_space_source = _replace_enum_body(space_source, "SpaceHardware", new_space_lines)
    new_jobs_source = _replace_enum_body(jobs_source, "JobHardware", new_job_lines)

    SPACE_API_PATH.write_text(new_space_source)
    JOBS_API_PATH.write_text(new_jobs_source)

    print("\n✅ Hardware enums updated.\n   Please review the changes and commit them.")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check/update hardware flavor enums against the Hub API.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update the enum source files if changes are detected.",
    )
    args = parser.parse_args()
    check_hardware_flavors(update=args.update)
