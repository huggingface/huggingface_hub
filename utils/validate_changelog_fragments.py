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
"""Validates changelog fragment syntax for CI integration.

Usage:
    # Validate all fragments
    python utils/validate_changelog_fragments.py

    # Validate specific files
    python utils/validate_changelog_fragments.py .changelog/123.md .changelog/456.md

Exit codes:
    0: All fragments valid
    1: Errors found
"""

import argparse
import re
import sys
from pathlib import Path

import yaml


# Same regex as in huggingface_hub/repocard.py
REGEX_YAML_BLOCK = re.compile(r"^(\s*---[\r\n]+)([\S\s]*?)([\r\n]+---(\r\n|\n|$))")

CHANGELOG_DIR = Path(__file__).parents[1] / ".changelog"

VALID_LABELS = {"breaking", "feature", "fix", "docs", "internal", "misc"}
REQUIRED_FIELDS = {"label", "title", "author"}


def validate_fragment(path: Path) -> list[str]:
    """Validate a single changelog fragment file.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    try:
        content = path.read_text()
    except Exception as e:
        return [f"Failed to read file: {e}"]

    # Check for YAML front matter
    match = REGEX_YAML_BLOCK.match(content)
    if not match:
        return ["Missing or invalid YAML front matter (must start with '---')"]

    yaml_content = match.group(2)

    # Parse YAML
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        return [f"Invalid YAML syntax: {e}"]

    if not isinstance(data, dict):
        return ["YAML front matter must be a mapping"]

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")

    # Validate label
    if "label" in data:
        label = data["label"]
        if label not in VALID_LABELS:
            errors.append(f"Invalid label '{label}'. Must be one of: {', '.join(sorted(VALID_LABELS))}")

    # Validate 'related' field if present and not empty
    if "related" in data and data["related"] is not None:
        related = data["related"]
        if not isinstance(related, int):
            errors.append("'related' field must be an integer (PR number)")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate changelog fragment files")
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Specific fragment files to validate (defaults to all fragments in .changelog/)",
    )

    args = parser.parse_args()

    # Determine which files to validate
    if args.files:
        fragments = args.files
    else:
        fragments = sorted(CHANGELOG_DIR.glob("*.md"))

    if not fragments:
        print("No changelog fragments found.")
        return 0

    has_errors = False

    for fragment in fragments:
        errors = validate_fragment(fragment)
        if errors:
            has_errors = True
            for error in errors:
                print(f"\u274c {fragment}: {error}")
        else:
            print(f"\u2705 {fragment}: Valid")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
