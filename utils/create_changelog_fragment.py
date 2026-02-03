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
"""Creates or updates changelog fragment files for PRs.

Usage:
    # Create a new fragment
    python utils/create_changelog_fragment.py --pr-number 123 --pr-title "Add feature" --pr-author "username"

    # Update title only (preserves all other fields)
    python utils/create_changelog_fragment.py --pr-number 123 --pr-title "New title" --update-title

Exit codes:
    0: Success
    1: Error
    2: Fragment already exists (when not using --update-title)
"""

import argparse
import re
import sys
from pathlib import Path


# Same regex as in huggingface_hub/repocard.py
REGEX_YAML_BLOCK = re.compile(r"^(\s*---[\r\n]+)([\S\s]*?)([\r\n]+---(\r\n|\n|$))")

CHANGELOG_DIR = Path(__file__).parents[1] / ".changelog"

FRAGMENT_TEMPLATE = """---
label: misc
title: {title}
author: {author}
related: null
---
"""


def parse_yaml_front_matter(content: str) -> tuple[dict, str]:
    """Parse YAML front matter from markdown content.

    Returns:
        Tuple of (yaml_data dict, body content after front matter)
    """
    import yaml

    match = REGEX_YAML_BLOCK.match(content)
    if not match:
        return {}, content

    yaml_content = match.group(2)
    body = content[match.end() :]
    data = yaml.safe_load(yaml_content) or {}
    return data, body


def create_fragment(pr_number: int, pr_title: str, pr_author: str) -> int:
    """Create a new changelog fragment file.

    Returns:
        Exit code (0=success, 1=error, 2=already exists)
    """
    fragment_path = CHANGELOG_DIR / f"{pr_number}.md"

    if fragment_path.exists():
        print(f"Fragment already exists: {fragment_path}")
        return 2

    # Escape quotes in title for YAML
    escaped_title = pr_title.replace('"', '\\"')

    content = FRAGMENT_TEMPLATE.format(title=escaped_title, author=pr_author)

    fragment_path.write_text(content)
    print(f"Created fragment: {fragment_path}")
    return 0


def update_fragment_title(pr_number: int, pr_title: str) -> int:
    """Update the title field in an existing fragment, preserving all other fields.

    Returns:
        Exit code (0=success, 1=error)
    """
    import yaml

    fragment_path = CHANGELOG_DIR / f"{pr_number}.md"

    if not fragment_path.exists():
        print(f"Fragment does not exist: {fragment_path}")
        return 1

    content = fragment_path.read_text()
    data, body = parse_yaml_front_matter(content)

    if not data:
        print(f"Failed to parse YAML front matter in: {fragment_path}")
        return 1

    # Update only the title
    data["title"] = pr_title

    # Reconstruct the file with updated YAML
    # Preserve field order: label, title, author, related, then any others
    ordered_data = {}
    for key in ["label", "title", "author", "related"]:
        if key in data:
            ordered_data[key] = data.pop(key)
    # Add any remaining fields
    ordered_data.update(data)

    yaml_content = yaml.dump(ordered_data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    new_content = f"---\n{yaml_content}---\n{body}"

    fragment_path.write_text(new_content)
    print(f"Updated title in fragment: {fragment_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or update changelog fragment files")
    parser.add_argument("--pr-number", type=int, required=True, help="PR number")
    parser.add_argument("--pr-title", type=str, required=True, help="PR title")
    parser.add_argument("--pr-author", type=str, help="PR author (required for new fragments)")
    parser.add_argument("--update-title", action="store_true", help="Update title only in existing fragment")

    args = parser.parse_args()

    if args.update_title:
        return update_fragment_title(args.pr_number, args.pr_title)
    else:
        if not args.pr_author:
            parser.error("--pr-author is required when creating a new fragment")
        return create_fragment(args.pr_number, args.pr_title, args.pr_author)


if __name__ == "__main__":
    sys.exit(main())
