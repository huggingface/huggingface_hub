# coding=utf-8
# Copyright 2026-present, the HuggingFace Inc. team.
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
"""Generates release notes from changelog fragments.

Usage:
    # Output to stdout
    python utils/generate_release_notes.py

    # With annotations (preview mode)
    python utils/generate_release_notes.py --preview

Exit codes:
    0: Success
    1: Error
"""

import argparse
import re
import sys
from pathlib import Path

import yaml


# Same regex as in huggingface_hub/repocard.py
REGEX_YAML_BLOCK = re.compile(r"^(\s*---[\r\n]+)([\S\s]*?)([\r\n]+---(\r\n|\n|$))")

CHANGELOG_DIR = Path(__file__).parents[1] / ".changelog"

# Section order and display names
# Features are rendered individually, others are grouped under headers
SECTION_ORDER = ["feature", "breaking", "docs", "misc", "fix", "internal"]
SECTION_HEADERS = {
    "breaking": "Breaking Changes",
    "fix": "Bug Fixes",
    "docs": "Documentation",
    "internal": "Internal",
    "misc": "Misc",
    # "feature" is handled specially - each feature gets its own ## section
}


def parse_fragment(path: Path) -> dict | None:
    """Parse a changelog fragment file.

    Returns:
        Dict with fragment data or None if parsing fails
    """
    try:
        content = path.read_text()
    except Exception:
        return None

    match = REGEX_YAML_BLOCK.match(content)
    if not match:
        return None

    yaml_content = match.group(2)
    body = content[match.end() :].strip()

    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError:
        return None

    if not isinstance(data, dict):
        return None

    # Extract PR number from filename
    pr_number = path.stem
    try:
        pr_number = int(pr_number)
    except ValueError:
        pass

    data["_pr_number"] = pr_number
    data["_body"] = body
    data["_path"] = path

    return data


def format_pr_link(pr_number: int) -> str:
    """Format a PR number as a GitHub link."""
    return f"[#{pr_number}](https://github.com/huggingface/huggingface_hub/pull/{pr_number})"


def format_author_link(author: str) -> str:
    """Format an author as a GitHub link."""
    return f"[@{author}](https://github.com/{author})"


def generate_release_notes(preview: bool = False) -> str:
    """Generate release notes from all fragments.

    Args:
        preview: If True, include annotations for reviewing

    Returns:
        Formatted release notes as markdown
    """
    fragments = sorted(CHANGELOG_DIR.glob("*.md"))

    if not fragments:
        return "No changelog fragments found.\n"

    # Parse all fragments and group by label
    grouped: dict[str, list[dict]] = {label: [] for label in SECTION_ORDER}

    for fragment_path in fragments:
        data = parse_fragment(fragment_path)
        if data is None:
            if preview:
                print(f"Warning: Failed to parse {fragment_path}", file=sys.stderr)
            continue

        label = data.get("label", "misc")
        if label not in grouped:
            label = "misc"  # Default to misc for unknown labels

        grouped[label].append(data)

    # Build related PR mapping (PR -> list of related PRs that reference it)
    related_to: dict[int, list[dict]] = {}
    primary_fragments: dict[str, list[dict]] = {label: [] for label in SECTION_ORDER}

    for label, items in grouped.items():
        for item in items:
            related = item.get("related")
            if related is not None:
                # This fragment is related to a main PR, add to related_to mapping
                if related not in related_to:
                    related_to[related] = []
                related_to[related].append(item)
            else:
                # This is a primary fragment
                primary_fragments[label].append(item)

    # Generate output
    output = []

    for label in SECTION_ORDER:
        items = primary_fragments[label]
        if not items:
            continue

        if label == "feature":
            # Features get individual ## sections
            for item in items:
                pr_number = item["_pr_number"]
                title = item.get("title", "Untitled")
                author = item.get("author", "unknown")
                body = item.get("_body", "")

                section_title = f"## {title}"
                if preview:
                    section_title += f" (PR {format_pr_link(pr_number)})"

                output.append(section_title)
                output.append("")

                # Add body content if present
                if body:
                    output.append(body)
                    output.append("")

                # Add attribution line
                attribution = f"*Contributed by {format_author_link(author)} in {format_pr_link(pr_number)}*"
                output.append(attribution)

                # Add related PRs as sub-bullets
                if pr_number in related_to:
                    output.append("")
                    output.append("Related PRs:")
                    for related_item in related_to[pr_number]:
                        related_pr = related_item["_pr_number"]
                        related_title = related_item.get("title", "")
                        related_author = related_item.get("author", "unknown")
                        output.append(
                            f"- {format_pr_link(related_pr)}: {related_title} by {format_author_link(related_author)}"
                        )

                output.append("")
        else:
            # Other categories get a header with bullet points
            header = SECTION_HEADERS.get(label, label.title())
            output.append(f"## {header}")
            output.append("")

            for item in items:
                pr_number = item["_pr_number"]
                title = item.get("title", "Untitled")
                author = item.get("author", "unknown")
                body = item.get("_body", "")

                # Main bullet point
                bullet = f"- {title} ({format_pr_link(pr_number)}) by {format_author_link(author)}"
                output.append(bullet)

                # Add body as indented content if present
                if body:
                    for line in body.split("\n"):
                        output.append(f"  {line}")

                # Add related PRs as sub-bullets
                if pr_number in related_to:
                    for related_item in related_to[pr_number]:
                        related_pr = related_item["_pr_number"]
                        related_title = related_item.get("title", "")
                        related_author = related_item.get("author", "unknown")
                        output.append(
                            f"  - {format_pr_link(related_pr)}: {related_title} "
                            f"by {format_author_link(related_author)}"
                        )

            output.append("")

    return "\n".join(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate release notes from changelog fragments")
    parser.add_argument("--preview", action="store_true", help="Include annotations for review")

    args = parser.parse_args()

    try:
        notes = generate_release_notes(preview=args.preview)
        print(notes)
        return 0
    except Exception as e:
        print(f"Error generating release notes: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
