#!/usr/bin/env python3
"""Validate that all PRs from manifest appear in release notes.

Checks that:
- All PR numbers from manifest.json appear in the release notes
- PRs are referenced as #<number> in the markdown

Exit codes:
- 0: All PRs present
- 1: Missing PRs
"""

import json
import re
import sys
from pathlib import Path


OUTPUT_DIR = Path(".release-notes")

# Pattern to find PR references in markdown (#1234)
PR_REFERENCE_PATTERN = re.compile(r"#(\d+)")


def load_manifest() -> list[int]:
    """Load PR numbers from manifest.json."""
    manifest_file = OUTPUT_DIR / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

    with open(manifest_file) as f:
        manifest = json.load(f)

    return manifest["pr_numbers"]


def find_release_notes_file(version: str | None = None) -> Path | None:
    """Find the release notes file for a version."""
    if version:
        # Look for specific version file
        file_path = OUTPUT_DIR / f"RELEASE_NOTES_{version}.md"
        if file_path.exists():
            return file_path
        return None

    # Find the most recent release notes file
    release_files = list(OUTPUT_DIR.glob("RELEASE_NOTES_*.md"))
    if not release_files:
        return None

    # Sort by modification time, most recent first
    release_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return release_files[0]


def extract_pr_references(markdown_content: str) -> set[int]:
    """Extract all PR numbers referenced in the markdown."""
    matches = PR_REFERENCE_PATTERN.findall(markdown_content)
    return {int(m) for m in matches}


def validate_release_notes(version: str | None = None) -> list[int]:
    """Validate that all PRs are included in release notes.

    Args:
        version: Optional version string (e.g., "v1.3.8")

    Returns:
        List of missing PR numbers (empty if all present)
    """
    # Load expected PR numbers
    expected_prs = set(load_manifest())

    # Find and read release notes
    release_file = find_release_notes_file(version)
    if not release_file:
        print(f"No release notes file found in {OUTPUT_DIR}")
        return list(expected_prs)

    with open(release_file) as f:
        content = f.read()

    # Extract PR references from release notes
    found_prs = extract_pr_references(content)

    # Find missing PRs
    missing_prs = expected_prs - found_prs
    return sorted(missing_prs)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate release notes completeness")
    parser.add_argument("--version", help="Version to validate (e.g., v1.3.8)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    try:
        missing = validate_release_notes(args.version)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps({"missing": missing, "count": len(missing)}))
    else:
        if missing:
            print(f"Missing {len(missing)} PRs:")
            for pr_num in missing:
                print(f"  #{pr_num}")
        else:
            print("All PRs included in release notes")

    sys.exit(0 if not missing else 1)


if __name__ == "__main__":
    main()
