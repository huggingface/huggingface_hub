#!/usr/bin/env python3
"""Validate that release notes match the manifest exactly.

Checks that:
- All PR numbers from manifest.json appear in the release notes (no missing PRs)
- No extra PR numbers appear in the release notes that aren't in the manifest
  (i.e., PRs shipped in a different release should not be mentioned)

Exit codes:
- 0: Release notes match the manifest exactly
- 1: Missing or extra PRs detected
"""

import json
import os
import re
import sys
from pathlib import Path


OUTPUT_DIR = Path(os.environ.get("RELEASE_NOTES_OUTPUT_DIR", ".release-notes"))

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


def validate_release_notes(version: str | None = None) -> tuple[list[int], list[int]]:
    """Validate that release notes match the manifest exactly.

    Checks for both missing PRs (in manifest but not in notes) and extra PRs
    (in notes but not in manifest, i.e. belonging to a different release).

    Args:
        version: Optional version string (e.g., "v1.3.8")

    Returns:
        Tuple of (missing_prs, extra_prs) where each is a sorted list of PR numbers.
    """
    # Load expected PR numbers
    expected_prs = set(load_manifest())

    # Find and read release notes
    release_file = find_release_notes_file(version)
    if not release_file:
        print(f"No release notes file found in {OUTPUT_DIR}")
        return sorted(expected_prs), []

    with open(release_file) as f:
        content = f.read()

    # Extract PR references from release notes
    found_prs = extract_pr_references(content)

    # Find missing PRs (expected but not found in notes)
    missing_prs = expected_prs - found_prs
    # Find extra PRs (found in notes but not expected)
    extra_prs = found_prs - expected_prs
    return sorted(missing_prs), sorted(extra_prs)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate release notes completeness and accuracy")
    parser.add_argument("--version", help="Version to validate (e.g., v1.3.8)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    try:
        missing, extra = validate_release_notes(args.version)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    has_errors = bool(missing or extra)

    if args.json:
        print(
            json.dumps(
                {
                    "missing": missing,
                    "missing_count": len(missing),
                    "extra": extra,
                    "extra_count": len(extra),
                }
            )
        )
    else:
        if missing:
            print(f"Missing {len(missing)} PRs (in manifest but not in notes):")
            for pr_num in missing:
                print(f"  #{pr_num}")
        if extra:
            print(f"Extra {len(extra)} PRs (in notes but not in manifest):")
            for pr_num in extra:
                print(f"  #{pr_num}")
        if not has_errors:
            print("Release notes match the manifest exactly")

    sys.exit(0 if not has_errors else 1)


if __name__ == "__main__":
    main()
