#!/usr/bin/env python3
"""Main orchestrator for automated release notes generation.

This script:
1. Fetches all PRs merged since a given tag
2. Caches PR metadata as JSON files
3. Invokes OpenCode with a skill to generate release notes
4. Validates completeness and loops if PRs are missing
5. Outputs final draft to .release-notes/
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .fetch_prs import fetch_prs_since_tag
from .validate_notes import validate_release_notes


OUTPUT_DIR = Path(".release-notes")
TMP_DIR = OUTPUT_DIR / "tmp"


def setup_directories() -> None:
    """Create output directories."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def bump_version(tag: str, bump_type: str = "patch") -> str:
    """Bump a semver tag by major, minor, or patch.

    Args:
        tag: Version tag like "v1.3.7"
        bump_type: One of "major", "minor", or "patch"

    Returns:
        Bumped version (e.g., "v2.0.0", "v1.4.0", or "v1.3.8")
    """
    match = re.match(r"v?(\d+)\.(\d+)\.(\d+)", tag)
    if not match:
        raise ValueError(f"Invalid version tag format: {tag}")

    major, minor, patch = map(int, match.groups())

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    else:  # patch
        patch += 1

    prefix = "v" if tag.startswith("v") else ""
    return f"{prefix}{major}.{minor}.{patch}"


def run_opencode_skill(skill_name: str, version: str, missing_prs: list[int] | None = None) -> bool:
    """Run an OpenCode skill non-interactively.

    Args:
        skill_name: Name of the skill to run (e.g., "hf-release-notes")
        version: Target version for release notes
        missing_prs: List of missing PR numbers (for validation skill)

    Returns:
        True if successful, False otherwise
    """
    if missing_prs:
        # Validation skill - add missing PRs
        prompt = (
            f"Run the {skill_name} skill. "
            f"Add the following missing PRs to the release notes at .release-notes/RELEASE_NOTES_{version}.md: "
            f"{', '.join(f'#{pr}' for pr in missing_prs)}. "
            f"Read their details from .release-notes/tmp/pr_<number>.json files."
        )
    else:
        # Main generation skill
        prompt = (
            f"Run the {skill_name} skill. "
            f"Generate release notes for {version} from PR files in .release-notes/tmp/. "
            f"Output to .release-notes/RELEASE_NOTES_{version}.md"
        )

    # Check if opencode is available
    opencode_cmd = shutil.which("opencode")
    if not opencode_cmd:
        print("Error: 'opencode' command not found in PATH", file=sys.stderr)
        print("Please install OpenCode or ensure it's in your PATH", file=sys.stderr)
        return False

    # Run opencode non-interactively
    cmd = [opencode_cmd, "run", prompt]
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"OpenCode command failed with exit code {e.returncode}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("Error: 'opencode' command not found", file=sys.stderr)
        return False


def main(since_tag: str, bump_type: str = "patch", max_iterations: int = 3) -> int:
    """Run the full release notes generation pipeline.

    Args:
        since_tag: Git tag to compare against (e.g., "v1.3.7")
        bump_type: Version bump type ("major", "minor", or "patch")
        max_iterations: Maximum validation/fix iterations

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # 1. Setup directories
    print("Setting up directories...")
    setup_directories()

    # 2. Determine next version
    version = bump_version(since_tag, bump_type)
    print(f"Target version: {version} ({bump_type} bump)")

    # 3. Fetch all PRs since tag
    print(f"\nFetching PRs since {since_tag}...")
    try:
        pr_numbers = fetch_prs_since_tag(since_tag)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not pr_numbers:
        print("No PRs found since the specified tag")
        return 1

    print(f"Fetched {len(pr_numbers)} PRs")

    # 4. Generate initial draft with OpenCode
    print("\nGenerating release notes with OpenCode...")
    if not run_opencode_skill("hf-release-notes", version):
        print("Failed to generate initial release notes", file=sys.stderr)
        return 1

    # 5. Validation loop
    for i in range(max_iterations):
        print(f"\nValidation iteration {i + 1}/{max_iterations}...")
        missing = validate_release_notes(version)

        if not missing:
            print("All PRs included in release notes")
            break

        print(f"Missing {len(missing)} PRs: {', '.join(f'#{pr}' for pr in missing)}")

        if i < max_iterations - 1:
            print("Running validation skill to add missing PRs...")
            if not run_opencode_skill("hf-release-notes:validate", version, missing):
                print("Warning: Validation skill failed", file=sys.stderr)
    else:
        # Loop completed without all PRs included
        missing = validate_release_notes(version)
        if missing:
            print(f"\nWarning: Still missing {len(missing)} PRs after {max_iterations} iterations")
            print(f"Missing: {', '.join(f'#{pr}' for pr in missing)}")

    # 6. Final output
    output_file = OUTPUT_DIR / f"RELEASE_NOTES_{version}.md"
    print(f"\nRelease notes saved to {output_file}")
    return 0


def cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate release notes for huggingface_hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Patch release (v1.3.7 -> v1.3.8)
  python -m utils.release_notes.generate_release_notes --since v1.3.7 --patch

  # Minor release (v1.3.7 -> v1.4.0)
  python -m utils.release_notes.generate_release_notes --since v1.3.7 --minor

  # Major release (v1.3.7 -> v2.0.0)
  python -m utils.release_notes.generate_release_notes --since v1.3.7 --major
""",
    )
    parser.add_argument(
        "--since",
        required=True,
        help="Git tag to compare against (e.g., v1.3.7)",
    )

    # Version bump type (mutually exclusive)
    bump_group = parser.add_mutually_exclusive_group(required=True)
    bump_group.add_argument(
        "--major",
        action="store_const",
        const="major",
        dest="bump_type",
        help="Major version bump (v1.3.7 -> v2.0.0)",
    )
    bump_group.add_argument(
        "--minor",
        action="store_const",
        const="minor",
        dest="bump_type",
        help="Minor version bump (v1.3.7 -> v1.4.0)",
    )
    bump_group.add_argument(
        "--patch",
        action="store_const",
        const="patch",
        dest="bump_type",
        help="Patch version bump (v1.3.7 -> v1.3.8)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum validation/fix iterations (default: 3)",
    )
    args = parser.parse_args()

    sys.exit(main(args.since, args.bump_type, args.max_iterations))


if __name__ == "__main__":
    cli()
