#!/usr/bin/env python3
"""Generate draft social media posts from release notes.

This script:
1. Reads existing release notes (from file or auto-detected)
2. Invokes OpenCode with the hf-release-notes:social skill to generate
   6 draft posts (3 LinkedIn + 3 X) with different tones/angles
3. Validates that all expected files were produced
4. Outputs drafts to .release-notes/socials/

The drafts are saved to disk only. Uploading to a bucket is handled
separately by the caller (e.g. the release workflow).

Usage:
  python -m utils.release_notes.generate_social_posts --version v1.8.0

  # With explicit input
  python -m utils.release_notes.generate_social_posts \\
    --version v1.8.0 --input notes.md
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .validate_notes import find_release_notes_file


OUTPUT_DIR = Path(os.environ.get("RELEASE_NOTES_OUTPUT_DIR", ".release-notes"))

EXPECTED_FILES = [
    "linkedin_professional_1.txt",
    "linkedin_community_2.txt",
    "linkedin_tutorial_3.txt",
    "x_announcement_1.txt",
    "x_technical_2.txt",
    "x_thread_3.txt",
]


def run_opencode_skill(version: str, input_file: Path, output_dir: Path) -> bool:
    """Run the hf-release-notes:social OpenCode skill.

    Args:
        version: Release version (e.g., "v1.8.0")
        input_file: Path to the release notes markdown
        output_dir: Directory to write draft files

    Returns:
        True if successful, False otherwise
    """
    prompt = (
        f"Run the hf-release-notes:social skill. "
        f"Generate social media drafts for version {version}. "
        f"Read the release notes from {input_file}. "
        f"Write each draft file to {output_dir}/. "
    )

    opencode_cmd = shutil.which("opencode")
    if not opencode_cmd:
        print("Error: 'opencode' command not found in PATH", file=sys.stderr)
        return False

    cmd = [opencode_cmd]
    model = os.environ.get("RELEASE_NOTES_MODEL")
    if model:
        cmd.extend(["--model", model])
    cmd.extend(["run", prompt])
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


def main(version: str, input_file: Path | None = None) -> int:
    """Generate social media drafts from release notes.

    Args:
        version: Release version (e.g., "v1.8.0")
        input_file: Path to release notes markdown (auto-detected if None)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Resolve input
    if input_file is None:
        input_file = find_release_notes_file(version)
    if input_file is None or not input_file.exists():
        print(f"Error: Release notes file not found for {version}", file=sys.stderr)
        print(f"Expected at: {OUTPUT_DIR}/RELEASE_NOTES_{version}.md", file=sys.stderr)
        return 1

    # Prepare output directory
    socials_dir = OUTPUT_DIR / "socials"
    socials_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_file}")
    print(f"Output: {socials_dir}")

    # Generate with OpenCode
    if not run_opencode_skill(version, input_file, socials_dir):
        print("Failed to generate social media drafts", file=sys.stderr)
        return 1

    # Validate outputs
    generated = []
    missing = []
    for filename in EXPECTED_FILES:
        path = socials_dir / filename
        if path.exists() and path.stat().st_size > 0:
            generated.append(filename)
        else:
            missing.append(filename)

    print(f"\nGenerated {len(generated)}/{len(EXPECTED_FILES)} drafts in {socials_dir}")

    if missing:
        print(f"Missing: {', '.join(missing)}", file=sys.stderr)

    # Print summaries
    for filename in generated:
        path = socials_dir / filename
        content = path.read_text().strip()
        preview = content[:120] + "..." if len(content) > 120 else content
        print(f"  {filename} ({path.stat().st_size} bytes): {preview}")

    return 0 if not missing else 1


def cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate draft social media posts from release notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate drafts for v1.8.0
  python -m utils.release_notes.generate_social_posts --version v1.8.0

  # With explicit input file
  python -m utils.release_notes.generate_social_posts \\
    --version v1.8.0 --input notes.md
""",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Release version (e.g., v1.8.0)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to release notes file (auto-detected if omitted)",
    )
    args = parser.parse_args()

    sys.exit(main(args.version, args.input))


if __name__ == "__main__":
    cli()
