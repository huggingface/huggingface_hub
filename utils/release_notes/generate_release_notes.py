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
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .fetch_prs import fetch_prs_since_tag
from .validate_notes import validate_release_notes


OUTPUT_DIR = Path(os.environ.get("RELEASE_NOTES_OUTPUT_DIR", ".release-notes"))
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


def check_opencode_model(model: str) -> None:
    """Verify that ``model`` is listed by ``opencode models``.

    OpenCode exits 0 and prints an error line when an unknown model is passed
    via ``--model``, so calls silently no-op unless we validate up front.
    Raises ``RuntimeError`` if opencode is missing or the model is unknown.
    """
    opencode_cmd = shutil.which("opencode")
    if not opencode_cmd:
        raise RuntimeError("'opencode' command not found in PATH")

    result = subprocess.run(
        [opencode_cmd, "models"], check=True, capture_output=True, text=True
    )
    available = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if model not in available:
        raise RuntimeError(
            f"RELEASE_NOTES_MODEL={model!r} not found in `opencode models` output. "
            f"Expected the full `provider/model` form (e.g. `huggingface/zai-org/GLM-4.6`). "
            f"{len(available)} model(s) available — first 10: {', '.join(available[:10])}"
        )


def run_opencode_skill(
    skill_name: str,
    version: str,
    missing_prs: list[int] | None = None,
    extra_prs: list[int] | None = None,
) -> bool:
    """Run an OpenCode skill non-interactively.

    Args:
        skill_name: Name of the skill to run (e.g., "hf-release-notes")
        version: Target version for release notes
        missing_prs: List of missing PR numbers (for validation skill)
        extra_prs: List of extra PR numbers to remove (for validation skill)

    Returns:
        True if successful, False otherwise
    """
    if missing_prs or extra_prs:
        # Validation skill - fix missing and/or extra PRs
        parts = [
            f"Run the {skill_name} skill. ",
            f"The output directory is {OUTPUT_DIR}. ",
            f"Fix the release notes at {OUTPUT_DIR}/RELEASE_NOTES_{version}.md. ",
        ]
        if missing_prs:
            parts.append(
                f"Add the following missing PRs: {', '.join(f'#{pr}' for pr in missing_prs)}. "
                f"Read their details from {TMP_DIR}/pr_<number>.json files. "
            )
        if extra_prs:
            parts.append(
                f"Remove the following extra PRs that do not belong to this release: "
                f"{', '.join(f'#{pr}' for pr in extra_prs)}. "
            )
        prompt = "".join(parts)
    else:
        # Main generation skill
        prompt = (
            f"Run the {skill_name} skill. "
            f"The output directory is {OUTPUT_DIR}. "
            f"Generate release notes for {version} from PR files in {TMP_DIR}/. "
            f"Output to {OUTPUT_DIR}/RELEASE_NOTES_{version}.md"
        )

    # Check if opencode is available
    opencode_cmd = shutil.which("opencode")
    if not opencode_cmd:
        print("Error: 'opencode' command not found in PATH", file=sys.stderr)
        print("Please install OpenCode or ensure it's in your PATH", file=sys.stderr)
        return False

    # Run opencode non-interactively
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


def main(since_tag: str, bump_type: str = "patch", max_iterations: int = 3) -> int:
    """Run the full release notes generation pipeline.

    Args:
        since_tag: Git tag to compare against (e.g., "v1.3.7")
        bump_type: Version bump type ("major", "minor", or "patch")
        max_iterations: Maximum validation/fix iterations

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    t_total_start = time.monotonic()

    # 0. Validate the configured model before doing any expensive work.
    #    OpenCode exits 0 on unknown models, so a typo here would otherwise
    #    silently produce empty release notes.
    model = os.environ.get("RELEASE_NOTES_MODEL")
    if model:
        try:
            check_opencode_model(model)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # 1. Clean up and setup directories
    if OUTPUT_DIR.exists():
        print("Cleaning up previous release notes...")
        shutil.rmtree(OUTPUT_DIR)
    print("Setting up directories...")
    setup_directories()

    # 2. Determine next version
    version = bump_version(since_tag, bump_type)
    print(f"Target version: {version} ({bump_type} bump)")

    # 3. Fetch all PRs since tag
    print(f"\nFetching PRs since {since_tag}...")
    t_fetch_start = time.monotonic()
    try:
        pr_numbers = fetch_prs_since_tag(since_tag)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    t_fetch = time.monotonic() - t_fetch_start

    if not pr_numbers:
        print("No PRs found since the specified tag")
        return 1

    print(f"Fetched {len(pr_numbers)} PRs")

    # 4. Generate initial draft with OpenCode
    print("\nGenerating release notes with OpenCode...")
    t_agent_start = time.monotonic()
    agent_calls = 0
    if not run_opencode_skill("hf-release-notes", version):
        print("Failed to generate initial release notes", file=sys.stderr)
        return 1
    agent_calls += 1

    # OpenCode can exit 0 without writing the expected file (e.g. auth/quota
    # issues). Fail fast instead of falling through to a 0-byte output.
    initial_output = OUTPUT_DIR / f"RELEASE_NOTES_{version}.md"
    if not initial_output.exists() or initial_output.stat().st_size == 0:
        print(
            f"Error: OpenCode did not produce {initial_output} (or file is empty). "
            f"Check OpenCode logs above for the real error.",
            file=sys.stderr,
        )
        return 1

    # 5. Validation loop
    validation_iterations = 0
    missing_at_end = []
    extra_at_end = []
    for i in range(max_iterations):
        validation_iterations += 1
        print(f"\nValidation iteration {i + 1}/{max_iterations}...")
        missing, extra = validate_release_notes(version)

        if not missing and not extra:
            print("Release notes match the manifest exactly")
            break

        if missing:
            print(f"Missing {len(missing)} PRs: {', '.join(f'#{pr}' for pr in missing)}")
        if extra:
            print(f"Extra {len(extra)} PRs: {', '.join(f'#{pr}' for pr in extra)}")

        if i < max_iterations - 1:
            print("Running validation skill to fix release notes...")
            if not run_opencode_skill("hf-release-notes:validate", version, missing or None, extra or None):
                print("Warning: Validation skill failed", file=sys.stderr)
            agent_calls += 1
    else:
        # Loop completed without exact match
        missing, extra = validate_release_notes(version)
        if missing or extra:
            missing_at_end = missing
            extra_at_end = extra
            if missing:
                print(f"\nWarning: Still missing {len(missing)} PRs after {max_iterations} iterations")
                print(f"Missing: {', '.join(f'#{pr}' for pr in missing)}")
            if extra:
                print(f"\nWarning: Still {len(extra)} extra PRs after {max_iterations} iterations")
                print(f"Extra: {', '.join(f'#{pr}' for pr in extra)}")
    t_agent = time.monotonic() - t_agent_start

    # 6. Final output
    output_file = OUTPUT_DIR / f"RELEASE_NOTES_{version}.md"
    output_size = output_file.stat().st_size if output_file.exists() else 0
    t_total = time.monotonic() - t_total_start

    # 7. Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Version:               {version} ({bump_type} bump from {since_tag})")
    print(f"  Output dir:            {OUTPUT_DIR}")
    print(f"  Model:                 {os.environ.get('RELEASE_NOTES_MODEL', '(default)')}")
    print(f"  PRs fetched:           {len(pr_numbers)}")
    print(f"  PRs missing:           {len(missing_at_end)}")
    print(f"  PRs extra:             {len(extra_at_end)}")
    print(f"  Agent calls:           {agent_calls}")
    print(f"  Validation iterations: {validation_iterations}")
    print(f"  Fetch time:            {t_fetch:.1f}s")
    print(f"  Agent time:            {t_agent:.1f}s")
    print(f"  Total time:            {t_total:.1f}s")
    print(f"  Output file:           {output_file}")
    print(f"  Output size:           {output_size:,} bytes")
    print("=" * 60)

    if output_size == 0:
        print(
            f"Error: {output_file} is empty after validation loop.",
            file=sys.stderr,
        )
        return 1

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
