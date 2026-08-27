#!/usr/bin/env python3
"""Generate a Slack announcement message from existing release notes.

This script:
1. Reads the generated release notes markdown
2. Invokes an OpenCode skill to summarize into a Slack-formatted message
3. Appends a deterministic "Ping" section with CI links
4. Outputs the final message to .release-notes/

Usage:
  python -m utils.release_notes.generate_slack_message --version v1.7.0 --rc-version 1.7.0.rc0
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .opencode import check_opencode_model
from .validate_notes import find_release_notes_file


OUTPUT_DIR = Path(os.environ.get("RELEASE_NOTES_OUTPUT_DIR", ".release-notes"))

# Downstream repos to ping for RC testing.
DEFAULT_PING_LIST: list[str] = [
    "transformers",
    "datasets",
    "diffusers",
    "sentence-transformers",
]


def build_ping_section(rc_version: str) -> str:
    """Build the "Ping:" block with CI compare URLs and closing line.

    Args:
        rc_version: RC version used in the CI test branch names (e.g., "1.7.0.rc0")

    Returns:
        The ping section + closing line as a string.
    """

    lines = ["Ping:"]
    for repo in DEFAULT_PING_LIST:
        url = f"https://github.com/huggingface/{repo}/compare/main...ci-test-huggingface-hub-{rc_version}-release"
        lines.append(f"- {repo}  => {url}")

    lines.append("")
    lines.append("Let us know if you spot any regressions! Release should be happening pretty soon :hugging_face:")

    return "\n".join(lines)


def run_opencode_skill(
    version: str,
    input_file: Path,
    output_file: Path,
) -> bool:
    """Run the hf-release-notes:slack OpenCode skill.

    Args:
        version: Base version (e.g., "v1.7.0")
        input_file: Path to the release notes markdown
        output_file: Path to write the Slack message body

    Returns:
        True if successful, False otherwise
    """
    prompt = (
        f"Run the hf-release-notes:slack skill. "
        f"Generate a Slack announcement message for version {version}. "
        f"Read the release notes from {input_file}. "
        f"Write the message body to {output_file}. "
        f"Do NOT include the Ping section or closing line — those are appended by the script."
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


def main(version: str, rc_version: str, input_file: Path | None = None, output_file: Path | None = None) -> int:
    """Generate a Slack announcement message.

    Args:
        version: Base version (e.g., "v1.7.0")
        rc_version: RC version used in the CI test branch names (e.g., "1.7.0.rc0")
        input_file: Path to release notes (auto-detected if None)
        output_file: Output path (defaults to .release-notes/SLACK_MESSAGE_{version}.md)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Validate the configured model before doing any expensive work.
    # OpenCode exits 0 on unknown models, so a typo here would otherwise
    # silently produce an empty Slack message.
    model = os.environ.get("RELEASE_NOTES_MODEL")
    if model:
        try:
            check_opencode_model(model)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # Resolve input file
    if input_file is None:
        input_file = find_release_notes_file(version)
    if input_file is None or not input_file.exists():
        print(f"Error: Release notes file not found for {version}", file=sys.stderr)
        print(f"Expected at: {OUTPUT_DIR}/RELEASE_NOTES_{version}.md", file=sys.stderr)
        return 1

    # Resolve output file
    if output_file is None:
        output_file = OUTPUT_DIR / f"SLACK_MESSAGE_{version}.md"

    # Clean previous output
    if output_file.exists():
        output_file.unlink()

    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")

    # Generate message body with OpenCode
    body_file = OUTPUT_DIR / f"_slack_body_{version}.md"
    if not run_opencode_skill(version, input_file, body_file):
        print("Failed to generate Slack message body", file=sys.stderr)
        return 1

    if not body_file.exists():
        print(f"Error: OpenCode did not produce output at {body_file}", file=sys.stderr)
        return 1

    body = body_file.read_text().strip()

    # Build ping section
    ping_section = build_ping_section(rc_version=rc_version)

    # Combine and write final message
    final_message = f"{body}\n\n{ping_section}\n"
    output_file.write_text(final_message)

    # Clean up intermediate file
    body_file.unlink(missing_ok=True)

    # Print to stdout for convenience
    print("\n" + "=" * 60)
    print("SLACK MESSAGE")
    print("=" * 60)
    print(final_message)
    print("=" * 60)
    print(f"Saved to {output_file}")

    return 0


def cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate a Slack announcement message from release notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Slack message for v1.7.0 prerelease
  python -m utils.release_notes.generate_slack_message --version v1.7.0 --rc-version 1.7.0.rc0

  # With custom input/output paths
  python -m utils.release_notes.generate_slack_message \\
    --version v1.7.0 --rc-version 1.7.0.rc0 \\
    --input .release-notes/RELEASE_NOTES_v1.7.0.md \\
    --output slack_message.md
""",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Base version for release notes lookup (e.g., v1.7.0)",
    )
    parser.add_argument(
        "--rc-version",
        required=True,
        help="Full RC version used in the CI test branch names (e.g., 1.7.0.rc0)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to release notes file (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (defaults to .release-notes/SLACK_MESSAGE_{version}.md)",
    )
    args = parser.parse_args()

    sys.exit(main(args.version, args.rc_version, args.input, args.output))


if __name__ == "__main__":
    cli()
