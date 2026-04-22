#!/usr/bin/env python3
"""Generate draft social media posts from release notes.

This script:
1. Reads existing release notes (from file or GitHub draft release)
2. Generates multiple draft posts for LinkedIn and X (Twitter)
   with different tones/angles using an LLM via huggingface_hub
3. Writes each draft as a .txt file under a local output directory
4. Optionally uploads to a HF bucket via the `hf` CLI

Each platform gets posts with several distinct angles:
- LinkedIn: professional/technical, community/ecosystem, tutorial/practical
- X: announcement/hype, technical/developer, thread/storytelling

Usage:
  python -m utils.release_notes.generate_social_posts \
    --version v1.8.0 \
    --input .release-notes/RELEASE_NOTES_v1.8.0.md

  # With bucket upload
  python -m utils.release_notes.generate_social_posts \
    --version v1.8.0 \
    --upload-bucket huggingface/releases
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import InferenceClient

from .validate_notes import find_release_notes_file


OUTPUT_DIR = Path(os.environ.get("RELEASE_NOTES_OUTPUT_DIR", ".release-notes"))

# ── Post specifications ─────────────────────────────────────────────
# Each entry: (platform, tone_slug, system_prompt_extra, user_prompt_template)
# The user_prompt_template receives {version} and {release_notes} as format keys.

LINKEDIN_SPECS: list[tuple[str, str]] = [
    (
        "professional",
        (
            "You are a developer relations expert writing a LinkedIn post about an open-source "
            "Python library release. Write in a professional but approachable tone. Focus on "
            "the technical value and what problems these features solve for ML engineers and "
            "data scientists. Use short paragraphs. Include 2-3 relevant hashtags at the end "
            "(e.g. #MachineLearning #OpenSource #Python). The post should be 150-250 words. "
            "Suggest a picture idea in brackets at the very end, e.g. "
            "[Picture: screenshot of the new CLI output with a terminal theme]."
        ),
    ),
    (
        "community",
        (
            "You are a community manager writing a LinkedIn post celebrating an open-source "
            "release. Emphasize the collaborative nature: thank contributors, highlight the "
            "ecosystem impact (transformers, diffusers, datasets all depend on this library), "
            "and invite people to try it out. Keep a warm, inclusive tone. Mention that this "
            "is the Python client for the Hugging Face Hub. 150-250 words. Include 2-3 "
            "hashtags. Suggest a picture idea in brackets at the very end, e.g. "
            "[Picture: collage of contributor avatars or a community graphic]."
        ),
    ),
    (
        "tutorial",
        (
            "You are a technical writer creating a LinkedIn post that is practical and "
            "tutorial-oriented. Pick the 1-2 most impactful new features and show a quick "
            "before/after or a mini code snippet (use LinkedIn code formatting). The goal is "
            "to make readers think 'I need to try this right now.' 150-250 words. Include "
            "2-3 hashtags. Suggest a picture idea in brackets at the very end, e.g. "
            "[Picture: side-by-side code comparison or a short GIF of the feature in action]."
        ),
    ),
]

X_SPECS: list[tuple[str, str]] = [
    (
        "announcement",
        (
            "You are writing a punchy X (Twitter) post announcing a new release of "
            "huggingface_hub, the Python client for the Hugging Face Hub. Keep it under 280 "
            "characters for the main tweet. Use an energetic, excited tone. Lead with the "
            "single most exciting feature. Use 1-2 emoji max. Add a short follow-up tweet "
            "(also under 280 chars) with a link placeholder '[LINK]' to the release notes. "
            "Separate the two tweets with '---'. Suggest a picture idea in brackets at the "
            "very end, e.g. [Picture: hero image of the main new feature]."
        ),
    ),
    (
        "technical",
        (
            "You are a developer writing an X (Twitter) post targeting other developers. "
            "Focus on DX improvements, new APIs, or performance gains. Be concise and "
            "specific — mention concrete method names or CLI commands if relevant. Keep the "
            "main tweet under 280 characters. Include a follow-up tweet (under 280 chars) "
            "with a code snippet or command example. Separate tweets with '---'. Suggest a "
            "picture idea in brackets at the very end, e.g. "
            "[Picture: terminal screenshot showing the new command in action]."
        ),
    ),
    (
        "thread",
        (
            "You are writing a short X (Twitter) thread (3-5 tweets) walking through the "
            "highlights of a new release. Each tweet must be under 280 characters. Number "
            "them 1/, 2/, etc. The first tweet should hook the reader. The last tweet should "
            "link to the release notes with a '[LINK]' placeholder. Separate tweets with "
            "'---'. Suggest a picture idea in brackets at the very end, e.g. "
            "[Picture: infographic summarizing the top 3 features]."
        ),
    ),
]

USER_PROMPT_TEMPLATE = (
    "Write a {platform} post about huggingface_hub {version}.\n\nHere are the release notes:\n\n{release_notes}"
)


def generate_post(
    client: InferenceClient,
    model: str,
    platform: str,
    tone: str,
    system_prompt: str,
    version: str,
    release_notes: str,
) -> str:
    """Generate a single social media draft using the inference API.

    Returns the generated text.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        platform=platform,
        version=version,
        release_notes=release_notes,
    )
    response = client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


def upload_to_bucket(local_dir: Path, bucket_id: str, version: str) -> bool:
    """Upload generated drafts to an HF bucket using the `hf` CLI.

    Uploads files from local_dir to:
      hf://buckets/<bucket_id>/huggingface_hub/<version>/socials/

    Returns True on success.
    """
    hf_cmd = shutil.which("hf")
    if not hf_cmd:
        print("Error: 'hf' CLI not found in PATH", file=sys.stderr)
        return False

    dest = f"hf://buckets/{bucket_id}/huggingface_hub/{version}/socials/"
    cmd = [hf_cmd, "buckets", "sync", str(local_dir), dest]
    print(f"Uploading to bucket: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Bucket upload failed with exit code {e.returncode}", file=sys.stderr)
        return False


def main(
    version: str,
    input_file: Path | None = None,
    upload_bucket: str | None = None,
    model: str | None = None,
) -> int:
    """Generate social media drafts from release notes.

    Args:
        version: Release version (e.g., "v1.8.0")
        input_file: Path to release notes markdown (auto-detected if None)
        upload_bucket: Optional HF bucket ID to upload results (e.g., "huggingface/releases")
        model: HF model ID for text generation

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

    release_notes = input_file.read_text().strip()
    if not release_notes:
        print("Error: Release notes file is empty", file=sys.stderr)
        return 1

    # Resolve model
    if model is None:
        model = os.environ.get("RELEASE_NOTES_MODEL")
    if not model:
        print("Error: No model specified. Set RELEASE_NOTES_MODEL or use --model.", file=sys.stderr)
        return 1

    # Prepare output directory
    socials_dir = OUTPUT_DIR / "socials"
    socials_dir.mkdir(parents=True, exist_ok=True)

    # Create inference client
    client = InferenceClient()

    generated = 0
    errors = 0

    # Generate LinkedIn posts
    for idx, (tone, system_prompt) in enumerate(LINKEDIN_SPECS, start=1):
        filename = f"linkedin_{tone}_{idx}.txt"
        print(f"Generating {filename}...")
        try:
            text = generate_post(client, model, "LinkedIn", tone, system_prompt, version, release_notes)
            (socials_dir / filename).write_text(text + "\n")
            generated += 1
        except Exception as e:
            print(f"  Error generating {filename}: {e}", file=sys.stderr)
            errors += 1

    # Generate X posts
    for idx, (tone, system_prompt) in enumerate(X_SPECS, start=1):
        filename = f"x_{tone}_{idx}.txt"
        print(f"Generating {filename}...")
        try:
            text = generate_post(client, model, "X (Twitter)", tone, system_prompt, version, release_notes)
            (socials_dir / filename).write_text(text + "\n")
            generated += 1
        except Exception as e:
            print(f"  Error generating {filename}: {e}", file=sys.stderr)
            errors += 1

    print(f"\nGenerated {generated} drafts ({errors} errors) in {socials_dir}")

    # Upload to bucket if requested
    if upload_bucket and generated > 0:
        # Strip leading "v" for the bucket path if present
        bucket_version = version.lstrip("v")
        if not upload_to_bucket(socials_dir, upload_bucket, bucket_version):
            print("Warning: Bucket upload failed", file=sys.stderr)
            return 1
        print(f"Uploaded to bucket {upload_bucket}/huggingface_hub/{bucket_version}/socials/")

    return 0 if errors == 0 else 1


def cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate draft social media posts from release notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate drafts for v1.8.0
  python -m utils.release_notes.generate_social_posts --version v1.8.0

  # With explicit input and bucket upload
  python -m utils.release_notes.generate_social_posts \\
    --version v1.8.0 \\
    --input notes.md \\
    --upload-bucket huggingface/releases
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
    parser.add_argument(
        "--upload-bucket",
        default=None,
        help="HF bucket ID to upload results (e.g., huggingface/releases)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HF model ID for generation (defaults to RELEASE_NOTES_MODEL env var)",
    )
    args = parser.parse_args()

    sys.exit(main(args.version, args.input, args.upload_bucket, args.model))


if __name__ == "__main__":
    cli()
