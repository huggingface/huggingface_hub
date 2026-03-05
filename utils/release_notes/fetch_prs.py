#!/usr/bin/env python3
"""Fetch PR metadata from GitHub for release notes generation.

Uses PyGithub to:
- Get commits between a tag and HEAD on main
- Extract PR numbers from merge commits (GitHub squash-merge format)
- Fetch full PR details and save as JSON files
"""

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from github import Github


# Pattern to extract PR number from squash-merge commit titles
# Format: "PR title (#1234)"
PR_NUMBER_PATTERN = re.compile(r"\(#(\d+)\)$")

OUTPUT_DIR = Path(os.environ.get("RELEASE_NOTES_OUTPUT_DIR", ".release-notes"))
TMP_DIR = OUTPUT_DIR / "tmp"


def get_github_client() -> Github:
    """Get authenticated GitHub client."""
    token = os.environ.get("GITHUB_TOKEN_RELEASE_NOTES") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN_RELEASE_NOTES or GITHUB_TOKEN environment variable is required")
    return Github(token)


def get_commits_since_tag(repo, tag_name: str) -> list:
    """Get all commits between a tag and HEAD on main."""
    # Get the tag's commit SHA
    tags = {tag.name: tag for tag in repo.get_tags()}
    if tag_name not in tags:
        raise ValueError(f"Tag '{tag_name}' not found in repository")

    tag = tags[tag_name]
    tag_sha = tag.commit.sha

    # Get commits on main since the tag
    comparison = repo.compare(tag_sha, "main")
    return list(comparison.commits)


def extract_pr_number(commit_message: str) -> int | None:
    """Extract PR number from a squash-merge commit message."""
    # Get the first line (title) of the commit message
    title = commit_message.split("\n")[0].strip()
    match = PR_NUMBER_PATTERN.search(title)
    if match:
        return int(match.group(1))
    return None


def fetch_doc_diffs(pr) -> list[dict]:
    """Extract diffs for .md files under docs/ from a PR.

    Returns a list of dicts with filename, status, and patch for each changed doc file.
    """
    doc_diffs = []
    for f in pr.get_files():
        if f.filename.startswith("docs/") and f.filename.endswith(".md") and f.patch:
            doc_diffs.append(
                {
                    "filename": f.filename,
                    "status": f.status,
                    "patch": f.patch,
                }
            )
    return doc_diffs


def fetch_pr_details(repo, pr_number: int) -> dict:
    """Fetch full details for a PR, including doc diffs."""
    pr = repo.get_pull(pr_number)
    return {
        "number": pr.number,
        "title": pr.title,
        "author": pr.user.login,
        "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
        "body": pr.body or "",
        "labels": [label.name for label in pr.labels],
        "url": pr.html_url,
        "doc_diffs": fetch_doc_diffs(pr),
    }


def save_pr_json(pr_data: dict, output_dir: Path) -> None:
    """Save PR data to a JSON file."""
    output_file = output_dir / f"pr_{pr_data['number']}.json"
    with open(output_file, "w") as f:
        json.dump(pr_data, f, indent=2)


def save_manifest(pr_numbers: list[int], output_dir: Path) -> None:
    """Save manifest with all PR numbers."""
    manifest = {
        "pr_numbers": sorted(pr_numbers),
        "count": len(pr_numbers),
    }
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)


def fetch_prs_since_tag(tag_name: str, repo_name: str = "huggingface/huggingface_hub") -> list[int]:
    """Fetch all PRs merged since a tag and save to JSON files.

    Args:
        tag_name: The git tag to compare against (e.g., "v1.3.7")
        repo_name: The GitHub repository name

    Returns:
        List of PR numbers that were fetched
    """
    # Setup output directories
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Get GitHub client and repo
    gh = get_github_client()
    repo = gh.get_repo(repo_name)

    # Get commits since tag
    print(f"Fetching commits since {tag_name}...")
    commits = get_commits_since_tag(repo, tag_name)
    print(f"Found {len(commits)} commits")

    # Extract PR numbers from commits
    pr_numbers = []
    for commit in commits:
        pr_num = extract_pr_number(commit.commit.message)
        if pr_num:
            pr_numbers.append(pr_num)

    pr_numbers = list(set(pr_numbers))  # Deduplicate
    print(f"Found {len(pr_numbers)} unique PRs")

    # Fetch and save PR details concurrently (one GitHub client per thread for thread safety)
    _thread_local = threading.local()

    def _get_thread_repo():
        """Get a thread-local GitHub repo instance."""
        if not hasattr(_thread_local, "repo"):
            _thread_local.repo = get_github_client().get_repo(repo_name)
        return _thread_local.repo

    def _fetch_and_save(pr_num: int) -> int:
        thread_repo = _get_thread_repo()
        pr_data = fetch_pr_details(thread_repo, pr_num)
        save_pr_json(pr_data, TMP_DIR)
        return pr_num

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_and_save, pr_num): pr_num for pr_num in pr_numbers}
        for i, future in enumerate(as_completed(futures), 1):
            pr_num = futures[future]
            try:
                future.result()
                print(f"  [{i}/{len(pr_numbers)}] Fetched PR #{pr_num}")
            except Exception as e:
                print(f"  [{i}/{len(pr_numbers)}] Warning: Failed to fetch PR #{pr_num}: {e}")

    # Save manifest
    save_manifest(pr_numbers, OUTPUT_DIR)
    print(f"Saved manifest with {len(pr_numbers)} PRs to {OUTPUT_DIR / 'manifest.json'}")

    return pr_numbers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch PRs merged since a tag")
    parser.add_argument("--since", required=True, help="Tag to compare against (e.g., v1.3.7)")
    parser.add_argument("--repo", default="huggingface/huggingface_hub", help="GitHub repository")
    args = parser.parse_args()

    fetch_prs_since_tag(args.since, args.repo)
