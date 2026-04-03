---
name: hf-release-notes:validate
description: Validate and fix release notes by adding missing PRs and removing extra PRs. Use when the release notes don't match the manifest exactly.
---

# Validate Release Notes

## Overview

This skill validates that release notes match the manifest exactly: all expected PRs are included, and no extra PRs from other releases are present. It fixes both missing and extra PRs.

**Output directory:** The prompt will specify the output directory as `<output_dir>`. All paths below use this placeholder.

## Workflow

### 1. Read current release notes

Read the existing release notes from `<output_dir>/RELEASE_NOTES_<version>.md`.

### 2. Fix missing PRs

You may be provided with a list of missing PR numbers. For each missing PR:
- Read the PR details from `<output_dir>/tmp/pr_<number>.json`
- Check `doc_diffs` for documentation changes that can inform the summary
- Determine the appropriate section based on labels and title

For each missing PR:
- If it has the `"highlight"` label, create a new highlight section with:
  - Emoji header
  - 2-5 sentence summary
  - Code examples if applicable
  - Attribution line
  - If the PR introduces a user-facing feature, check for relevant documentation at
    `https://huggingface.co/docs/huggingface_hub/main/en/` and add a doc link
    (see main skill for details on how to fetch and verify doc pages)
- Otherwise, add to the appropriate standard section:
  - Match to existing sections first
  - Create new section if needed (following `references/sections.md`)

### 3. Remove extra PRs

You may be provided with a list of extra PR numbers that do not belong to this release (they were shipped in a different release). For each extra PR:
- Find all references to the PR (e.g., `#1234`) in the release notes
- Remove the entire line or bullet point that references the PR
- If the PR was the only item in a highlight section, remove the entire highlight section
- If removing the PR leaves a section empty, remove that section entirely
- Do NOT remove PRs that are in the manifest — only remove the ones explicitly listed as extra

### 4. Update the file

Write the updated release notes back to `<output_dir>/RELEASE_NOTES_<version>.md`.

### 5. Verify

After updating, confirm that:
- All previously missing PRs now appear in the document
- All previously extra PRs have been removed from the document

## Input

- Version string (e.g., "v1.3.8")
- List of missing PR numbers (may be empty)
- List of extra PR numbers to remove (may be empty)
- PR JSON files in `<output_dir>/tmp/`

## Output

- Updated release notes at `<output_dir>/RELEASE_NOTES_<version>.md`

## Resources

- `references/sections.md`: Section classification guidance
