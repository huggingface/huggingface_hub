---
name: hf-release-notes:validate
description: Validate and fix release notes by adding missing PRs. Use when PRs are missing from an existing release notes draft.
---

# Validate Release Notes

## Overview

This skill validates that all PRs are included in the release notes and adds any missing ones to the appropriate sections.

## Workflow

### 1. Read current release notes

Read the existing release notes from `.release-notes/RELEASE_NOTES_<version>.md`.

### 2. Identify missing PRs

You will be provided with a list of missing PR numbers. For each missing PR:
- Read the PR details from `.release-notes/tmp/pr_<number>.json`
- Determine the appropriate section based on labels and title

### 3. Add missing PRs

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

### 4. Update the file

Write the updated release notes back to `.release-notes/RELEASE_NOTES_<version>.md`.

### 5. Verify

After updating, confirm that all previously missing PRs now appear in the document.

## Input

- Version string (e.g., "v1.3.8")
- List of missing PR numbers
- PR JSON files in `.release-notes/tmp/`

## Output

- Updated release notes at `.release-notes/RELEASE_NOTES_<version>.md`

## Resources

- `references/sections.md`: Section classification guidance
