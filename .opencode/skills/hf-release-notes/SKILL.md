---
name: hf-release-notes
description: Generate Hugging Face Hub (huggingface_hub) release notes from cached PR JSON files. Use when asked to draft release notes from .release-notes/tmp/ PR files.
---

# HF Release Notes

## Overview

Generate release notes for huggingface_hub from cached PR JSON files in `.release-notes/tmp/`. This skill reads PR metadata, categorizes entries, and produces a formatted markdown release notes document.

## Workflow

### 1. Read PR data

Read all PR JSON files from `.release-notes/tmp/pr_*.json`. Each file contains:

```json
{
  "number": 1234,
  "title": "...",
  "author": "username",
  "merged_at": "2026-01-15T10:30:00Z",
  "body": "...",
  "labels": ["highlight", "cli"],
  "url": "https://github.com/huggingface/huggingface_hub/pull/1234"
}
```

### 2. Identify highlights

PRs with the `"highlight"` label should get detailed sections with:
- An emoji header
- A 2-5 sentence summary of the user-visible change
- Code examples if the PR introduces new commands or APIs
- The PR attribution line

### 3. Classify standard items

For non-highlight PRs, classify into sections using `references/sections.md` heuristics based on:
- PR labels
- PR title keywords
- PR body content

### 4. Fetch relevant documentation

For highlighted PRs and other PRs that introduce new features, commands, or APIs, check
if there is related documentation on the huggingface_hub docs site.

**How to check:**
1. Start by fetching the docs index page to discover the site structure:
   `https://huggingface.co/docs/huggingface_hub/main/en/index`
2. Based on the PR content (title, body, labels), identify which doc pages might be
   relevant. For example:
   - A CLI PR → check `https://huggingface.co/docs/huggingface_hub/main/en/guides/cli`
   - An Inference PR → check `https://huggingface.co/docs/huggingface_hub/main/en/guides/inference`
   - A new API class or method → check the corresponding reference page
3. Fetch the candidate doc page to confirm it covers the feature from the PR.
4. Only include a doc link if the page genuinely documents the feature.

**How to reference:**
- In highlight sections, add a line like:
  `📚 **Documentation:** [Guide name](https://huggingface.co/docs/huggingface_hub/main/en/guides/...)`
- In standard sections, append the doc link inline after the attribution when relevant:
  `- PR title by @author in #1234 — [docs](https://huggingface.co/docs/huggingface_hub/main/en/...)`
- Do NOT add doc links for internal/CI/test PRs or PRs that don't have user-facing docs.
- Do NOT fabricate doc URLs — only link to pages you have actually fetched and verified.

### 5. Generate release notes

Output to `.release-notes/RELEASE_NOTES_<version>.md` using the structure from `references/release-notes-template.md`:

- Title: `# vX.Y.Z: <tagline>` (derive tagline from main highlights)
- One section per highlight with emoji header and narrative
- Standard sections for remaining items (only include sections with items)

### 6. Quality checks

Before finishing:
- Verify every PR from `.release-notes/tmp/` appears exactly once
- No empty sections
- Consistent emoji headings
- Every item ends with attribution: `by @author in #1234`
- Doc links point to real, verified pages

## Input

- Version string (e.g., "v1.3.8")
- PR JSON files in `.release-notes/tmp/`

## Output

- Release notes markdown at `.release-notes/RELEASE_NOTES_<version>.md`

## Resources

- `references/release-notes-template.md`: Skeleton structure for release notes
- `references/sections.md`: Keyword-based section mapping and guidance
- Documentation site: `https://huggingface.co/docs/huggingface_hub/main/en/index`
