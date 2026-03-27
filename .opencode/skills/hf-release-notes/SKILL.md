---
name: hf-release-notes
description: Generate Hugging Face Hub (huggingface_hub) release notes from cached PR JSON files. Use when asked to draft release notes from PR files.
---

# HF Release Notes

## Overview

Generate release notes for huggingface_hub from cached PR JSON files. This skill reads PR metadata, categorizes entries, and produces a formatted markdown release notes document.

**Output directory:** The prompt will specify the output directory as `<output_dir>`. All paths below use this placeholder.

## Workflow

### 1. Read PR data

Read all PR JSON files from `<output_dir>/tmp/pr_*.json`. Each file contains:

```json
{
  "number": 1234,
  "title": "...",
  "author": "username",
  "merged_at": "2026-01-15T10:30:00Z",
  "body": "...",
  "labels": ["highlight", "cli"],
  "url": "https://github.com/huggingface/huggingface_hub/pull/1234",
  "doc_diffs": [
    {
      "filename": "docs/source/en/guides/cli.md",
      "status": "modified",
      "patch": "@@ -10,6 +10,10 @@ ..."
    }
  ]
}
```

The `doc_diffs` field contains unified diffs for any `.md` files under `docs/` that were
changed in the PR. This is empty (`[]`) for PRs with no documentation changes.

### 2. Identify highlights

A PR should be highlighted if:
- It has the `"highlight"` label on GitHub, **or**
- You judge it significant enough to deserve a detailed section (e.g., a major new
  feature, a meaningful UX improvement, or a notable breaking change) even without the label.

Use your judgment — not every feature PR needs a highlight, but don't limit highlights
to only labeled PRs. Each highlight section follows this structure:

1. **Emoji header** — e.g., `## 🖥️ New CLI commands`
2. **Prose summary** — 2-5 sentences describing the user-visible change in flowing text.
   Avoid bullet points here; write natural sentences. Only use bullets if the content
   truly calls for a list (e.g., enumerating 4+ distinct sub-features).
3. **Code examples** — if the PR introduces new commands or APIs, include a fenced code
   block with a concrete usage example (cherry-pick from doc diffs when available).
4. **PR attribution lines** — one bullet per PR that contributed to this highlight:
   `- PR title by @author in #1234`

### 3. Classify standard items

For non-highlight PRs, classify into sections using `references/sections.md` heuristics based on:
- PR labels
- PR title keywords
- PR body content

### 4. Use doc diffs and fetch relevant documentation

For highlighted PRs and other PRs that introduce new features, commands, or APIs:

**Use doc diffs first:**
- Check the `doc_diffs` field in the PR JSON. If present, these contain the actual
  documentation changes made in the PR (unified diff format).
- Use these diffs to understand what was documented, extract code examples, and write
  more accurate summaries. The diffs show exactly what the PR author wrote in the docs.
- The `filename` field maps to a docs page URL. For example,
  `docs/source/en/guides/cli.md` → `https://huggingface.co/docs/huggingface_hub/main/en/guides/cli`

**Fetch full doc pages when needed:**
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

Output to `<output_dir>/RELEASE_NOTES_<version>.md` using the structure from `references/release-notes-template.md`:

- Title: `# [vX.Y.Z] <tagline>` (derive tagline from main highlights; always use the base version without prerelease suffix, e.g. `[v1.8.0]` even when generating for `v1.8.0.rc0`)
- One section per highlight with emoji header and narrative
- Standard sections for remaining items (only include sections with items)

### 6. Quality checks

Before finishing:
- Verify every PR from `<output_dir>/tmp/` appears exactly once
- **Do NOT include any PR that does not have a corresponding file in `<output_dir>/tmp/`** — those PRs belong to a different release and must not appear in these notes
- No empty sections
- Consistent emoji headings
- Every item ends with attribution: `by @author in #1234`
- Doc links point to real, verified pages

## Input

- Version string (e.g., "v1.3.8")
- PR JSON files in `<output_dir>/tmp/`

## Output

- Release notes markdown at `<output_dir>/RELEASE_NOTES_<version>.md`

## Resources

- `references/release-notes-template.md`: Skeleton structure for release notes
- `references/sections.md`: Keyword-based section mapping and guidance
- Documentation site: `https://huggingface.co/docs/huggingface_hub/main/en/index`
