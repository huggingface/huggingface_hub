# Changelog Automation System — Specification

This document describes an automated changelog management system for a Python package hosted on GitHub. It is inspired by [towncrier](https://github.com/twisted/towncrier) but uses a custom implementation.

## Overview

The system maintains changelog entries as individual fragment files (one per PR) that are collected at release time to generate release notes. This avoids merge conflicts and allows both automated creation and manual editing.

## Goals

1. Automatically create a changelog fragment when a PR is opened
2. Allow PR authors to edit and enrich the fragment
3. At release time, collect all fragments and generate structured release notes
4. Support "highlighted" PRs with detailed write-ups alongside simple one-liner entries
5. Eventually integrate with a GitHub Actions workflow to fully automate releases

---

## Fragment Files

### Location

All fragment files are stored in the `.changelog/` directory at the repository root.

### Naming Convention

```
.changelog/{pr_number}.md
```

Example: `.changelog/3669.md`

### File Format

Each fragment file contains a YAML metadata section followed by an optional free-form markdown body.

**Template:**

```markdown
---
label: misc  # one of: breaking, feature, fix, docs, internal, misc
title: <PR title>
author: <@username>
related:  # optional: main PR number this PR is related to
---

<!-- Optional: Add detailed description below -->
```

### Metadata Fields

| Field     | Type   | Default  | Description                                                                                   |
| --------- | ------ | -------- | --------------------------------------------------------------------------------------------- |
| `label`   | string | `misc`   | Category for the PR. Must be one of: `breaking`, `feature`, `fix`, `docs`, `internal`, `misc` |
| `title`   | string | *(auto)* | One-line description (auto-populated from PR title, always synced on PR title changes)        |
| `author`  | string | *(auto)* | GitHub username with `@` prefix (auto-populated from PR author)                               |
| `related` | number | *(none)* | Optional main PR number this PR is related to (e.g., `3668`). Groups PRs together in release notes. |

### Free-Form Content

The section below the YAML metadata section is optional. This content will be rendered in the release notes.

**Example of an edited fragment with highlight content:**

```markdown
---
label: feature  # one of: breaking, feature, fix, docs, internal, misc
title: Add `hf models`/`hf datasets`/`hf spaces` commands
author: @hanouticelina
---

## 🖥️ CLI: `hf models`, `hf datasets`, `hf spaces` Commands

The CLI has been reorganized with dedicated commands for Hub discovery, while `hf repo` stays focused on managing your own repositories.

**New commands:**

```bash
# Models
hf models ls --author=Qwen --limit=10
hf models info Qwen/Qwen-Image-2512

# Datasets
hf datasets ls --filter "format:parquet" --sort=downloads
hf datasets info HuggingFaceFW/fineweb
```

This organization mirrors the Python API (`list_models`, `model_info`, etc.), keeps the `hf <resource> <action>` pattern, and is extensible for future commands like `hf papers` or `hf collections`.
```

---

## GitHub Actions Bot

### Trigger

The bot runs on every PR event:
- `opened` (to create the fragment)
- `edited` (to update title if PR title changed)

### Skip Condition

If the PR title contains `[skip-changelog]` (case-insensitive), the bot does **not** create or update a fragment.

### Behavior

**On `opened` event:**
1. Check if fragment exists at `.changelog/{pr_number}.md`
2. If fragment does not exist: create it with default values (see template above)
3. If fragment exists: do nothing (exit code 2)
4. Commit and push the fragment to the PR branch (if created)
5. Comment on the PR (only once) to notify the author that a fragment was created and can be edited

**On `edited` event:**
1. Run script with `--update-title` flag to sync the `title` field with the current PR title
2. Commit and push if title was updated
3. Do not post a new comment

### Authentication

Use `secrets.HF_STYLE_BOT_ACTION` as the token for pushing commits.

### Fragment Creation Script

Location: `utils/create_changelog_fragment.py`

This script handles fragment creation logic, keeping the template and logic separate from the workflow file.

**Usage:**
```bash
# Create a new fragment
python utils/create_changelog_fragment.py --pr-number 123 --pr-title "Add feature" --pr-author "username"

# Update title in existing fragment (used when PR title changes)
python utils/create_changelog_fragment.py --pr-number 123 --pr-title "New title" --update-title
```

**Exit codes:**
- `0` - Fragment created or updated successfully
- `1` - Error occurred
- `2` - Fragment already exists (when not using `--update-title`)

**Flags:**
- `--update-title` - Update only the `title` field in an existing fragment, preserving all other fields and content. No-op if fragment doesn't exist.

### Workflow File

Location: `.github/workflows/changelog-bot.yml`

**Requirements:**
- Checkout the PR branch (not the base branch)
- Use the bot token for checkout to enable pushing
- Configure git user as a bot account (take inspiration from .github/workflows/style-bot.yml and .github/workflows/style-bot-action.yml)
- Call `utils/create_changelog_fragment.py` to create the fragment
- Only commit if there are changes (avoid empty commits)
- Only comment once per PR (check existing comments before posting)
- If label is "feature" and no free-form text in fragment, recommend to the user to add one (in a comment on the PR)

### Bot Comment Template

**Standard comment (for non-feature PRs):**

```markdown
📝 **Changelog fragment created**

A changelog entry has been created at `.changelog/{pr_number}.md`.

**Please review and edit if needed:**
- Set the appropriate `label` (breaking, feature, fix, docs, internal, misc)
- Add a detailed description below the metadata section for highlighted changes

[📄 View fragment](https://github.com/{owner}/{repo}/blob/{branch}/.changelog/{pr_number}.md) · [✏️ Edit fragment](https://github.com/{owner}/{repo}/edit/{branch}/.changelog/{pr_number}.md)
```

**Feature-specific comment (when `label: feature` is detected or for new features):**

```markdown
📝 **Changelog fragment created**

A changelog entry has been created at `.changelog/{pr_number}.md`.

**Please review and edit if needed:**
- Verify the `label` is correct (currently set to `feature`)
- **Recommended for features:** Add a detailed write-up below the metadata section to highlight this in the release notes

**Tips for a great feature description:**
- What problem does this solve?
- Show a quick usage example (code snippet, CLI command)
- Mention any related PR using the `related: 1234` field

[📄 View fragment](https://github.com/{owner}/{repo}/blob/{branch}/.changelog/{pr_number}.md) · [✏️ Edit fragment](https://github.com/{owner}/{repo}/edit/{branch}/.changelog/{pr_number}.md)
```

---

## CI Validation Workflow

### Purpose

Validates changelog fragment syntax on every PR to catch errors before merge, rather than at release time.

### Workflow File

Location: `.github/workflows/changelog-validate.yml`

### Trigger

Runs on every PR that modifies files in `.changelog/`:
- `pull_request` (opened, synchronize, reopened)

### Validation Rules

The workflow should check:

1. **Valid YAML metadata section:** Fragment must have valid YAML between `---` delimiters
2. **Required fields present:** `label`, `title`, and `author` must exist
3. **Valid label value:** `label` must be one of: `breaking`, `feature`, `fix`, `docs`, `internal`, `misc`
4. **PR number match:** Fragment filename must match an open PR number (optional, can be relaxed)
5. **Related field format:** If `related` is present, it must be an integer

### Validation Script

Location: `utils/validate_changelog_fragments.py`

**Usage:**
```bash
# Validate all fragments
python utils/validate_changelog_fragments.py

# Validate specific fragment(s)
python utils/validate_changelog_fragments.py .changelog/3669.md .changelog/3670.md
```

**Exit codes:**
- `0` - All fragments valid
- `1` - One or more fragments have errors (prints details to stderr)

### Error Output Format

```
❌ .changelog/3669.md: Invalid label 'enhancement'. Must be one of: breaking, feature, fix, docs, internal, misc
❌ .changelog/3670.md: Missing required field 'author'
✅ .changelog/3671.md: Valid
```

---

## Release Notes Generation

### Script Location

`utils/generate_release_notes.py`

### Input

All `.md` files in `.changelog/` directory.

### Output

Markdown-formatted release notes printed to stdout (can be redirected to a file or used in GitHub Release body).

### Generation Rules

1. **Parse all fragments:** Read YAML metadata section and optional body content
2. **Handle related PRs:** When a fragment has a `related` field pointing to another PR, it is grouped under that primary PR. The primary PR's entry includes the related PRs as sub-bullets.
3. **Group by label:** Organize PRs into categories
4. **Order categories:** Use this fixed order:
   - `feature` → New features are special as they have 1 section per new feature. No need for a `## ✨ New Features` title.
   - `breaking` → `## 💔 Breaking Changes`
   - `docs` → `## 📖 Documentation`
   - `misc` → `## 🔧 Miscellaneous`
   - `fix` → `## 🐛 Bug Fixes`
   - `internal` → `## 🏗️ Internal`
5. **Within each category:**
   - **First:** Render each PR with a free-form content followed by `* {title} by {author} in #{pr_number}` (for each of them)
   - **Then:** List remaining PRs as one-liners in format: `* {title} by {author} in #{pr_number}`
6. **Skip empty categories:** Don't render section headers for categories with no PRs

### Output Format Example

```markdown
## 🖥️ CLI: `hf models`, `hf datasets`, `hf spaces` Commands

The CLI has been reorganized with dedicated commands for Hub discovery...

[full free-form content from highlighted PR]

* Add `hf models`/`hf datasets`/`hf spaces` commands by @hanouticelina in #3669

## Add `hf jobs stats` command
<-- ^ here, no free-form text in the fragment => set a `##` section from fragment title and keep empty section -->

* Add `hf jobs stats` command by @lhoestq in #3655

## 💔 Breaking Changes

* Deprecate `direction` parameter in list methods by @hanouticelina in #3630

## 🐛 Bug Fixes

* Fix unbound local error when reading corrupted metadata files by @Wauplin in #3610
* Fix `create_repo` returning wrong `repo_id` by @hanouticelina in #3634
```

### Related PRs Output Example

When PRs use the `related` field to link to a main PR (e.g., a feature split across multiple PRs):

**Fragment `.changelog/3680.md` (main PR):**
```markdown
---
label: feature
title: Add async support for inference client (part 1)
author: @developer
related:
---

## Async Inference Client

The `InferenceClient` now supports async operations...
```

**Fragment `.changelog/3681.md` (related PR):**
```markdown
---
label: feature
title: Add async support for inference client (part 2)
author: @developer
related: 3680
---
```

**Fragment `.changelog/3682.md` (related PR):**
```markdown
---
label: feature
title: Add async support for inference client (part 3)
author: @developer
related: 3680
---
```

**Generated output:**
```markdown
## Async Inference Client

The `InferenceClient` now supports async operations...

* Add async support for inference client (part 1) by @developer in #3680
  * Add async support for inference client (part 2) by @developer in #3681
  * Add async support for inference client (part 3) by @developer in #3682
```

### CLI Interface

```bash
# Print release notes to stdout
python utils/generate_release_notes.py

# Preview mode: shows release notes with additional info (fragment sources, warnings)
python utils/generate_release_notes.py --preview
```

**Flags:**
- `--preview` - Preview mode for maintainers. Shows the same release notes output but with additional annotations: which fragment each entry comes from, any warnings (missing related PRs, etc.), and a summary of entries per category.

---

## Release Workflow (Future Goal)

A `workflow_dispatch` triggered workflow that:

1. Accepts version number as input
2. Runs `generate_release_notes.py` to create release notes
3. Updates version in `src/__init__.py`
4. Deletes all files in `.changelog/`
5. Commits changes with message `Release v{version}`
6. Creates and pushes git tag `v{version}`
7. Creates GitHub Release with generated notes
8. Publishes to PyPI

This is out of scope for now.

---

## File Structure

```
.
├── .changelog/
│   ├── 3669.md
│   ├── 3666.md
│   ├── 3655.md
│   └── ...
├── .github/
│   └── workflows/
│       ├── changelog-bot.yml           # Workflow: auto-creates fragments + comments
│       └── changelog-validate.yml      # Workflow: validates fragment syntax on PRs
└── utils/
    ├── create_changelog_fragment.py    # Creates a single fragment (called by workflow)
    ├── validate_changelog_fragments.py # Validates fragment syntax
    └── generate_release_notes.py       # Collects fragments → release notes
```

---

## Edge Cases to Handle

| Scenario                             | Behavior                                                     |
| ------------------------------------ | ------------------------------------------------------------ |
| PR title contains `[skip-changelog]` | No fragment created, no comment posted                       |
| Fragment already exists              | Do not overwrite; preserve manual edits                      |
| PR title is updated                  | Always sync `title` field (preserve other fields and content)|
| PR is closed without merge           | Fragment stays on the unmerged branch; main branch stays clean |
| PR is reopened                       | Fragment already exists; no action needed                    |
| Multiple PRs with same number        | Not possible on GitHub                                       |
| Fragment has invalid YAML            | Validation workflow fails with clear error message           |
| Fragment missing required fields     | Validation workflow fails; script should use defaults and warn |
| `related` references non-existent PR | Validation warns but doesn't fail; release script skips missing |
| `related` PR has different label     | Release script uses the primary PR's label for grouping      |
