---
name: hf-release-notes:slack
description: Generate a concise Slack announcement message from drafted release notes. Use when asked to create a Slack post for a huggingface_hub prerelease.
---

# Slack Announcement Message

## Overview

Generate a concise Slack announcement message from existing release notes. The message is intended for internal team communication to announce a prerelease and solicit testing from downstream maintainers.

**Important:** You generate ONLY the message body (greeting through the breaking changes line). The "Ping:" section and closing line are appended by the calling script — do NOT generate those.

## Workflow

### 1. Read inputs

The prompt will specify:
- **Version**: The base release version (e.g., `v1.7.0`)
- **Release notes path**: Path to the full release notes markdown file
- **Output path**: Where to write the Slack message

Read the release notes file first to understand what's in the release.

### 2. Read reference examples

Read `references/slack-post-template.md`. This file contains the template structure and real Slack messages from past releases. Use these to calibrate your tone, formatting, and level of detail. Match their style closely.

### 3. Generate the message

Write the Slack message body with these sections in order:

#### Greeting
```
Hello @channel :hello: Release `huggingface_hub vX.Y.Z` is on its way!
```

#### Release notes link
```
Release notes :point_right: https://github.com/huggingface/huggingface_hub/releases/tag/vX.Y.Z
```

#### Highlights
```
**Highlights:**
- **[Feature name]** 1 sentence summary of the feature.
- **[Another feature]** brief description.
```

Rules for highlights:
- Keep each highlight to 1-2 sentences max, even less — this is a summary, not the full release notes. Doesn't even have to be a full sentence.
- Drop ALL PR attribution lines (`by @author in #1234`)
- Drop ALL code examples and fenced code blocks
- Drop internal/CI/test items entirely
- Drop documentation-only items
- Drop bug fixes from highlights
- Group related small improvements together rather than listing each individually

#### Breaking changes
If there are breaking changes:
```
:warning: Breaking changes: brief description of what changed.
```
If none:
```
No breaking changes in this release.
```

### 4. Formatting rules

- **No markdown headers** (`##`, `###`) — Slack doesn't render these
- **Use backticks** for command names and code: `` `hf extensions install` ``
- **Keep it informal and friendly** — this is team communication, not a formal changelog
- **No trailing newlines** at the end of the output

### 5. Write output

Write ONLY the message body to the specified output path. Stop after the breaking changes line. Do NOT include:
- Any pip install command
- The "Ping:" section
- The "Let us know if you spot any regressions..." closing line
- Any separator lines

## Input

- Version string (e.g., `v1.7.0`)
- Path to the release notes markdown file
- Output path for the Slack message

## Output

- Slack message body at the specified output path

## Resources

- `references/slack-post-template.md`: Template structure and past Slack messages for tone/format reference
