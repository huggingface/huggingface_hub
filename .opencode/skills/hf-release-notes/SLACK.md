---
name: hf-release-notes:slack
description: Generate a concise Slack announcement message from drafted release notes. Use when asked to create a Slack post for a huggingface_hub prerelease.
---

# Slack Announcement Message

## Overview

Generate a concise Slack announcement message from existing release notes. The message is intended for internal team communication to announce a prerelease and solicit testing from downstream maintainers.

**Important:** You generate ONLY the message body (greeting through the pip install command). The "Pinging:" section and closing line are appended by the calling script — do NOT generate those.

## Workflow

### 1. Read inputs

The prompt will specify:
- **Version**: The base release version (e.g., `v1.7.0`)
- **RC version**: The prerelease version for pip install (e.g., `1.7.0rc0`)
- **Release notes path**: Path to the full release notes markdown file
- **Output path**: Where to write the Slack message

Read the release notes file first to understand what's in the release.

### 2. Read reference examples

Read `references/slack-post-template.md`. This file contains the template structure and real Slack messages from past releases. Use these to calibrate your tone, formatting, and level of detail. Match their style closely.

### 3. Generate the message

Write the Slack message body with these sections in order:

#### Greeting
```
Hello @canal :hello: The next release of `huggingface_hub` (vX.Y.Z) is on its way! :tadaco:
```

#### Release notes link
```
Release notes :point_right: https://github.com/huggingface/huggingface_hub/releases/tag/vX.Y.Z
```

#### Highlights
```
:sparkles: Highlights
 :emoji: Feature name: brief 1-2 sentence description
 :emoji: Another feature: description
 A bunch of QoL improvements:
  Sub-feature 1
  Sub-feature 2
```

Rules for highlights:
- Use Slack emoji codes (`:package:`, `:robot_face:`, `:fire:`, `:electric_plug:`, `:bucket:`, `:computer:`, `:zap:`, `:chart_with_upwards_trend:`, etc.), NOT Unicode emoji
- Keep each highlight to 1-2 sentences max — this is a summary, not the full release notes
- Drop ALL PR attribution lines (`by @author in #1234`)
- Drop ALL code examples and fenced code blocks
- Drop internal/CI/test items entirely
- Drop documentation-only items
- Drop bug fixes from highlights (mention them as "a bunch of QoL improvements and fixes" if there are several)
- Group related small improvements together rather than listing each individually
- Use single-space indentation for sub-items, matching the examples

#### Breaking changes
If there are breaking changes:
```
:warning: Breaking changes: brief description of what changed.
```
If none:
```
No breaking changes in this release.
```

#### Pre-release install command
```
You can try the pre-release now:
pip install -U huggingface_hub==<RC_VERSION>
```
Where `<RC_VERSION>` is the RC version provided in the prompt (e.g., `1.7.0rc0`).

### 4. Formatting rules

- **No markdown headers** (`##`, `###`) — Slack doesn't render these
- **No bold** (`**text**`) — use plain text or Slack formatting (` `` ` for inline code)
- **Use backticks** for command names and code: `` `hf extensions install` ``
- **Use Slack emoji codes** (`:sparkles:`) not Unicode emoji
- **Keep it informal and friendly** — this is team communication, not a formal changelog
- **Single space indent** for highlight items under the `:sparkles: Highlights` header
- **No trailing newlines** at the end of the output

### 5. Write output

Write ONLY the message body to the specified output path. Stop after the pip install command. Do NOT include:
- The "Pinging:" section
- The "Let us know if you spot any regressions..." closing line
- Any separator lines

## Input

- Version string (e.g., `v1.7.0`)
- RC version string (e.g., `1.7.0rc0`)
- Path to the release notes markdown file
- Output path for the Slack message

## Output

- Slack message body at the specified output path

## Resources

- `references/slack-post-template.md`: Template structure and past Slack messages for tone/format reference
