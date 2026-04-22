---
name: hf-release-notes:social
description: Generate draft social media posts (LinkedIn and X) from drafted release notes. Use when asked to create social media drafts for a huggingface_hub release.
---

# Social Media Drafts

## Overview

Generate draft social media posts for LinkedIn and X (Twitter) from existing release notes. Each platform gets multiple drafts with different tones and angles, ready for the team to pick, edit, and post.

## Workflow

### 1. Read inputs

The prompt will specify:
- **Version**: The release version (e.g., `v1.8.0`)
- **Release notes path**: Path to the full release notes markdown file
- **Output directory**: Where to write the drafts

Read the release notes file first to understand what's in the release.

### 2. Generate drafts

For each platform, generate multiple posts with distinct angles:

**LinkedIn** (150-250 words each):
- `professional`: Technical value, what problems these features solve for ML engineers
- `community`: Celebrate contributors, ecosystem impact, invite people to try it
- `tutorial`: Pick top 1-2 features, show a quick code snippet or before/after

**X / Twitter** (280 chars per tweet):
- `announcement`: Punchy, energetic, lead with the #1 feature. Main tweet + follow-up.
- `technical`: Developer-focused, mention specific APIs/commands. Main tweet + code follow-up.
- `thread`: 3-5 tweet thread walking through highlights. Numbered 1/, 2/, etc.

### 3. Formatting rules

- LinkedIn posts: natural paragraphs, 2-3 hashtags at end
- X posts: tweets separated by `---`, each under 280 characters
- Every draft ends with a picture suggestion in brackets:
  `[Picture: description of suggested image/screenshot]`
- Use `[LINK]` placeholder for the release notes URL
- No internal jargon — write for the broader ML/dev community

### 4. Write output

Write each draft to the output directory as:
- `linkedin_<tone>_<number>.txt`
- `x_<tone>_<number>.txt`

## Input

- Version string (e.g., "v1.8.0")
- Path to release notes markdown file
- Output directory

## Output

- 6 draft text files (3 LinkedIn + 3 X)
