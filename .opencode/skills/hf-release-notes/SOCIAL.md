---
name: hf-release-notes:social
description: Generate draft social media posts (LinkedIn and X) from release notes. Use when asked to create social media drafts for a huggingface_hub release.
---

# Social Media Drafts

## Overview

Generate draft social media posts for LinkedIn and X (Twitter) from existing release notes for a huggingface_hub release. The output is a set of `.txt` files — multiple drafts per platform with distinct tones — ready for the team to review, edit, and publish.

**Important:** The prompt will specify the release notes path and output directory. Write each draft as a separate `.txt` file using the naming convention below.

## Workflow

### 1. Read inputs

The prompt will specify:
- **Version**: The release version (e.g., `v1.8.0`)
- **Release notes path**: Path to the full release notes markdown file
- **Output directory**: Where to write the draft files

Read the release notes file to understand what's in the release.

### 2. Generate LinkedIn drafts

Generate **3 LinkedIn posts**, each 150–250 words, with a distinct angle:

#### `linkedin_professional_1.txt` — Technical / professional

Write as a developer relations expert. Focus on the technical value and what problems the new features solve for ML engineers and data scientists. Use short paragraphs. Mention concrete features, APIs, or CLI commands by name. Include 2–3 relevant hashtags at the end (e.g., `#MachineLearning #OpenSource #Python`).

#### `linkedin_community_2.txt` — Community / ecosystem

Write as a community manager celebrating an open-source release. Emphasize the collaborative nature: thank contributors (mention a few by handle if names appear in the notes), highlight the ecosystem impact (transformers, diffusers, datasets, sentence-transformers all depend on huggingface_hub), and invite people to try it out or contribute. Warm, inclusive tone. Include 2–3 hashtags.

#### `linkedin_tutorial_3.txt` — Tutorial / practical

Write as a technical writer. Pick the 1–2 most impactful new features and show a quick before/after or a mini code snippet. The goal is to make readers think "I need to try this right now." Include 2–3 hashtags.

### 3. Generate X (Twitter) drafts

Generate **3 X posts**. Each tweet must be under 280 characters. Separate tweets within a draft with `---` on its own line.

#### `x_announcement_1.txt` — Announcement / hype

One punchy main tweet + one follow-up. Lead with the single most exciting feature. Use an energetic tone. 1–2 emoji max. The follow-up should contain a `[LINK]` placeholder for the release notes URL.

#### `x_technical_2.txt` — Technical / developer

One main tweet + one follow-up with a code snippet or command example. Focus on DX improvements, new APIs, or performance gains. Mention concrete method names or CLI commands.

#### `x_thread_3.txt` — Thread / storytelling

A short thread of 3–5 tweets walking through the highlights. Number them `1/`, `2/`, etc. The first tweet should hook the reader. The last tweet should link to the release notes with a `[LINK]` placeholder.

### 4. Picture suggestions

Every draft file must end with a picture suggestion on the last line, in brackets:

```
[Picture: description of suggested image, screenshot, or graphic]
```

Examples:
- `[Picture: screenshot of the new CLI output with a dark terminal theme]`
- `[Picture: side-by-side code comparison showing the old vs new API]`
- `[Picture: infographic summarizing the top 3 features with icons]`
- `[Picture: collage of contributor avatars or a community celebration graphic]`
- `[Picture: short GIF of the new command in action in a terminal]`

### 5. Formatting rules

- LinkedIn: natural paragraphs, no markdown headers. 2–3 hashtags at end.
- X: tweets separated by `---`, each under 280 characters.
- Use `[LINK]` placeholder for the release notes URL (never hardcode a URL).
- No internal jargon — write for the broader ML/developer community.
- Mention "huggingface_hub" (the Python package name) and "Hugging Face Hub" (the platform).
- Do NOT fabricate features — only mention what's actually in the release notes.

### 6. Write output

Write each draft as a separate file in the output directory:
- `linkedin_professional_1.txt`
- `linkedin_community_2.txt`
- `linkedin_tutorial_3.txt`
- `x_announcement_1.txt`
- `x_technical_2.txt`
- `x_thread_3.txt`

Each file should contain ONLY the post text (including picture suggestion). No metadata, no headers, no extra formatting.

## Input

- Version string (e.g., `v1.8.0`)
- Path to release notes markdown file
- Output directory path

## Output

- 6 draft `.txt` files in the output directory
