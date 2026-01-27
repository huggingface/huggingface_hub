# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains commands to manage skills for AI assistants.

Usage:
    # install the hf-cli skill for Claude
    hf skills add --claude

    # install for multiple assistants
    hf skills add --claude --codex --opencode

    # install into the current project directory (.claude/skills/)
    hf skills add --local

    # install to a custom directory
    hf skills add --dest=~/my-skills

    # overwrite an existing skill
    hf skills add --claude --force
"""

import shutil
from pathlib import Path
from typing import Annotated, Optional

import typer

from ._cli_utils import typer_factory


DEFAULT_SKILL_ID = "hf-cli"

_GITHUB_RAW_BASE = "https://raw.githubusercontent.com/huggingface/huggingface_hub/main/docs/source/en"
_SKILL_MD_URL = f"{_GITHUB_RAW_BASE}/guides/cli.md"
_REFERENCE_URL = f"{_GITHUB_RAW_BASE}/package_reference/cli.md"

_SKILL_YAML_PREFIX = """\
---
name: hf-cli
description: >
  Hugging Face Hub CLI (`hf`) for downloading, uploading, and managing
  repositories, models, datasets, and Spaces on the Hugging Face Hub.
---

"""

TARGETS = {
    "codex": Path("~/.codex/skills"),
    "claude": Path("~/.claude/skills"),
    "opencode": Path("~/.config/opencode/skills"),
}

skills_cli = typer_factory(help="Manage skills for AI assistants.")


def _download(url: str) -> str:
    """Download text content from a URL."""
    from huggingface_hub.utils import get_session

    response = get_session().get(url)
    response.raise_for_status()
    return response.text


def _install_to(target: Path, skill_name: str, force: bool) -> None:
    """Download and install the skill files into a target skills directory."""
    target = target.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    dest = target / skill_name

    if dest.exists():
        if not force:
            raise SystemExit(f"Skill already exists at {dest}.\nRe-run with --force to overwrite.")
        shutil.rmtree(dest)

    dest.mkdir()

    # SKILL.md – the main guide, prefixed with YAML metadata
    skill_content = _download(_SKILL_MD_URL)
    (dest / "SKILL.md").write_text(_SKILL_YAML_PREFIX + skill_content)

    # references/cli.md – the full CLI reference
    ref_dir = dest / "references"
    ref_dir.mkdir()
    ref_content = _download(_REFERENCE_URL)
    (ref_dir / "cli.md").write_text(ref_content)


@skills_cli.command("add")
def skills_add(
    skill_id: Annotated[
        str,
        typer.Argument(help="The skill to install."),
    ] = DEFAULT_SKILL_ID,
    claude: Annotated[
        bool,
        typer.Option("--claude", help="Install for Claude."),
    ] = False,
    codex: Annotated[
        bool,
        typer.Option("--codex", help="Install for Codex."),
    ] = False,
    opencode: Annotated[
        bool,
        typer.Option("--opencode", help="Install for OpenCode."),
    ] = False,
    local: Annotated[
        bool,
        typer.Option("--local", help="Install into the current directory (.claude/skills/)."),
    ] = False,
    dest: Annotated[
        Optional[Path],
        typer.Option(help="Install into a custom destination (path to skills directory)."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing skills in the destination."),
    ] = False,
) -> None:
    """Download a skill and install it for an AI assistant."""
    if skill_id != DEFAULT_SKILL_ID:
        print(f"For now, the only supported skill is '{DEFAULT_SKILL_ID}' (which is the default).")
        raise typer.Exit(code=1)

    if not (claude or codex or opencode or local or dest):
        print("Pick a destination via --claude, --codex, --opencode, --local, or --dest.")
        raise typer.Exit(code=1)

    targets: list[Path] = []
    if claude:
        targets.append(TARGETS["claude"])
    if codex:
        targets.append(TARGETS["codex"])
    if opencode:
        targets.append(TARGETS["opencode"])
    if local:
        targets.append(Path(".claude/skills"))
    if dest:
        targets.append(dest)

    for target in targets:
        _install_to(target, skill_id, force)
        installed_path = (target / skill_id).expanduser().resolve()
        print(f"Installed '{skill_id}' to {installed_path}")
