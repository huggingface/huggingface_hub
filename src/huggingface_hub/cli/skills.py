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
    # install the hf-cli skill for Claude (project-level, in current directory)
    hf skills add --claude

    # install for multiple assistants (project-level)
    hf skills add --claude --codex --opencode

    # install globally (user-level)
    hf skills add --claude --global

    # install to a custom directory
    hf skills add --dest=~/my-skills

    # overwrite an existing skill
    hf skills add --claude --force
"""

import os
import shutil
from pathlib import Path
from typing import Annotated, Optional

import typer

from huggingface_hub.utils import get_session

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

The Hugging Face Hub CLI tool `hf` is available. IMPORTANT: The `hf` command replaces the deprecated `huggingface_cli` command.

Use `hf --help` to view available functions. Note that auth commands are now all under `hf auth` e.g. `hf auth whoami`.
"""

# Central location for skills (shared across all agents)
CENTRAL_LOCAL = Path(".agents/skills")
CENTRAL_GLOBAL = Path("~/.agents/skills")

# Agent-specific directories that will contain symlinks to central location
GLOBAL_TARGETS = {
    "codex": Path("~/.codex/skills"),
    "claude": Path("~/.claude/skills"),
    "opencode": Path("~/.opencode/skills"),
}

LOCAL_TARGETS = {
    "codex": Path(".codex/skills"),
    "claude": Path(".claude/skills"),
    "opencode": Path(".opencode/skills"),
}

skills_cli = typer_factory(help="Manage skills for AI assistants.")


def _download(url: str) -> str:
    """Download text content from a URL."""
    response = get_session().get(url)
    response.raise_for_status()
    return response.text


def _install_to_central(central_path: Path, force: bool) -> Path:
    """Download and install the skill files into the central skills directory.

    Args:
        central_path: Path to the central skills directory (e.g., `.agents/skills/`)
        force: Whether to overwrite existing skill

    Returns:
        Path to the installed skill directory
    """
    central_path = central_path.expanduser().resolve()
    central_path.mkdir(parents=True, exist_ok=True)
    dest = central_path / DEFAULT_SKILL_ID

    if dest.exists():
        if dest.is_symlink():
            dest.unlink()
        elif not force:
            raise SystemExit(f"Skill already exists at {dest}.\nRe-run with --force to overwrite.")
        else:
            shutil.rmtree(dest)

    dest.mkdir()

    # SKILL.md – the main guide, prefixed with YAML metadata
    skill_content = _download(_SKILL_MD_URL)
    (dest / "SKILL.md").write_text(_SKILL_YAML_PREFIX + skill_content, encoding="utf-8")

    # references/cli.md – the full CLI reference
    ref_dir = dest / "references"
    ref_dir.mkdir()
    ref_content = _download(_REFERENCE_URL)
    (ref_dir / "cli.md").write_text(ref_content, encoding="utf-8")

    return dest


def _create_symlink(agent_skills_dir: Path, central_skill_path: Path, force: bool) -> Path:
    """Create a relative symlink from agent directory to the central skill location.

    Args:
        agent_skills_dir: Path to the agent's skills directory (e.g., `.claude/skills/`)
        central_skill_path: Absolute path to the central skill directory
        force: Whether to overwrite existing symlink/directory

    Returns:
        Path to the created symlink
    """
    agent_skills_dir = agent_skills_dir.expanduser().resolve()
    agent_skills_dir.mkdir(parents=True, exist_ok=True)
    link_path = agent_skills_dir / DEFAULT_SKILL_ID

    if link_path.exists() or link_path.is_symlink():
        if not force:
            raise SystemExit(f"Skill already exists at {link_path}.\nRe-run with --force to overwrite.")
        if link_path.is_symlink():
            link_path.unlink()
        elif link_path.is_dir():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()

    # Calculate relative path from agent skills dir to central skill path
    relative_target = os.path.relpath(central_skill_path, agent_skills_dir)
    link_path.symlink_to(relative_target)

    return link_path


@skills_cli.command("add")
def skills_add(
    claude: Annotated[bool, typer.Option("--claude", help="Install for Claude.")] = False,
    codex: Annotated[bool, typer.Option("--codex", help="Install for Codex.")] = False,
    opencode: Annotated[bool, typer.Option("--opencode", help="Install for OpenCode.")] = False,
    global_: Annotated[
        bool,
        typer.Option(
            "--global",
            "-g",
            help="Install globally (user-level) instead of in the current project directory.",
        ),
    ] = False,
    dest: Annotated[
        Optional[Path],
        typer.Option(
            help="Install into a custom destination (path to skills directory).",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing skills in the destination.",
        ),
    ] = False,
) -> None:
    """Download a skill and install it for an AI assistant.

    The skill is installed in a central location (.agents/skills/hf-cli/ for local,
    ~/.agents/skills/hf-cli/ for global) and relative symlinks are created from
    agent-specific directories (.claude/skills/, .codex/skills/, .opencode/skills/)
    pointing to the central location.
    """
    if not (claude or codex or opencode or dest):
        print("Pick a destination via --claude, --codex, --opencode, or --dest.")
        raise typer.Exit(code=1)

    # Determine which agent targets to create symlinks for
    targets_dict = GLOBAL_TARGETS if global_ else LOCAL_TARGETS
    agent_targets: list[Path] = []
    if claude:
        agent_targets.append(targets_dict["claude"])
    if codex:
        agent_targets.append(targets_dict["codex"])
    if opencode:
        agent_targets.append(targets_dict["opencode"])

    # Handle --dest option: install directly to custom destination (no symlink)
    if dest:
        dest_resolved = dest.expanduser().resolve()
        dest_resolved.mkdir(parents=True, exist_ok=True)
        skill_dest = dest_resolved / DEFAULT_SKILL_ID
        if skill_dest.exists() or skill_dest.is_symlink():
            if not force:
                raise SystemExit(f"Skill already exists at {skill_dest}.\nRe-run with --force to overwrite.")
            if skill_dest.is_symlink():
                skill_dest.unlink()
            elif skill_dest.is_dir():
                shutil.rmtree(skill_dest)
            else:
                skill_dest.unlink()
        # For custom dest, install directly (no central location / symlink)
        skill_dest.mkdir()
        skill_content = _download(_SKILL_MD_URL)
        (skill_dest / "SKILL.md").write_text(_SKILL_YAML_PREFIX + skill_content, encoding="utf-8")
        ref_dir = skill_dest / "references"
        ref_dir.mkdir()
        ref_content = _download(_REFERENCE_URL)
        (ref_dir / "cli.md").write_text(ref_content, encoding="utf-8")
        print(f"Installed '{DEFAULT_SKILL_ID}' to {skill_dest}")

    # For agent targets, install to central location and create symlinks
    if agent_targets:
        central_path = CENTRAL_GLOBAL if global_ else CENTRAL_LOCAL
        central_skill_path = _install_to_central(central_path, force)
        print(f"Installed '{DEFAULT_SKILL_ID}' to central location: {central_skill_path}")

        for agent_target in agent_targets:
            link_path = _create_symlink(agent_target, central_skill_path, force)
            print(f"Created symlink: {link_path} -> {central_skill_path}")
