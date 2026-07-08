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
"""Contains commands to manage skills for AI assistants."""

import os
import shutil
from pathlib import Path
from typing import Annotated

from huggingface_hub.errors import CLIError

from ..utils import disable_progress_bars
from . import _skills
from ._cli_utils import TokenOpt, get_hf_api, typer_factory
from ._framework import Argument, Option
from ._output import out
from ._skills import DEFAULT_SKILL_ID, build_skill_md


CENTRAL_LOCAL = Path(".agents/skills")
CENTRAL_GLOBAL = Path("~/.agents/skills")
CLAUDE_LOCAL = Path(".claude/skills")
CLAUDE_GLOBAL = Path("~/.claude/skills")
skills_cli = typer_factory(help="Manage skills for AI assistants.")


def _remove_existing(path: Path, force: bool) -> None:
    """Remove existing file/directory/symlink if force is True, otherwise raise an error."""
    if not (path.exists() or path.is_symlink()):
        return
    if not force:
        raise CLIError(f"Skill already exists at {path}.\nRe-run with --force to overwrite.")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _install_to(skills_dir: Path, skill_name: str, force: bool) -> Path:
    """Install a marketplace skill into a skills directory. Returns the installed path."""
    try:
        return _skills.add_skill(skill_name, skills_dir, force=force)
    except FileExistsError as exc:
        raise CLIError(f"{exc}\nRe-run with --force to overwrite.") from exc


def _create_symlink(agent_skills_dir: Path, skill_name: str, central_skill_path: Path, force: bool) -> Path:
    """Create a relative symlink from agent directory to the central skill location."""
    agent_skills_dir = agent_skills_dir.expanduser().resolve()
    agent_skills_dir.mkdir(parents=True, exist_ok=True)
    link_path = agent_skills_dir / skill_name

    _remove_existing(link_path, force)
    link_path.symlink_to(os.path.relpath(central_skill_path, agent_skills_dir))

    return link_path


def _resolve_update_roots(
    *,
    claude: bool,
    global_: bool,
    dest: Path | None,
) -> list[Path]:
    if dest is not None:
        if claude or global_:
            raise CLIError("--dest cannot be combined with --claude or --global.")
        return [dest.expanduser().resolve()]

    roots: list[Path] = [CENTRAL_GLOBAL if global_ else CENTRAL_LOCAL]
    if claude:
        roots.append(CLAUDE_GLOBAL if global_ else CLAUDE_LOCAL)
    return [root.expanduser().resolve() for root in roots]


@skills_cli.command("preview")
def skills_preview() -> None:
    """Print the generated `hf-cli` SKILL.md to stdout."""
    print(build_skill_md())


@skills_cli.command(
    "list | ls",
    examples=[
        "hf skills list",
        "hf skills list --format json",
    ],
)
def skills_list(
    token: TokenOpt = None,
) -> None:
    """List available skills from the Hugging Face marketplace."""
    install_locations: list[tuple[str, Path]] = [
        ("project", CENTRAL_LOCAL),
        ("project (claude)", CLAUDE_LOCAL),
        ("global", CENTRAL_GLOBAL),
        ("global (claude)", CLAUDE_GLOBAL),
    ]
    installed: dict[str, set[str]] = {}
    for label, root in install_locations:
        for skill_dir in _skills._iter_unique_skill_dirs([root]):
            installed.setdefault(skill_dir.name.lower(), set()).add(label)

    api = get_hf_api(token=token)
    with disable_progress_bars():
        skills = _skills._load_marketplace_skills(api)
    results = [
        {
            "name": skill.name,
            "description": skill.description or "",
            **{
                label: "yes" if label in installed.get(skill.name.lower(), set()) else ""
                for label, _ in install_locations
            },
        }
        for skill in skills
    ]
    out.table(
        results,
        id_key="name",
        alignments={"project": "right", "global": "right", "project (claude)": "right", "global (claude)": "right"},
    )


@skills_cli.command(
    "add",
    examples=[
        "hf skills add",
        "hf skills add huggingface-gradio --dest=~/my-skills",
        "hf skills add --global",
        "hf skills add --claude",
        "hf skills add huggingface-gradio --claude --global",
    ],
)
def skills_add(
    name: Annotated[
        str,
        Argument(help="Marketplace skill name.", show_default=False),
    ] = DEFAULT_SKILL_ID,
    claude: Annotated[bool, Option("--claude", help="Install for Claude.")] = False,
    global_: Annotated[
        bool,
        Option(
            "--global",
            "-g",
            help="Install globally (user-level) instead of in the current project directory.",
        ),
    ] = False,
    dest: Annotated[
        Path | None,
        Option(
            help="Install into a custom destination (path to skills directory).",
        ),
    ] = None,
    force: Annotated[
        bool,
        Option(
            "--force",
            help="Overwrite existing skills in the destination.",
        ),
    ] = False,
) -> None:
    """Install a Hugging Face skill for an AI assistant.

    The default `hf-cli` skill is generated locally from the installed CLI version;
    other skills are downloaded from the Hugging Face marketplace.
    Default location is in the current directory (.agents/skills) or user-level (~/.agents/skills).
    If `--claude` is specified, the skill is also symlinked into Claude's legacy skills directory.
    """
    if dest is not None:
        if claude or global_:
            raise CLIError("--dest cannot be combined with --claude or --global.")
        skill_dest = _install_to(dest, name, force)
        print(f"Installed '{name}' to {skill_dest}")
        return

    # Install to central location
    central_path = CENTRAL_GLOBAL if global_ else CENTRAL_LOCAL
    central_skill_path = _install_to(central_path, name, force)
    print(f"Installed '{name}' to central location: {central_skill_path}")

    if claude:
        agent_target = CLAUDE_GLOBAL if global_ else CLAUDE_LOCAL
        link_path = _create_symlink(agent_target, name, central_skill_path, force)
        print(f"Created symlink: {link_path}")


@skills_cli.command(
    "update",
    examples=[
        "hf skills update",
        "hf skills update hf-cli",
        "hf skills update huggingface-gradio --dest=~/my-skills",
        "hf skills update --claude",
    ],
)
def skills_update(
    name: Annotated[
        str | None,
        Argument(help="Optional installed skill name to update.", show_default=False),
    ] = None,
    claude: Annotated[bool, Option("--claude", help="Update skills installed for Claude.")] = False,
    global_: Annotated[
        bool,
        Option(
            "--global",
            "-g",
            help="Use global skills directories instead of the current project.",
        ),
    ] = False,
    dest: Annotated[
        Path | None,
        Option(
            help="Update skills in a custom skills directory.",
        ),
    ] = None,
) -> None:
    """Update installed Hugging Face marketplace skills."""
    roots = _resolve_update_roots(claude=claude, global_=global_, dest=dest)

    results = _skills.update_skills(roots, selector=name)
    if not results:
        print("No installed skills found.")
        return

    for result in results:
        detail = f" ({result.detail})" if result.detail else ""
        print(f"{result.name}: {result.status}{detail}")
