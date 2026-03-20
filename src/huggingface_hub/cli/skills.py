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
    # install the hf-cli skill in common .agents/skills directory (either in current directory or user-level)
    hf skills add
    hf skills add --global

    # install the hf-cli skill for Claude (project-level, in current directory)
    hf skills add --claude

    # install for Cursor (project-level, in current directory)
    hf skills add --cursor

    # install for multiple assistants (project-level)
    hf skills add --claude --codex --opencode --cursor

    # install globally (user-level)
    hf skills add --claude --global

    # install to a custom directory
    hf skills add --dest=~/my-skills

    # overwrite an existing skill
    hf skills add --claude --force
"""

import shutil
from pathlib import Path
from typing import Annotated, Optional

import typer
from click import Command, Context, Group
from typer.main import get_command

from huggingface_hub.errors import CLIError

from . import _skills
from ._cli_utils import typer_factory


DEFAULT_SKILL_ID = "hf-cli"

_SKILL_YAML_PREFIX = """\
---
name: hf-cli
description: "Hugging Face Hub CLI (`hf`) for downloading, uploading, and managing repositories, models, datasets, and Spaces on the Hugging Face Hub. Replaces now deprecated `huggingface-cli` command."
---

Install: `curl -LsSf https://hf.co/cli/install.sh | bash -s`.

The Hugging Face Hub CLI tool `hf` is available. IMPORTANT: The `hf` command replaces the deprecated `huggingface-cli` command.

Use `hf --help` to view available functions. Note that auth commands are now all under `hf auth` e.g. `hf auth whoami`.
"""

_SKILL_TIPS = """
## Tips

- Use `hf <command> --help` for full options, usage, and real-world examples
- Authenticate with `HF_TOKEN` env var (recommended) or with `--token`
"""

CENTRAL_LOCAL = Path(".agents/skills")
CENTRAL_GLOBAL = Path("~/.agents/skills")

GLOBAL_TARGETS = {
    "codex": Path("~/.codex/skills"),
    "claude": Path("~/.claude/skills"),
    "cursor": Path("~/.cursor/skills"),
    "opencode": Path("~/.config/opencode/skills"),
}

LOCAL_TARGETS = {
    "codex": Path(".codex/skills"),
    "claude": Path(".claude/skills"),
    "cursor": Path(".cursor/skills"),
    "opencode": Path(".opencode/skills"),
}
# Flags worth explaining in the common-options glossary. Self-explanatory flags
# (--namespace, --yes, --private, …) are omitted even if they appear frequently.
_COMMON_FLAG_ALLOWLIST = {"--token", "--quiet", "--type", "--format", "--revision"}

_COMMON_FLAG_HELP_OVERRIDES: dict[str, str] = {
    "--format": "Output format: `--format json` (or `--json`) or `--format table` (default).",
}

skills_cli = typer_factory(help="Manage skills for AI assistants.")


def _format_params(cmd: Command) -> str:
    """Format required params: positional as UPPER_CASE, options as ``--name TYPE``."""
    parts = []
    for p in cmd.params:
        if not p.required or p.human_readable_name == "--help":
            continue
        if p.name and p.name.startswith("_"):
            continue
        long_name = next((o for o in getattr(p, "opts", []) if o.startswith("--")), None)
        if long_name is not None:
            type_name = getattr(p.type, "name", "").upper() or "VALUE"
            parts.append(f"{long_name} {type_name}")
        elif p.name:
            parts.append(p.human_readable_name)
    return " ".join(parts)


def _collect_leaf_commands(group: Group, ctx: Context, path_parts: list[str]) -> list[tuple[list[str], Command]]:
    """Recursively walk a Click Group, returning (full_path_parts, cmd) for every leaf command."""
    leaves: list[tuple[list[str], Command]] = []
    sub_ctx = Context(group, parent=ctx, info_name=path_parts[-1])
    for name in group.list_commands(sub_ctx):
        cmd = group.get_command(sub_ctx, name)
        if cmd is None or cmd.hidden:
            continue
        child_path = [*path_parts, name]
        if isinstance(cmd, Group):
            leaves.extend(_collect_leaf_commands(cmd, sub_ctx, child_path))
        else:
            leaves.append((child_path, cmd))
    return leaves


def _iter_optional_params(cmd: Command):
    """Yield (param, long_name, short_name) for each optional, non-internal param."""
    for p in cmd.params:
        if p.required or p.human_readable_name == "--help":
            continue
        if p.name and p.name.startswith("_"):
            continue
        long_name = None
        short_name = None
        for opt in getattr(p, "opts", []):
            if opt.startswith("--"):
                long_name = long_name or opt
            elif opt.startswith("-"):
                short_name = opt
        if long_name:
            yield p, long_name, short_name


def _get_flag_names(cmd: Command, *, exclude: Optional[set[str]] = None) -> list[str]:
    """Return long-form flag names (--foo) for optional, non-internal params.

    Boolean flags are bare (``--dry-run``).  Value-taking options include a
    type hint (``--include TEXT``, ``--max-workers INTEGER``).
    """
    flags: list[str] = []
    for p, long_name, _short in _iter_optional_params(cmd):
        if exclude and long_name in exclude:
            continue
        if getattr(p, "is_flag", False):
            flags.append(long_name)
        else:
            type_name = getattr(p.type, "name", "").upper() or "VALUE"
            flags.append(f"{long_name} {type_name}")
    return flags


def _compute_common_flags(
    leaf_commands: list[tuple[list[str], Command]],
) -> dict[str, tuple[str, str]]:
    """Collect display info for flags in the allowlist."""
    flag_info: dict[str, tuple[str, str]] = {}

    for _path, cmd in leaf_commands:
        for p, long_name, short_name in _iter_optional_params(cmd):
            if long_name not in _COMMON_FLAG_ALLOWLIST:
                continue
            # Prefer the version with a short form (e.g. "-q / --quiet" over just "--quiet")
            if long_name not in flag_info or (short_name and " / " not in flag_info[long_name][0]):
                display = f"{short_name} / {long_name}" if short_name else long_name
                help_text = (getattr(p, "help", None) or "").split("\n")[0].strip()
                flag_info[long_name] = (display, help_text)

    return flag_info


def _render_leaf(path_parts: list[str], cmd: Command, *, common_flag_names: set[str]) -> str:
    """Render a single leaf command as a markdown list entry."""
    help_text = (cmd.help or "").split("\n")[0].strip()
    params = _format_params(cmd)
    parts = ["hf", *path_parts] + ([params] if params else [])
    entry = f"- `{' '.join(parts)}` — {help_text}"
    flags = _get_flag_names(cmd, exclude=common_flag_names)
    if flags:
        entry += f" `[{' '.join(flags)}]`"
    return entry


def build_skill_md() -> str:
    # Lazy import to avoid circular dependency (hf.py imports skills_cli from this module)
    from huggingface_hub import __version__
    from huggingface_hub.cli.hf import app

    click_app = get_command(app)
    ctx = Context(click_app, info_name="hf")

    top_level: list[tuple[list[str], Command]] = []
    groups: list[tuple[str, Group]] = []
    for name in sorted(click_app.list_commands(ctx)):  # type: ignore[attr-defined]
        cmd = click_app.get_command(ctx, name)  # type: ignore[attr-defined]
        if cmd is None or cmd.hidden:
            continue
        if isinstance(cmd, Group):
            groups.append((name, cmd))
        else:
            top_level.append(([name], cmd))

    group_leaves: list[tuple[str, list[tuple[list[str], Command]]]] = []
    all_leaf_commands: list[tuple[list[str], Command]] = list(top_level)
    for name, group in groups:
        leaves = _collect_leaf_commands(group, ctx, [name])
        group_leaves.append((name, leaves))
        all_leaf_commands.extend(leaves)

    common_flags = _compute_common_flags(all_leaf_commands)
    common_flag_names = set(common_flags)

    # wrap in list to widen list[LiteralString] -> list[str] for `ty`
    lines: list[str] = list(_SKILL_YAML_PREFIX.splitlines())
    lines.append("")
    lines.append(f"Generated with `huggingface_hub v{__version__}`. Run `hf skills add --force` to regenerate.")
    lines.append("")
    lines.append("## Commands")
    lines.append("")

    for path_parts, cmd in top_level:
        lines.append(_render_leaf(path_parts, cmd, common_flag_names=common_flag_names))

    groups_dict = dict(groups)
    for name, leaves in group_leaves:
        group_cmd = groups_dict[name]
        help_text = (group_cmd.help or "").split("\n")[0].strip()
        lines.append("")
        lines.append(f"### `hf {name}` — {help_text}")
        lines.append("")
        for path_parts, cmd in leaves:
            lines.append(_render_leaf(path_parts, cmd, common_flag_names=common_flag_names))

    if common_flags:
        lines.append("")
        lines.append("## Common options")
        lines.append("")
        for long_name, (display, help_text) in sorted(common_flags.items()):
            help_text = _COMMON_FLAG_HELP_OVERRIDES.get(long_name, help_text)
            if help_text:
                lines.append(f"- `{display}` — {help_text}")
            else:
                lines.append(f"- `{display}`")

    lines.extend(_SKILL_TIPS.splitlines())

    return "\n".join(lines)


def _remove_existing(path: Path, force: bool) -> None:
    """Remove existing file/directory/symlink if force is True, otherwise raise an error."""
    if not (path.exists() or path.is_symlink()):
        return
    if not force:
        raise SystemExit(f"Skill already exists at {path}.\nRe-run with --force to overwrite.")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _install_to(skills_dir: Path, skill_name: str, force: bool) -> Path:
    """Install a marketplace skill into a skills directory. Returns the installed path."""
    skill = _skills.get_marketplace_skill(skill_name)
    try:
        return _skills.install_marketplace_skill(skill, skills_dir, force=force)
    except FileExistsError as exc:
        raise SystemExit(f"{exc}\nRe-run with --force to overwrite.") from exc


def _create_symlink(agent_skills_dir: Path, skill_name: str, central_skill_path: Path, force: bool) -> Path:
    """Create a relative symlink from agent directory to the central skill location."""
    agent_skills_dir = agent_skills_dir.expanduser().resolve()
    agent_skills_dir.mkdir(parents=True, exist_ok=True)
    link_path = agent_skills_dir / skill_name

    _remove_existing(link_path, force)
    link_path.symlink_to(central_skill_path.relative_to(agent_skills_dir, walk_up=True))

    return link_path


def _resolve_update_roots(
    *,
    claude: bool,
    codex: bool,
    cursor: bool,
    opencode: bool,
    global_: bool,
    dest: Optional[Path],
) -> list[Path]:
    if dest is not None:
        if claude or codex or cursor or opencode or global_:
            raise CLIError("--dest cannot be combined with --claude, --codex, --cursor, --opencode, or --global.")
        return [dest.expanduser().resolve()]

    targets_dict = GLOBAL_TARGETS if global_ else LOCAL_TARGETS
    roots: list[Path] = [CENTRAL_GLOBAL if global_ else CENTRAL_LOCAL]
    if not any([claude, codex, cursor, opencode]):
        roots.extend(targets_dict.values())
    else:
        if claude:
            roots.append(targets_dict["claude"])
        if codex:
            roots.append(targets_dict["codex"])
        if cursor:
            roots.append(targets_dict["cursor"])
        if opencode:
            roots.append(targets_dict["opencode"])
    return [root.expanduser().resolve() for root in roots]


@skills_cli.command("preview")
def skills_preview() -> None:
    """Print the generated SKILL.md to stdout."""
    print(build_skill_md())


@skills_cli.command(
    "add",
    examples=[
        "hf skills add",
        "hf skills add --global",
        "hf skills add --claude --cursor",
        "hf skills add --codex --opencode --cursor --global",
    ],
)
def skills_add(
    name: Annotated[
        str,
        typer.Argument(help="Marketplace skill name.", show_default=False),
    ] = DEFAULT_SKILL_ID,
    claude: Annotated[bool, typer.Option("--claude", help="Install for Claude.")] = False,
    codex: Annotated[bool, typer.Option("--codex", help="Install for Codex.")] = False,
    cursor: Annotated[bool, typer.Option("--cursor", help="Install for Cursor.")] = False,
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

    Default location is in the current directory (.agents/skills) or user-level (~/.agents/skills).
    If custom agents are specified (e.g. --claude --codex --cursor --opencode, etc), the skill will be symlinked to the agent's skills directory.
    """
    try:
        if dest:
            if claude or codex or cursor or opencode or global_:
                print("--dest cannot be combined with --claude, --codex, --cursor, --opencode, or --global.")
                raise typer.Exit(code=1)
            skill_dest = _install_to(dest, name, force)
            print(f"Installed '{name}' to {skill_dest}")
            return

        # Install to central location
        central_path = CENTRAL_GLOBAL if global_ else CENTRAL_LOCAL
        central_skill_path = _install_to(central_path, name, force)
        print(f"Installed '{name}' to central location: {central_skill_path}")

        # Create symlinks in agent directories
        targets_dict = GLOBAL_TARGETS if global_ else LOCAL_TARGETS
        agent_targets: list[Path] = []
        if claude:
            agent_targets.append(targets_dict["claude"])
        if codex:
            agent_targets.append(targets_dict["codex"])
        if cursor:
            agent_targets.append(targets_dict["cursor"])
        if opencode:
            agent_targets.append(targets_dict["opencode"])

        for agent_target in agent_targets:
            link_path = _create_symlink(agent_target, name, central_skill_path, force)
            print(f"Created symlink: {link_path}")
    except CLIError as exc:
        print(str(exc))
        raise typer.Exit(code=1) from exc


@skills_cli.command(
    "update",
    examples=[
        "hf skills update",
        "hf skills update hf-cli",
        "hf skills update gradio --dest=~/my-skills",
        "hf skills update --claude --force",
    ],
)
def skills_update(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Optional installed skill name to update.", show_default=False),
    ] = None,
    claude: Annotated[bool, typer.Option("--claude", help="Update skills installed for Claude.")] = False,
    codex: Annotated[bool, typer.Option("--codex", help="Update skills installed for Codex.")] = False,
    cursor: Annotated[bool, typer.Option("--cursor", help="Update skills installed for Cursor.")] = False,
    opencode: Annotated[bool, typer.Option("--opencode", help="Update skills installed for OpenCode.")] = False,
    global_: Annotated[
        bool,
        typer.Option(
            "--global",
            "-g",
            help="Use global skills directories instead of the current project.",
        ),
    ] = False,
    dest: Annotated[
        Optional[Path],
        typer.Option(
            help="Update skills in a custom skills directory.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite local modifications when updating a skill.",
        ),
    ] = False,
) -> None:
    """Update installed marketplace-managed skills."""
    try:
        roots = _resolve_update_roots(
            claude=claude,
            codex=codex,
            cursor=cursor,
            opencode=opencode,
            global_=global_,
            dest=dest,
        )

        results = _skills.apply_updates(roots, selector=name, force=force)
        if not results:
            print("No installed skills found.")
            return

        for result in results:
            detail = f" ({result.detail})" if result.detail else ""
            print(f"{result.name}: {result.status}{detail}")
    except CLIError as exc:
        print(str(exc))
        raise typer.Exit(code=1) from exc
