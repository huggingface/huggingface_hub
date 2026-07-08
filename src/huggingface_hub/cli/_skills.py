"""Internal helpers for Hugging Face skill installation, updates, and SKILL.md generation."""

import json
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from click import Command, Context, Group

from huggingface_hub import __version__
from huggingface_hub._buckets import BucketFile
from huggingface_hub.errors import CLIError

from ..utils import disable_progress_bars
from ._cli_utils import _has_local_formatting_option, get_hf_api


DEFAULT_SKILL_ID = "hf-cli"
DEFAULT_SKILLS_BUCKET_ID = "huggingface/skills"
MARKETPLACE_PATH = "marketplace.json"
# Empty marker file dropped into managed skill installs so `hf skills update` knows
# to touch them and leave user-placed skill dirs alone. Filename is historical (used
# to be a JSON manifest with a revision); we keep it for backward compat with installs
# made by previous versions.
MANAGED_MARKER_FILENAME = ".hf-skill-manifest.json"

SkillUpdateStatus = Literal["up_to_date", "unmanaged", "source_unreachable"]


@dataclass(frozen=True)
class MarketplaceSkill:
    name: str
    repo_path: str
    description: str | None = None


@dataclass(frozen=True)
class SkillUpdateInfo:
    name: str
    skill_dir: Path
    status: SkillUpdateStatus
    detail: str | None = None


def add_skill(skill_name: str, destination_root: Path, force: bool = False) -> Path:
    """Resolve a skill by name and install it.

    The default ``hf-cli`` skill is generated locally from the installed CLI version
    (no network); other skills are downloaded from the marketplace bucket.
    """
    if skill_name.strip().lower() == DEFAULT_SKILL_ID:
        return _install_generated_skill(destination_root, force=force)
    api = get_hf_api()
    with disable_progress_bars():
        marketplace_skills = _load_marketplace_skills(api)
        skill = _select_marketplace_skill(marketplace_skills, skill_name)
        if skill is None:
            raise CLIError(
                f"Skill '{skill_name}' not found in {DEFAULT_SKILLS_BUCKET_ID}. "
                "Try `hf skills add` to install `hf-cli` or use a known skill name."
            )
        return _install_marketplace_skill(api, skill, destination_root, force=force)


def update_skills(roots: list[Path], selector: str | None = None) -> list[SkillUpdateInfo]:
    """Re-sync managed skill installs (``hf-cli`` is regenerated locally, the rest re-downloaded from the bucket)."""
    skill_dirs = _iter_unique_skill_dirs(roots)
    if selector is not None:
        selector_lower = selector.strip().lower()
        skill_dirs = [d for d in skill_dirs if d.name.lower() == selector_lower]
        if not skill_dirs:
            raise CLIError(f"No installed skill matches '{selector}'. Install it with `hf skills add {selector}`.")

    # `hf-cli` is regenerated locally, so only hit the marketplace when another managed skill needs it.
    needs_marketplace = any(
        d.name.lower() != DEFAULT_SKILL_ID and (d / MANAGED_MARKER_FILENAME).exists() for d in skill_dirs
    )
    api = None
    marketplace_skills: dict[str, MarketplaceSkill] = {}
    if needs_marketplace:
        api = get_hf_api()
        with disable_progress_bars():
            marketplace_skills = {skill.name.lower(): skill for skill in _load_marketplace_skills(api)}

    return [_apply_single_update(api, skill_dir, marketplace_skills) for skill_dir in skill_dirs]


def _load_marketplace_skills(api) -> list[MarketplaceSkill]:
    payload = _load_marketplace_payload(api)
    plugins = payload.get("plugins")
    if not isinstance(plugins, list):
        raise CLIError("Invalid marketplace payload: expected a top-level 'plugins' list.")

    skills: list[MarketplaceSkill] = []
    for plugin in plugins:
        if not isinstance(plugin, dict):
            continue
        name = plugin.get("name")
        source = plugin.get("source")
        if not isinstance(name, str) or not isinstance(source, str):
            continue
        description = plugin.get("description")
        skills.append(
            MarketplaceSkill(
                name=name,
                repo_path=_normalize_repo_path(source),
                description=description if isinstance(description, str) else None,
            )
        )
    return skills


def _install_marketplace_skill(api, skill: MarketplaceSkill, destination_root: Path, force: bool = False) -> Path:
    """Install a marketplace skill into a local skills directory."""

    def populate(install_dir: Path) -> None:
        install_dir.mkdir(parents=True, exist_ok=True)
        bucket_files = _list_skill_files(api, skill)
        _download_skill_files(api, skill, bucket_files, install_dir)
        _validate_installed_skill_dir(install_dir)
        (install_dir / MANAGED_MARKER_FILENAME).touch()

    return _install_skill(skill.name, destination_root, populate=populate, force=force)


def _install_generated_skill(destination_root: Path, force: bool = False) -> Path:
    """Install the `hf-cli` skill by generating SKILL.md from the installed CLI (no bucket download)."""
    content = build_skill_md()

    def populate(install_dir: Path) -> None:
        install_dir.mkdir(parents=True, exist_ok=True)
        (install_dir / "SKILL.md").write_text(content, encoding="utf-8")
        (install_dir / MANAGED_MARKER_FILENAME).touch()

    return _install_skill(DEFAULT_SKILL_ID, destination_root, populate=populate, force=force)


_VALID_SKILL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _install_skill(
    name: str,
    destination_root: Path,
    populate: Callable[[Path], None],
    force: bool = False,
) -> Path:
    """Install a skill into ``destination_root`` by calling ``populate(install_dir)`` to fill it.

    Used by both the marketplace install (populate = download from bucket) and the
    locally-generated install (populate = write content). When the install already
    exists and ``force`` is set, the new content is staged in a sibling tempdir and
    atomically swapped in, so the existing install stays intact if ``populate``
    fails halfway through.
    """
    # `name` may come from the remote marketplace payload and the install dir is removed on
    # reinstall: validate it as defense-in-depth against path traversal.
    if not _VALID_SKILL_NAME.fullmatch(name):
        raise CLIError(f"Invalid skill name '{name}'.")
    destination_root = destination_root.expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    install_dir = destination_root / name
    already_exists = install_dir.exists()

    if already_exists and not force:
        raise FileExistsError(f"Skill already exists: {install_dir}")

    if already_exists:
        with tempfile.TemporaryDirectory(dir=destination_root, prefix=f".{install_dir.name}.install-") as tmp_dir_str:
            staged_dir = Path(tmp_dir_str) / install_dir.name
            populate(staged_dir)
            _atomic_replace_directory(existing_dir=install_dir, staged_dir=staged_dir)
        return install_dir

    try:
        populate(install_dir)
    except Exception:
        if install_dir.exists():
            shutil.rmtree(install_dir)
        raise
    return install_dir


def _load_marketplace_payload(api) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = Path(tmp_dir) / "marketplace.json"
        api.download_bucket_files(
            DEFAULT_SKILLS_BUCKET_ID,
            [(MARKETPLACE_PATH, local_path)],
            raise_on_missing_files=True,
        )
        parsed = json.loads(local_path.read_text(encoding="utf-8"))

    if not isinstance(parsed, dict):
        raise CLIError("Invalid marketplace payload: expected a JSON object.")
    return parsed


def _select_marketplace_skill(skills: list[MarketplaceSkill], selector: str) -> MarketplaceSkill | None:
    selector_lower = selector.strip().lower()
    for skill in skills:
        if skill.name.lower() == selector_lower:
            return skill
    return None


def _normalize_repo_path(path: str) -> str:
    normalized = path.strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.strip("/")
    if not normalized:
        raise CLIError("Invalid marketplace entry: empty source path.")
    return normalized


def _validate_installed_skill_dir(skill_dir: Path) -> None:
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        raise RuntimeError(f"Installed skill is missing SKILL.md: {skill_file}")


def _list_skill_files(api, skill: MarketplaceSkill) -> list[BucketFile]:
    """List all files under `skill.repo_path` in the marketplace bucket."""
    prefix = skill.repo_path.rstrip("/")
    files: list[BucketFile] = [
        item
        for item in api.list_bucket_tree(DEFAULT_SKILLS_BUCKET_ID, prefix=prefix, recursive=True)
        if isinstance(item, BucketFile)
    ]
    if not files:
        raise FileNotFoundError(f"Path '{prefix}' not found in bucket '{DEFAULT_SKILLS_BUCKET_ID}'.")
    return files


def _download_skill_files(api, skill: MarketplaceSkill, files: list[BucketFile], install_dir: Path) -> None:
    """Download bucket files into `install_dir`."""
    prefix = skill.repo_path.rstrip("/")
    prefix_with_slash = f"{prefix}/"

    # `list_bucket_tree(prefix=...)` matches as a raw string prefix, so e.g. asking for
    # "skills/gradio" can also return "skills/gradio-tools/...". Filter on the trailing
    # slash to keep only files actually inside the directory, then strip it so files land
    # directly under `install_dir` preserving any nested structure.
    download_specs: list[tuple[str | BucketFile, str | Path]] = []
    for bucket_file in files:
        if not bucket_file.path.startswith(prefix_with_slash):
            continue
        relative = bucket_file.path[len(prefix_with_slash) :]
        local_file = install_dir.joinpath(*PurePosixPath(relative).parts)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        download_specs.append((bucket_file, local_file))

    if not download_specs:
        raise FileNotFoundError(f"No files found under '{prefix}' in bucket '{DEFAULT_SKILLS_BUCKET_ID}'.")

    api.download_bucket_files(DEFAULT_SKILLS_BUCKET_ID, download_specs)


def _atomic_replace_directory(existing_dir: Path, staged_dir: Path) -> None:
    backup_dir = staged_dir.parent / f"{existing_dir.name}.backup"
    try:
        existing_dir.rename(backup_dir)
        staged_dir.rename(existing_dir)
        shutil.rmtree(backup_dir)
    except Exception:
        if backup_dir.exists() and not existing_dir.exists():
            backup_dir.rename(existing_dir)
        raise


def _iter_unique_skill_dirs(roots: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    discovered: list[Path] = []
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.name.startswith("."):
                continue
            if not child.is_dir() and not child.is_symlink():
                continue
            resolved = child.resolve()
            if resolved in seen or not resolved.is_dir():
                continue
            seen.add(resolved)
            discovered.append(resolved)
    return discovered


def _apply_single_update(api, skill_dir: Path, marketplace_skills: dict[str, MarketplaceSkill]) -> SkillUpdateInfo:
    base = SkillUpdateInfo(name=skill_dir.name, skill_dir=skill_dir, status="unmanaged")

    if not (skill_dir / MANAGED_MARKER_FILENAME).exists():
        return base

    if skill_dir.name.lower() == DEFAULT_SKILL_ID:
        try:
            _install_generated_skill(skill_dir.parent, force=True)
        except Exception as exc:
            return replace(base, status="source_unreachable", detail=str(exc))
        return replace(base, status="up_to_date")

    skill = marketplace_skills.get(skill_dir.name.lower())
    if skill is None:
        return replace(
            base,
            status="source_unreachable",
            detail=f"Skill '{skill_dir.name}' is no longer available in {DEFAULT_SKILLS_BUCKET_ID}.",
        )

    try:
        _install_marketplace_skill(api, skill, skill_dir.parent, force=True)
    except Exception as exc:
        return replace(base, status="source_unreachable", detail=str(exc))

    return replace(base, status="up_to_date")


# --- `hf-cli` SKILL.md generation (used by `hf skills preview` and the local hf-cli install) ---

_SKILL_DESCRIPTION = (
    "Hugging Face Hub CLI (`hf`) for downloading, uploading, and managing"
    " models, datasets, spaces, buckets, repos, papers, jobs, and more on the Hugging Face Hub."
    " Use when: handling authentication;"
    " managing local cache;"
    " managing Hugging Face Buckets;"
    " running or scheduling jobs on Hugging Face infrastructure;"
    " managing Hugging Face repos;"
    " discussions and pull requests;"
    " browsing models, datasets and spaces;"
    " reading, searching, or browsing academic papers;"
    " managing collections;"
    " querying datasets;"
    " configuring spaces;"
    " setting up webhooks;"
    " or deploying and managing HF Inference Endpoints."
    " Make sure to use this skill whenever the user mentions"
    " 'hf', 'huggingface', 'Hugging Face', 'huggingface-cli', or 'hugging face cli',"
    " or wants to do anything related to the Hugging Face ecosystem and to AI and ML in general."
    " Also use for cloud storage needs like training checkpoints, data pipelines, or agent traces."
    " Use even if the user doesn't explicitly ask for a CLI command."
    " Replaces the deprecated `huggingface-cli`."
)

_SKILL_YAML_PREFIX = f"""\
---
name: hf-cli
description: "{_SKILL_DESCRIPTION}"
---

Install: `curl -LsSf https://hf.co/cli/install.sh | bash -s`.

The Hugging Face Hub CLI tool `hf` is available. IMPORTANT: The `hf` command replaces the deprecated `huggingface-cli` command.

Use `hf --help` to view available functions. Note that auth commands are now all under `hf auth` e.g. `hf auth whoami`.
"""

_SKILL_TIPS = """
## Mounting repos as local filesystems

To mount Hub repositories or buckets as local filesystems — no download, no copy, no waiting — use `hf-mount`. Files are fetched on demand. GitHub: https://github.com/huggingface/hf-mount

Install: `curl -fsSL https://raw.githubusercontent.com/huggingface/hf-mount/main/install.sh | sh`

Some command examples:
- `hf-mount start repo openai-community/gpt2 /tmp/gpt2` — mount a repo (read-only)
- `hf-mount start --hf-token $HF_TOKEN bucket myuser/my-bucket /tmp/data` — mount a bucket (read-write)
- `hf-mount status` / `hf-mount stop /tmp/data` — list or unmount

## Tips

- Use `hf <command> --help` for full options, descriptions, usage, and real-world examples
- Authenticate with `HF_TOKEN` env var (recommended) or with `--token`
- Update the CLI with `hf update` (uses the correct command for the detected install method)
"""

# Flags worth explaining in the common-options glossary. Self-explanatory flags
# (--namespace, --yes, --private, …) are omitted even if they appear frequently.
_COMMON_FLAG_ALLOWLIST = {"--token", "--quiet", "--type", "--format", "--revision"}
# Keep token out of inline command signatures to encourage env based auth.
_INLINE_FLAG_EXCLUDE = {"--token"}

_COMMON_FLAG_HELP_OVERRIDES: dict[str, str] = {
    "--format": "Output format: `--format json` (or `--json`) or `--format table` (default).",
    "--token": "Use a User Access Token. Prefer setting `HF_TOKEN` env var instead of passing `--token`.",
}

# Global formatting flags injected into the skill markdown for commands that
# accept them. They aren't real click params on the command (they're consumed
# globally — see ``_consume_format_flags_for_leaf`` in ``_cli_utils.py``) so we
# add them synthetically here.
_GLOBAL_FORMAT_INLINE_FLAGS = ["--format [auto|human|agent|json|quiet]"]
_GLOBAL_COMMON_FLAGS: dict[str, tuple[str, str]] = {
    "--format": ("--format", "Output format."),
    "--quiet": ("-q / --quiet", "Quiet output (one ID per line)."),
}


def _type_hint(param) -> str:
    """Value hint for an option: enum choices inline as ``[a|b|c]``, otherwise the TYPE name.

    e.g. `--sort [downloads|likes|trending_score]` instead of `--sort CHOICE`.
    """
    choices = getattr(param.type, "choices", None)
    if choices:
        return "[" + "|".join(str(c) for c in choices) + "]"
    return getattr(param.type, "name", "").upper() or "VALUE"


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
            type_name = _type_hint(p)
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


def _accepts_global_format_flags(cmd: Command) -> bool:
    """Return True if the leaf command accepts the global '--format' / '--json' / '-q' flags."""
    if cmd.context_settings.get("ignore_unknown_options"):
        return False
    return not _has_local_formatting_option(cmd)


def _get_flag_names(cmd: Command, *, exclude: set[str] | None = None) -> list[str]:
    """Return long-form flag names (--foo) for optional, non-internal params.

    Boolean flags are bare ('--dry-run').  Value-taking options include a type hint ('--include TEXT', '--max-workers INTEGER').
    Synthetic global formatting flags are appended for commands that accept them.
    """
    flags: list[str] = []
    for p, long_name, _short in _iter_optional_params(cmd):
        if exclude and long_name in exclude:
            continue
        if getattr(p, "is_flag", False):
            flags.append(long_name)
        else:
            type_name = _type_hint(p)
            flags.append(f"{long_name} {type_name}")
    if _accepts_global_format_flags(cmd):
        flags.extend(flag for flag in _GLOBAL_FORMAT_INLINE_FLAGS if not (exclude and flag.split()[0] in exclude))
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

    # Inject the global formatting flags as common flags whenever any leaf
    # command accepts them (the vast majority do).
    if any(_accepts_global_format_flags(cmd) for _path, cmd in leaf_commands):
        for long_name, entry in _GLOBAL_COMMON_FLAGS.items():
            flag_info.setdefault(long_name, entry)

    return flag_info


def _render_leaf(path_parts: list[str], cmd: Command) -> str:
    """Render a single leaf command as a markdown list entry."""
    help_text = (cmd.help or "").split("\n")[0].strip()
    params = _format_params(cmd)
    parts = ["hf", *path_parts] + ([params] if params else [])
    entry = f"- `{' '.join(parts)}` — {help_text}"
    flags = _get_flag_names(cmd, exclude=_INLINE_FLAG_EXCLUDE)
    if flags:
        entry += f" `[{' '.join(flags)}]`"
    return entry


def build_skill_md() -> str:
    # Deferred import to avoid a circular import: `hf.py` aggregates all command modules
    # (including `skills.py`, which imports this module) into the `app` walked here.
    from .hf import app

    click_app = app  # the app is already a click.Group
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

    # wrap in list to widen list[LiteralString] -> list[str] for `ty`
    lines: list[str] = list(_SKILL_YAML_PREFIX.splitlines())
    lines.append("")
    lines.append(f"Generated with `huggingface_hub v{__version__}`. Run `hf skills add --force` to regenerate.")
    lines.append("")
    lines.append("## Commands")
    lines.append("")

    for path_parts, cmd in top_level:
        lines.append(_render_leaf(path_parts, cmd))

    groups_dict = dict(groups)
    for name, leaves in group_leaves:
        group_cmd = groups_dict[name]
        help_text = (group_cmd.help or "").split("\n")[0].strip()
        lines.append("")
        lines.append(f"### `hf {name}` — {help_text}")
        lines.append("")
        for path_parts, cmd in leaves:
            lines.append(_render_leaf(path_parts, cmd))

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
