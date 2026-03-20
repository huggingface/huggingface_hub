"""Internal helpers for marketplace-backed skill installation and updates."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlparse

from huggingface_hub.errors import CLIError
from huggingface_hub.utils import get_session


DEFAULT_SKILLS_REPO = "https://github.com/huggingface/skills"
MARKETPLACE_PATH = ".claude-plugin/marketplace.json"
MARKETPLACE_TIMEOUT = 10
SKILL_SOURCE_FILENAME = ".skill-source.json"
SKILL_SOURCE_SCHEMA_VERSION = 1

SkillSourceOrigin = Literal["remote", "local"]
SkillUpdateStatus = Literal[
    "dirty",
    "up_to_date",
    "update_available",
    "updated",
    "unmanaged",
    "invalid_metadata",
    "source_unreachable",
]


@dataclass(frozen=True)
class MarketplaceSkill:
    name: str
    description: str | None
    repo_url: str
    repo_ref: str | None
    repo_path: str
    source_url: str | None = None

    @property
    def install_dir_name(self) -> str:
        path = PurePosixPath(self.repo_path)
        if path.name.lower() == "skill.md":
            return path.parent.name or self.name
        return path.name or self.name


@dataclass(frozen=True)
class InstalledSkillSource:
    schema_version: int
    installed_via: str
    source_origin: SkillSourceOrigin
    repo_url: str
    repo_ref: str | None
    repo_path: str
    source_url: str | None
    installed_commit: str | None
    installed_path_oid: str | None
    installed_revision: str
    installed_at: str
    content_fingerprint: str


@dataclass(frozen=True)
class SkillUpdateInfo:
    name: str
    skill_dir: Path
    status: SkillUpdateStatus
    detail: str | None = None
    current_revision: str | None = None
    available_revision: str | None = None


def load_marketplace_skills(repo_url: str | None = None) -> list[MarketplaceSkill]:
    """Load marketplace skills from the default Hugging Face skills repository or a local override."""
    repo_url = repo_url or DEFAULT_SKILLS_REPO
    payload = _load_marketplace_payload(repo_url)
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
        description = plugin.get("description") if isinstance(plugin.get("description"), str) else None
        skills.append(
            MarketplaceSkill(
                name=name,
                description=description,
                repo_url=repo_url,
                repo_ref=None,
                repo_path=_normalize_repo_path(source),
                source_url=None,
            )
        )
    return skills


def get_marketplace_skill(selector: str, repo_url: str | None = None) -> MarketplaceSkill:
    """Resolve a marketplace skill by name."""
    repo_url = repo_url or DEFAULT_SKILLS_REPO
    skills = load_marketplace_skills(repo_url)
    selected = _select_marketplace_skill(skills, selector)
    if selected is None:
        raise CLIError(
            f"Skill '{selector}' not found in huggingface/skills. "
            "Try `hf skills add` to install `hf-cli` or use a known skill name."
        )
    return selected


def install_marketplace_skill(skill: MarketplaceSkill, destination_root: Path, force: bool = False) -> Path:
    """Install a marketplace skill into a local skills directory."""
    destination_root = destination_root.expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    install_dir = destination_root / skill.install_dir_name

    if install_dir.exists() and not force:
        raise FileExistsError(f"Skill already exists: {install_dir}")

    if install_dir.exists():
        with tempfile.TemporaryDirectory(dir=destination_root, prefix=f".{install_dir.name}.install-") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            staged_dir = tmp_dir / install_dir.name
            _populate_install_dir(skill=skill, install_dir=staged_dir)
            _atomic_replace_directory(existing_dir=install_dir, staged_dir=staged_dir)
        return install_dir

    try:
        _populate_install_dir(skill=skill, install_dir=install_dir)
    except Exception:
        if install_dir.exists():
            shutil.rmtree(install_dir)
        raise
    return install_dir


def check_for_updates(
    roots: list[Path],
    selector: str | None = None,
) -> list[SkillUpdateInfo]:
    """Check managed skill installs for newer upstream revisions."""
    updates = [_evaluate_update(skill_dir) for skill_dir in _iter_unique_skill_dirs(roots)]
    filtered = _filter_updates(updates, selector)
    if selector is not None and not filtered:
        raise CLIError(f"No installed skills match '{selector}'.")
    return filtered


def apply_updates(
    roots: list[Path],
    selector: str | None = None,
    force: bool = False,
) -> list[SkillUpdateInfo]:
    """Update managed skills in place, skipping dirty installs unless forced."""
    updates = check_for_updates(roots, selector)
    results: list[SkillUpdateInfo] = []
    for update in updates:
        results.append(_apply_single_update(update, force=force))
    return results


def compute_skill_content_fingerprint(skill_dir: Path) -> str:
    """Hash installed skill contents while ignoring the provenance sidecar."""
    digest = hashlib.sha256()
    root = skill_dir.resolve()
    sidecar_path = root / SKILL_SOURCE_FILENAME

    for path in sorted(root.rglob("*")):
        if path == sidecar_path or not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return f"sha256:{digest.hexdigest()}"


def read_installed_skill_source(skill_dir: Path) -> tuple[InstalledSkillSource | None, str | None]:
    """Read installed skill provenance metadata from the local sidecar file."""
    sidecar_path = skill_dir / SKILL_SOURCE_FILENAME
    if not sidecar_path.exists():
        return None, None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return None, "metadata root must be an object"
    try:
        return _parse_installed_skill_source(payload), None
    except ValueError as exc:
        return None, str(exc)


def write_installed_skill_source(skill_dir: Path, source: InstalledSkillSource) -> None:
    payload = {
        "schema_version": source.schema_version,
        "installed_via": source.installed_via,
        "source_origin": source.source_origin,
        "repo_url": source.repo_url,
        "repo_ref": source.repo_ref,
        "repo_path": source.repo_path,
        "source_url": source.source_url,
        "installed_commit": source.installed_commit,
        "installed_path_oid": source.installed_path_oid,
        "installed_revision": source.installed_revision,
        "installed_at": source.installed_at,
        "content_fingerprint": source.content_fingerprint,
    }
    (skill_dir / SKILL_SOURCE_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_marketplace_payload(repo_url: str) -> dict[str, Any]:
    if _is_local_repo(repo_url):
        marketplace_path = _resolve_local_repo_path(repo_url) / MARKETPLACE_PATH
        if not marketplace_path.is_file():
            raise CLIError(f"Marketplace file not found: {marketplace_path}")
        try:
            payload = json.loads(marketplace_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CLIError(f"Failed to parse marketplace file: {exc}") from exc
        if not isinstance(payload, dict):
            raise CLIError("Invalid marketplace payload: expected a JSON object.")
        return payload

    raw_url = _raw_github_url(repo_url, "main", MARKETPLACE_PATH)
    response = get_session().get(raw_url, follow_redirects=True, timeout=MARKETPLACE_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise CLIError("Invalid marketplace payload: expected a JSON object.")
    return payload


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


def _populate_install_dir(skill: MarketplaceSkill, install_dir: Path) -> None:
    metadata = _resolve_source_metadata(skill)
    install_dir.mkdir(parents=True, exist_ok=True)

    if metadata.source_origin == "local":
        _extract_local_git_path(
            repo_path=_resolve_local_repo_path(skill.repo_url),
            repo_ref=metadata.install_ref,
            source_path=skill.repo_path,
            install_dir=install_dir,
        )
    else:
        _extract_remote_github_path(
            repo_url=skill.repo_url,
            revision=metadata.installed_revision,
            source_path=skill.repo_path,
            install_dir=install_dir,
        )

    _validate_installed_skill_dir(install_dir)
    fingerprint = compute_skill_content_fingerprint(install_dir)
    write_installed_skill_source(
        install_dir,
        InstalledSkillSource(
            schema_version=SKILL_SOURCE_SCHEMA_VERSION,
            installed_via="marketplace",
            source_origin=metadata.source_origin,
            repo_url=skill.repo_url,
            repo_ref=skill.repo_ref,
            repo_path=skill.repo_path,
            source_url=skill.source_url,
            installed_commit=metadata.installed_commit,
            installed_path_oid=None,
            installed_revision=metadata.installed_revision,
            installed_at=_iso_utc_now(),
            content_fingerprint=fingerprint,
        ),
    )


def _validate_installed_skill_dir(skill_dir: Path) -> None:
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        raise RuntimeError(f"Installed skill is missing SKILL.md: {skill_file}")
    skill_file.read_text(encoding="utf-8")


@dataclass(frozen=True)
class _ResolvedSourceMetadata:
    source_origin: SkillSourceOrigin
    install_ref: str
    installed_commit: str | None
    installed_revision: str


def _resolve_source_metadata(skill: MarketplaceSkill) -> _ResolvedSourceMetadata:
    if _is_local_repo(skill.repo_url):
        repo_path = _resolve_local_repo_path(skill.repo_url)
        install_ref = skill.repo_ref or "HEAD"
        installed_commit = _git_stdout(repo_path, "rev-parse", install_ref)
        return _ResolvedSourceMetadata(
            source_origin="local",
            install_ref=install_ref,
            installed_commit=installed_commit,
            installed_revision=installed_commit,
        )

    installed_commit = _git_ls_remote(skill.repo_url, skill.repo_ref)
    return _ResolvedSourceMetadata(
        source_origin="remote",
        install_ref=installed_commit,
        installed_commit=installed_commit,
        installed_revision=installed_commit,
    )


def _extract_local_git_path(repo_path: Path, repo_ref: str, source_path: str, install_dir: Path) -> None:
    proc = subprocess.run(
        ["git", "-C", str(repo_path), "archive", "--format=tar", repo_ref, source_path],
        check=False,
        capture_output=True,
        text=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise FileNotFoundError(stderr or f"Path '{source_path}' not found in {repo_ref}.")
    _extract_tar_subpath(proc.stdout, source_path=source_path, install_dir=install_dir)


def _extract_remote_github_path(repo_url: str, revision: str, source_path: str, install_dir: Path) -> None:
    owner, repo = _parse_github_repo(repo_url)
    tarball_url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/{revision}"
    response = get_session().get(tarball_url, follow_redirects=True, timeout=MARKETPLACE_TIMEOUT)
    response.raise_for_status()
    _extract_tar_subpath(response.content, source_path=source_path, install_dir=install_dir)


def _extract_tar_subpath(tar_bytes: bytes, source_path: str, install_dir: Path) -> None:
    """Extract a skill subdirectory from either a git archive or a GitHub tarball.

    Local `git archive` paths start directly at `skills/<name>/...`, while GitHub tarballs
    include a leading `<repo>-<revision>/` directory. This helper accepts both layouts.
    """
    source_parts = PurePosixPath(source_path).parts
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as archive:
        members = archive.getmembers()
        matched = False
        for member in members:
            relative_parts = _member_relative_parts(member_name=member.name, source_parts=source_parts)
            if relative_parts is None:
                continue
            if not relative_parts:
                matched = True
                continue
            matched = True
            relative_path = Path(*relative_parts)
            if ".." in relative_path.parts:
                raise RuntimeError(f"Invalid path found in archive for {source_path}.")
            destination_path = install_dir / relative_path
            if member.isdir():
                destination_path.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Failed to extract {member.name}.")
            destination_path.write_bytes(extracted.read())
    if not matched:
        raise FileNotFoundError(f"Path '{source_path}' not found in source archive.")


def _member_relative_parts(member_name: str, source_parts: tuple[str, ...]) -> tuple[str, ...] | None:
    path_parts = PurePosixPath(member_name).parts
    if tuple(path_parts[: len(source_parts)]) == source_parts:
        return path_parts[len(source_parts) :]
    if len(path_parts) > len(source_parts) and tuple(path_parts[1 : 1 + len(source_parts)]) == source_parts:
        return path_parts[1 + len(source_parts) :]
    return None


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


def _evaluate_update(skill_dir: Path) -> SkillUpdateInfo:
    source, error = read_installed_skill_source(skill_dir)
    if source is None:
        return SkillUpdateInfo(
            name=skill_dir.name,
            skill_dir=skill_dir,
            status="invalid_metadata" if error is not None else "unmanaged",
            detail=error,
        )

    current_revision = source.installed_revision
    try:
        available_revision = _resolve_available_revision(source)
    except Exception as exc:
        return SkillUpdateInfo(
            name=skill_dir.name,
            skill_dir=skill_dir,
            status="source_unreachable",
            detail=str(exc),
            current_revision=current_revision,
        )

    fingerprint = compute_skill_content_fingerprint(skill_dir)
    if fingerprint != source.content_fingerprint:
        return SkillUpdateInfo(
            name=skill_dir.name,
            skill_dir=skill_dir,
            status="dirty",
            detail="local modifications detected",
            current_revision=current_revision,
            available_revision=available_revision,
        )

    if available_revision == current_revision:
        return SkillUpdateInfo(
            name=skill_dir.name,
            skill_dir=skill_dir,
            status="up_to_date",
            current_revision=current_revision,
            available_revision=available_revision,
        )

    return SkillUpdateInfo(
        name=skill_dir.name,
        skill_dir=skill_dir,
        status="update_available",
        detail="update available",
        current_revision=current_revision,
        available_revision=available_revision,
    )


def _apply_single_update(update: SkillUpdateInfo, *, force: bool) -> SkillUpdateInfo:
    if update.status in {"up_to_date", "unmanaged", "invalid_metadata", "source_unreachable"}:
        return update
    if update.status == "dirty" and not force:
        return update

    source, error = read_installed_skill_source(update.skill_dir)
    if source is None:
        detail = error or "missing source metadata"
        return SkillUpdateInfo(
            name=update.name,
            skill_dir=update.skill_dir,
            status="invalid_metadata",
            detail=detail,
            current_revision=update.current_revision,
            available_revision=update.available_revision,
        )

    skill = MarketplaceSkill(
        name=update.name,
        description=None,
        repo_url=source.repo_url,
        repo_ref=source.repo_ref,
        repo_path=source.repo_path,
        source_url=source.source_url,
    )
    try:
        install_marketplace_skill(skill, update.skill_dir.parent, force=True)
        refreshed = _evaluate_update(update.skill_dir)
    except Exception as exc:
        return SkillUpdateInfo(
            name=update.name,
            skill_dir=update.skill_dir,
            status="source_unreachable",
            detail=str(exc),
            current_revision=update.current_revision,
            available_revision=update.available_revision,
        )

    return SkillUpdateInfo(
        name=update.name,
        skill_dir=update.skill_dir,
        status="updated",
        detail="updated",
        current_revision=update.current_revision,
        available_revision=refreshed.current_revision,
    )


def _filter_updates(updates: list[SkillUpdateInfo], selector: str | None) -> list[SkillUpdateInfo]:
    if selector is None:
        return updates
    selector_lower = selector.strip().lower()
    return [update for update in updates if update.name.lower() == selector_lower]


def _resolve_available_revision(source: InstalledSkillSource) -> str:
    if source.source_origin == "local":
        repo_path = _resolve_local_repo_path(source.repo_url)
        return _git_stdout(repo_path, "rev-parse", source.repo_ref or "HEAD")
    return _git_ls_remote(source.repo_url, source.repo_ref)


def _parse_installed_skill_source(payload: dict[str, Any]) -> InstalledSkillSource:
    if payload.get("schema_version") != SKILL_SOURCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {payload.get('schema_version')}")
    repo_url = payload.get("repo_url")
    repo_path = payload.get("repo_path")
    source_origin = payload.get("source_origin")
    installed_via = payload.get("installed_via")
    installed_revision = payload.get("installed_revision")
    installed_at = payload.get("installed_at")

    if not isinstance(repo_url, str) or not repo_url:
        raise ValueError("missing repo_url")
    if not isinstance(repo_path, str) or not repo_path:
        raise ValueError("missing repo_path")
    if source_origin not in {"local", "remote"}:
        raise ValueError("invalid source_origin")
    if not isinstance(installed_via, str) or not installed_via:
        raise ValueError("missing installed_via")
    if not isinstance(installed_revision, str) or not installed_revision:
        raise ValueError("missing installed_revision")
    if not isinstance(installed_at, str) or not installed_at:
        raise ValueError("missing installed_at")

    repo_ref = payload.get("repo_ref")
    source_url = payload.get("source_url")
    installed_commit = payload.get("installed_commit")
    installed_path_oid = payload.get("installed_path_oid")
    content_fingerprint = payload.get("content_fingerprint")

    if repo_ref is not None and not isinstance(repo_ref, str):
        raise ValueError("invalid repo_ref")
    if source_url is not None and not isinstance(source_url, str):
        raise ValueError("invalid source_url")
    if installed_commit is not None and not isinstance(installed_commit, str):
        raise ValueError("invalid installed_commit")
    if installed_path_oid is not None and not isinstance(installed_path_oid, str):
        raise ValueError("invalid installed_path_oid")
    if not isinstance(content_fingerprint, str) or not content_fingerprint:
        raise ValueError("missing content_fingerprint")

    return InstalledSkillSource(
        schema_version=SKILL_SOURCE_SCHEMA_VERSION,
        installed_via=installed_via,
        source_origin=source_origin,
        repo_url=repo_url,
        repo_ref=repo_ref,
        repo_path=repo_path,
        source_url=source_url,
        installed_commit=installed_commit,
        installed_path_oid=installed_path_oid,
        installed_revision=installed_revision,
        installed_at=installed_at,
        content_fingerprint=content_fingerprint,
    )


def _is_local_repo(repo_url: str) -> bool:
    parsed = urlparse(repo_url)
    return parsed.scheme in {"", "file"}


def _resolve_local_repo_path(repo_url: str) -> Path:
    parsed = urlparse(repo_url)
    if parsed.scheme == "file":
        return Path(parsed.path).expanduser().resolve()
    return Path(repo_url).expanduser().resolve()


def _raw_github_url(repo_url: str, revision: str, path: str) -> str:
    owner, repo = _parse_github_repo(repo_url)
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{revision}/{path}"


def _parse_github_repo(repo_url: str) -> tuple[str, str]:
    parsed = urlparse(repo_url)
    if parsed.netloc not in {"github.com", "www.github.com"}:
        raise CLIError(f"Unsupported skills repository URL: {repo_url}")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        raise CLIError(f"Unsupported skills repository URL: {repo_url}")
    repo = parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return parts[0], repo


def _git_stdout(repo_path: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or proc.stdout.strip()
        raise CLIError(stderr or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def _git_ls_remote(repo_url: str, repo_ref: str | None) -> str:
    ref = repo_ref or "HEAD"
    proc = subprocess.run(
        ["git", "ls-remote", repo_url, ref],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or proc.stdout.strip()
        raise CLIError(stderr or f"Unable to resolve {ref} for {repo_url}")
    for line in proc.stdout.splitlines():
        parts = line.split()
        if parts:
            return parts[0]
    raise CLIError(f"Unable to resolve {ref} for {repo_url}")


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
