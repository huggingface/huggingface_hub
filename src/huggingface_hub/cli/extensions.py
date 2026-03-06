# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Contains helper utilities for hf CLI extensions."""

import errno
import json
import os
import re
import shutil
import subprocess
import venv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from huggingface_hub.errors import CLIError
from huggingface_hub.utils import get_session, tabulate

from ._cli_utils import typer_factory


DEFAULT_EXTENSION_OWNER = "huggingface"
EXTENSIONS_ROOT = Path("~/.local/share/hf/extensions")
MANIFEST_FILENAME = "manifest.json"
EXTENSIONS_HELP = (
    "Manage hf CLI extensions.\n\n"
    "Security Warning: extensions are third-party executables or Python packages. "
    "Install only from sources you trust."
)
extensions_cli = typer_factory(help=EXTENSIONS_HELP)
_EXTENSIONS_DEFAULT_BRANCH = "main"  # Fallback when the GitHub API is unreachable.
_EXTENSIONS_DOWNLOAD_TIMEOUT = 10
_EXTENSIONS_PIP_INSTALL_TIMEOUT = 300


@dataclass
class ExtensionManifest:
    owner: str
    repo: str
    repo_id: str
    short_name: str
    executable_name: str
    executable_path: str
    type: str  # "binary" or "python"
    installed_at: str
    source: str
    description: str = ""


@extensions_cli.command(
    "install",
    examples=[
        "hf extensions install hf-claude",
        "hf extensions install hanouticelina/hf-claude",
        "hf extensions install alvarobartt/hf-mem",
    ],
)
def extension_install(
    ctx: typer.Context,
    repo_id: Annotated[
        str,
        typer.Argument(help="GitHub extension repository in `[OWNER/]hf-<name>` format."),
    ],
    force: Annotated[bool, typer.Option("--force", help="Overwrite if already installed.")] = False,
) -> None:
    """Install an extension from a public GitHub repository.

    Security warning: this installs a third-party executable or Python package.
    Install only from sources you trust.
    """
    owner, repo_name, short_name = _normalize_repo_id(repo_id)
    root_ctx = ctx.find_root()
    reserved_commands = set(getattr(root_ctx.command, "commands", {}).keys())
    if short_name in reserved_commands:
        raise CLIError(
            f"Cannot install extension '{short_name}' because it conflicts with an existing `hf {short_name}` command."
        )

    extension_dir = _get_extension_dir(short_name)
    extension_exists = extension_dir.exists()
    if extension_exists and not force:
        raise CLIError(f"Extension '{short_name}' is already installed. Use --force to overwrite.")

    branch = _resolve_github_default_branch(owner=owner, repo_name=repo_name)

    if extension_exists:
        shutil.rmtree(extension_dir)

    binary_manifest = _install_binary_extension(
        owner=owner,
        repo_name=repo_name,
        short_name=short_name,
        extension_dir=extension_dir,
        branch=branch,
    )
    if binary_manifest is None:
        _install_python_extension(
            owner=owner,
            repo_name=repo_name,
            short_name=short_name,
            extension_dir=extension_dir,
            branch=branch,
        )

    print(f"Installed extension '{owner}/{repo_name}'.")
    print(f"Run it with: hf {short_name}")
    print(f"Or with: hf extensions exec {short_name}")


@extensions_cli.command(
    "exec",
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False, "ignore_unknown_options": True},
    examples=[
        "hf extensions exec claude -- --help",
        "hf extensions exec claude --model zai-org/GLM-5",
    ],
)
def extension_exec(
    ctx: typer.Context,
    name: Annotated[
        str,
        typer.Argument(help="Extension name (with or without `hf-` prefix)."),
    ],
) -> None:
    """Execute an installed extension."""
    short_name = _normalize_extension_name(name)
    executable_path = _resolve_installed_executable_path(short_name)

    if not executable_path.is_file():
        raise CLIError(f"Extension '{short_name}' is not installed.")

    exit_code = _execute_extension_binary(executable_path=executable_path, args=list(ctx.args))
    raise typer.Exit(code=exit_code)


@extensions_cli.command("list", examples=["hf extensions list"])
def extension_list() -> None:
    """List installed extension commands."""
    root_dir = _get_extensions_root()
    if not root_dir.is_dir():
        print("No extensions installed.")
        return

    rows = []
    for extension_dir in sorted(root_dir.iterdir()):
        if not extension_dir.is_dir() or not extension_dir.name.startswith("hf-"):
            continue

        short_name = extension_dir.name[3:]
        data = _read_local_manifest(extension_dir)
        rows.append(
            [
                f"hf {short_name}",
                str(data.get("repo_id", "")),
                str(data.get("type", "")),
                str(data.get("installed_at", "")),
            ]
        )

    if not rows:
        print("No extensions installed.")
        return

    print(tabulate(rows, headers=["COMMAND", "REPOSITORY", "TYPE", "INSTALLED_AT"]))  # type: ignore[arg-type]


@extensions_cli.command("remove", examples=["hf extensions remove claude"])
def extension_remove(
    name: Annotated[
        str,
        typer.Argument(help="Extension name to remove (with or without `hf-` prefix)."),
    ],
) -> None:
    """Remove an installed extension."""
    short_name = _normalize_extension_name(name)
    extension_dir = _get_extension_dir(short_name)

    if not extension_dir.is_dir():
        raise CLIError(f"Extension '{short_name}' is not installed.")

    shutil.rmtree(extension_dir)
    print(f"Removed extension '{short_name}'.")


### HELPER FUNCTIONS


def _list_installed_extensions_for_help() -> list[tuple[str, str]]:
    root_dir = EXTENSIONS_ROOT.expanduser()
    if not root_dir.is_dir():
        return []
    entries = []
    for extension_dir in sorted(root_dir.iterdir()):
        if not extension_dir.is_dir() or not extension_dir.name.startswith("hf-"):
            continue
        short_name = extension_dir.name[3:]
        data = _read_local_manifest(extension_dir)
        description = data.get("description", "")
        repo_id = data.get("repo_id", "")
        tag = f" [extension {repo_id}]" if isinstance(repo_id, str) and repo_id else " [extension]"
        help_text = f"{description}{tag}" if isinstance(description, str) and description else tag.lstrip()
        entries.append((short_name, help_text))
    return entries


def _read_local_manifest(extension_dir: Path) -> dict:
    try:
        manifest_path = extension_dir / MANIFEST_FILENAME
        if manifest_path.is_file():
            data = json.loads(manifest_path.read_text())
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _fetch_remote_manifest(owner: str, repo_name: str) -> dict:
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/refs/heads/main/{MANIFEST_FILENAME}"
    try:
        response = get_session().get(raw_url, follow_redirects=True)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


def _dispatch_unknown_top_level_extension(args: list[str], known_commands: set[str]) -> Optional[int]:
    if not args:
        return None

    command_name = args[0]
    if command_name.startswith("-"):
        return None
    if command_name in known_commands:
        return None

    short_name = command_name[3:] if command_name.startswith("hf-") else command_name
    if not short_name:
        return None

    executable_path = _resolve_installed_executable_path(short_name)
    if not executable_path.is_file():
        return None

    return _execute_extension_binary(executable_path=executable_path, args=list(args[1:]))


def _install_binary_extension(
    *, owner: str, repo_name: str, short_name: str, extension_dir: Path, branch: str
) -> Optional[ExtensionManifest]:
    executable_name = _get_executable_name(short_name)
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/refs/heads/{branch}/{executable_name}"

    try:
        response = get_session().get(raw_url, follow_redirects=True, timeout=_EXTENSIONS_DOWNLOAD_TIMEOUT)
    except Exception as e:
        raise CLIError(
            f"Failed while probing for a root executable '{executable_name}' in '{owner}/{repo_name}': {e}"
        ) from e

    if response.status_code == 404:
        return None

    try:
        response.raise_for_status()
    except Exception as e:
        raise CLIError(
            f"Failed while probing for a root executable '{executable_name}' in '{owner}/{repo_name}': {e}"
        ) from e

    installed = False
    try:
        extension_dir.mkdir(parents=True, exist_ok=False)
        executable_path = extension_dir / executable_name
        executable_path.write_bytes(response.content)
        if os.name != "nt":
            os.chmod(executable_path, 0o755)

        manifest = ExtensionManifest(
            owner=owner,
            repo=repo_name,
            repo_id=f"{owner}/{repo_name}",
            short_name=short_name,
            executable_name=executable_name,
            executable_path=str(executable_path),
            type="binary",
            installed_at=datetime.now(timezone.utc).isoformat(),
            source=f"https://github.com/{owner}/{repo_name}",
        )
        (extension_dir / MANIFEST_FILENAME).write_text(
            json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        installed = True
        return manifest
    finally:
        if not installed:
            shutil.rmtree(extension_dir, ignore_errors=True)


def _install_python_extension(
    *, owner: str, repo_name: str, short_name: str, extension_dir: Path, branch: str
) -> ExtensionManifest:
    source_url = f"https://github.com/{owner}/{repo_name}/archive/refs/heads/{branch}.zip"
    venv_dir = extension_dir / "venv"
    installed = False

    try:
        if extension_dir.exists():
            shutil.rmtree(extension_dir, ignore_errors=True)
        extension_dir.mkdir(parents=True, exist_ok=False)
        venv.EnvBuilder(with_pip=True).create(str(venv_dir))

        venv_python = _get_venv_python_path(venv_dir)
        subprocess.run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                source_url,
            ],
            check=True,
            timeout=_EXTENSIONS_PIP_INSTALL_TIMEOUT,
        )

        executable_name = _get_executable_name(short_name)
        venv_executable = _get_venv_extension_executable_path(venv_dir, short_name)
        if not venv_executable.is_file():
            raise CLIError(
                f"Installed package from '{owner}/{repo_name}' does not expose the required console script "
                f"'{executable_name}'."
            )

        manifest = ExtensionManifest(
            owner=owner,
            repo=repo_name,
            repo_id=f"{owner}/{repo_name}",
            short_name=short_name,
            executable_name=executable_name,
            executable_path=str(venv_executable.resolve()),
            type="python",
            installed_at=datetime.now(timezone.utc).isoformat(),
            source=f"https://github.com/{owner}/{repo_name}",
        )
        (extension_dir / MANIFEST_FILENAME).write_text(
            json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        installed = True
        return manifest
    except CLIError:
        raise
    except subprocess.TimeoutExpired as e:
        raise CLIError(
            f"Pip install timed out after {_EXTENSIONS_PIP_INSTALL_TIMEOUT}s for '{owner}/{repo_name}'. "
            "See pip output above for details."
        ) from e
    except subprocess.CalledProcessError as e:
        raise CLIError(
            f"Failed to install pip package from '{owner}/{repo_name}' (exit code {e.returncode}). "
            "See pip output above for details."
        ) from e
    except Exception as e:
        raise CLIError(f"Failed to set up pip extension from '{owner}/{repo_name}': {e}") from e
    finally:
        if not installed:
            shutil.rmtree(extension_dir, ignore_errors=True)


def _load_extension_manifest(extension_dir: Path, short_name: str) -> Optional[dict[str, Any]]:
    manifest_path = extension_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise CLIError(f"Invalid manifest for extension '{short_name}': {e}") from e
    if not isinstance(payload, dict):
        raise CLIError(f"Invalid manifest for extension '{short_name}'.")
    return payload


def _get_extensions_root() -> Path:
    root_dir = EXTENSIONS_ROOT.expanduser()
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir


def _get_extension_dir(short_name: str) -> Path:
    safe_name = _validate_extension_short_name(short_name, original_input=short_name)
    root = _get_extensions_root().resolve()
    target = (root / f"hf-{safe_name}").resolve()
    if root not in target.parents:
        raise CLIError(f"Invalid extension name '{short_name}'.")
    return target


def _resolve_github_default_branch(owner: str, repo_name: str) -> str:
    try:
        response = get_session().get(
            f"https://api.github.com/repos/{owner}/{repo_name}",
            follow_redirects=True,
            timeout=_EXTENSIONS_DOWNLOAD_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["default_branch"]
    except Exception:
        return _EXTENSIONS_DEFAULT_BRANCH


def _get_executable_name(short_name: str) -> str:
    name = f"hf-{short_name}"
    if os.name == "nt":
        name += ".exe"
    return name


def _resolve_installed_executable_path(short_name: str) -> Path:
    extension_dir = _get_extension_dir(short_name)
    manifest_data = _load_extension_manifest(extension_dir=extension_dir, short_name=short_name)
    if manifest_data is None:
        return extension_dir / _get_executable_name(short_name)

    executable_value = manifest_data.get("executable_path")
    if not isinstance(executable_value, str) or not executable_value.strip():
        raise CLIError(f"Invalid executable path in manifest for extension '{short_name}'.")

    executable_path = Path(executable_value).expanduser()
    if not executable_path.is_absolute():
        executable_path = extension_dir / executable_path
    executable_path = executable_path.resolve()

    if extension_dir != executable_path and extension_dir not in executable_path.parents:
        raise CLIError(f"Invalid executable path in manifest for extension '{short_name}'.")

    return executable_path


def _get_venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _get_venv_extension_executable_path(venv_dir: Path, short_name: str) -> Path:
    executable_name = _get_executable_name(short_name)
    if os.name == "nt":
        return venv_dir / "Scripts" / executable_name
    return venv_dir / "bin" / executable_name


_ALLOWED_EXTENSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_extension_short_name(short_name: str, *, original_input: str) -> str:
    name = short_name.strip()
    if not name:
        raise CLIError("Extension name cannot be empty.")
    if any(sep in name for sep in ("/", "\\")):
        raise CLIError(f"Invalid extension name '{original_input}'.")
    if ".." in name or ":" in name:
        raise CLIError(f"Invalid extension name '{original_input}'.")
    if not _ALLOWED_EXTENSION_NAME.fullmatch(name):
        raise CLIError(
            f"Invalid extension name '{original_input}'. Allowed characters: letters, digits, '.', '_' and '-'."
        )
    return name


def _normalize_repo_id(repo_id: str) -> tuple[str, str, str]:
    if "://" in repo_id:
        raise CLIError("Only GitHub repositories in `[OWNER/]hf-<name>` format are supported.")

    parts = repo_id.split("/")
    if len(parts) == 1:
        owner = DEFAULT_EXTENSION_OWNER
        repo_name = parts[0]
    elif len(parts) == 2 and all(parts):
        owner, repo_name = parts
    else:
        raise CLIError(f"Expected `[OWNER/]REPO` format, got '{repo_id}'.")

    if not repo_name.startswith("hf-"):
        raise CLIError(f"Extension repository name must start with 'hf-', got '{repo_name}'.")

    short_name = repo_name[3:]
    if not short_name:
        raise CLIError("Invalid extension repository name 'hf-'.")
    _validate_extension_short_name(short_name, original_input=repo_id)

    return owner, repo_name, short_name


def _normalize_extension_name(name: str) -> str:
    candidate = name.strip()
    if not candidate:
        raise CLIError("Extension name cannot be empty.")
    normalized = candidate[3:] if candidate.startswith("hf-") else candidate
    return _validate_extension_short_name(normalized, original_input=name)


def _execute_extension_binary(executable_path: Path, args: list[str]) -> int:
    try:
        return subprocess.call([str(executable_path)] + args)
    except OSError as e:
        if os.name == "nt" or e.errno != errno.ENOEXEC:
            raise
        return subprocess.call(["sh", str(executable_path)] + args)
