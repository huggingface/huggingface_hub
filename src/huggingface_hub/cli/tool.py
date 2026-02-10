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
"""Contains commands to manage hf CLI tools.

Usage:
    hf tool install agent                             # Install from HF repo
    hf tool install https://example.com/hf-foo        # Install from URL
    hf tool install ./my-script                       # Install from local file
    hf tool ls                                        # List installed tools
    hf tool remove agent                              # Remove a tool
    hf tool run agent claude --model Qwen/Qwen3-235B  # Run a tool (auto-installs if missing)
"""

import errno
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from huggingface_hub.errors import CLIError
from huggingface_hub.utils import get_session, tabulate

from ._cli_utils import typer_factory


TOOLS_DIR = Path("~/.local/share/hf/plugins/bin")
TOOLS_REPO = "celinah/hf-tools"

tool_cli = typer_factory(help="Manage hf CLI tools.")


def _get_tools_dir() -> Path:
    tools_dir = TOOLS_DIR.expanduser()
    tools_dir.mkdir(parents=True, exist_ok=True)
    return tools_dir


def _resolve_source(source: str) -> str:
    """Determine install type: 'url', 'local', or 'name'."""
    if source.startswith("http://") or source.startswith("https://"):
        return "url"
    if "/" in source or "." in source:
        return "local"
    return "name"


def _derive_tool_name(filename: str) -> str:
    """Derive the tool name from a filename, stripping 'hf-' prefix if present."""
    if filename.startswith("hf-"):
        return filename[3:]
    return filename


def _install_from_name(name: str, tools_dir: Path) -> Path:
    """Download a tool by name from the HF tools repo."""
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    from huggingface_hub.file_download import hf_hub_download

    try:
        downloaded = hf_hub_download(repo_id=TOOLS_REPO, filename=name, repo_type="model")
    except EntryNotFoundError:
        raise CLIError(f"Tool '{name}' not found in {TOOLS_REPO}.")
    except RepositoryNotFoundError:
        raise CLIError(f"Tools repository '{TOOLS_REPO}' not found.")

    dest = tools_dir / f"hf-{name}"
    shutil.copy2(downloaded, dest)
    os.chmod(dest, 0o755)
    return dest


def _install_from_url(url: str, tools_dir: Path) -> Path:
    """Download single-file tool from URL."""
    filename = url.rstrip("/").split("/")[-1]
    dest = tools_dir / filename
    if not filename.startswith("hf-"):
        dest = tools_dir / f"hf-{filename}"

    response = get_session().get(url)
    response.raise_for_status()
    dest.write_bytes(response.content)
    os.chmod(dest, 0o755)
    return dest


def _install_from_local(path: str, tools_dir: Path) -> Path:
    """Copy local file into tools dir."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise CLIError(f"Source file not found: {source}")

    filename = source.name
    dest = tools_dir / filename
    if not filename.startswith("hf-"):
        dest = tools_dir / f"hf-{filename}"

    shutil.copy2(source, dest)
    os.chmod(dest, 0o755)
    return dest


@tool_cli.command(
    "install",
    examples=[
        "hf tool install agent",
        "hf tool install https://example.com/hf-foo",
        "hf tool install ./my-script",
    ],
)
def tool_install(
    source: Annotated[str, typer.Argument(help="Tool name, URL, or local path to install.")],
) -> None:
    """Install a tool by name (from HF repo), URL, or local file."""
    tools_dir = _get_tools_dir()
    source_type = _resolve_source(source)

    if source_type == "name":
        dest = _install_from_name(source, tools_dir)
    elif source_type == "url":
        dest = _install_from_url(source, tools_dir)
    else:
        dest = _install_from_local(source, tools_dir)

    name = _derive_tool_name(dest.name)
    print(f"Installed tool '{name}' to {dest}")
    print(f"Run it with: hf tool run {name}")


@tool_cli.command(
    "ls",
    examples=[
        "hf tool ls",
    ],
)
def tool_ls() -> None:
    """List installed tools."""
    tools_dir = TOOLS_DIR.expanduser()
    if not tools_dir.is_dir():
        print("No tools installed.")
        return

    tools = sorted(p for p in tools_dir.iterdir() if p.is_file() and p.name.startswith("hf-"))
    if not tools:
        print("No tools installed.")
        return

    rows = [[_derive_tool_name(p.name), str(p)] for p in tools]
    print(tabulate(rows, headers=["NAME", "PATH"]))


@tool_cli.command(
    "remove",
    examples=[
        "hf tool remove agent",
    ],
)
def tool_remove(
    name: Annotated[str, typer.Argument(help="Name of the tool to remove (without 'hf-' prefix).")],
) -> None:
    """Remove an installed tool."""
    tools_dir = TOOLS_DIR.expanduser()
    tool_path = tools_dir / f"hf-{name}"

    if not tool_path.is_file():
        raise CLIError(f"Tool '{name}' not found at {tool_path}")

    tool_path.unlink()
    print(f"Removed tool '{name}'.")


@tool_cli.command(
    "run",
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
    examples=[
        "hf tool run agent claude --model moonshotai/Kimi-K2.5",
    ],
)
def tool_run(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Name of the tool to run (without 'hf-' prefix).")],
) -> None:
    """Run a tool (auto-installs from HF repo if not found locally)."""
    tools_dir = _get_tools_dir()
    tool_path = tools_dir / f"hf-{name}"

    if not tool_path.is_file():
        print(f"Tool '{name}' not found locally. Installing from {TOOLS_REPO}...", file=sys.stderr)
        tool_path = _install_from_name(name, tools_dir)
        print(f"Installed tool '{name}' to {tool_path}", file=sys.stderr)

    try:
        exit_code = subprocess.call([str(tool_path)] + ctx.args)
    except OSError as e:
        if e.errno == errno.ENOEXEC:
            exit_code = subprocess.call(["sh", str(tool_path)] + ctx.args)
        else:
            raise

    raise typer.Exit(code=exit_code)
