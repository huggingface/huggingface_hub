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
"""Contains commands to interact with repositories on the Hugging Face Hub.

Usage:
    # create a new dataset repo on the Hub
    hf repos create my-cool-dataset --repo-type=dataset

    # create a private model repo on the Hub
    hf repos create my-cool-model --private

    # delete files from a repo on the Hub
    hf repos delete-files my-model file.txt

    # list repos or files in a repo
    hf repos list
    hf repos list user/my-model
    hf repos list hf://datasets/user/my-dataset@main
"""

import enum
import json
import sys
from datetime import datetime
from typing import Annotated, Optional, Union

import typer

from huggingface_hub.errors import CLIError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.hf_api import RepoFile, RepoFolder
from huggingface_hub.utils import ANSI, StatusLine, parse_hf_url

from ._cli_utils import (
    FormatOpt,
    OutputFormat,
    PrivateOpt,
    QuietOpt,
    RepoIdArg,
    RepoType,
    RepoTypeOpt,
    RevisionOpt,
    TokenOpt,
    api_object_to_dict,
    get_hf_api,
    print_list_output,
    typer_factory,
)


repos_cli = typer_factory(help="Manage repos on the Hub.")


@repos_cli.callback(invoke_without_command=True)
def _repos_callback(ctx: typer.Context) -> None:
    if ctx.info_name == "repo":
        print(
            ANSI.yellow("FutureWarning: `hf repo` is deprecated in favor of `hf repos`."),
            file=sys.stderr,
        )


tag_cli = typer_factory(help="Manage tags for a repo on the Hub.")
branch_cli = typer_factory(help="Manage branches for a repo on the Hub.")
repos_cli.add_typer(tag_cli, name="tag")
repos_cli.add_typer(branch_cli, name="branch")


def _parse_repo_argument(
    argument: str, repo_type: Optional[str] = None
) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Parse a repo argument accepting both plain paths and hf:// handles.

    Delegates to :func:`~huggingface_hub.utils.parse_hf_url` for the heavy lifting.

    Returns:
        tuple: (repo_type, identifier, revision, path_in_repo)
        - When path_in_repo is None, identifier is a namespace (or None) for listing repos.
        - When path_in_repo is a string (possibly empty), identifier is a repo_id for listing files.
    """
    parsed = parse_hf_url(argument)

    # Resolve resource type
    effective_type = parsed.resource_type
    if effective_type is not None and repo_type is not None and effective_type != repo_type:
        raise ValueError(f"Repo type from handle ('{effective_type}') conflicts with --type ('{repo_type}').")
    if effective_type is None:
        effective_type = repo_type or "model"

    # Determine if this is file-listing mode or repo-listing mode.
    # File mode: repo_id contains "/" (namespace/name) or has a revision.
    # Repo mode: repo_id is None (type-only) or a single-segment namespace (no "/", no "@").
    repo_id = parsed.repo_id
    if repo_id is None:
        return (effective_type, None, None, None)

    has_namespace = "/" in repo_id
    has_revision = parsed.revision is not None
    has_path = parsed.path != ""

    if has_namespace or has_revision:
        path_in_repo = parsed.path if parsed.path else ""
        return (effective_type, repo_id, parsed.revision, path_in_repo)

    if has_path:
        full_id = f"{repo_id}/{parsed.path.split('/')[0]}"
        remaining = "/".join(parsed.path.split("/")[1:])
        return (effective_type, full_id, None, remaining if remaining else "")

    # Single segment, no revision, no path → namespace for listing repos
    return (effective_type, repo_id, None, None)


def _format_size(size: Union[int, float], human_readable: bool = False) -> str:
    """Format a size in bytes."""
    if not human_readable:
        return str(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1000:
            if unit == "B":
                return f"{size} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} PB"


def _format_mtime(dt: Optional[datetime], human_readable: bool = False) -> str:
    """Format datetime to a readable date string."""
    if dt is None:
        return ""
    if human_readable:
        return dt.strftime("%b %d %H:%M")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _build_tree(
    items: list[Union[RepoFile, RepoFolder]],
    human_readable: bool = False,
    quiet: bool = False,
) -> list[str]:
    """Build a tree representation of files and directories."""
    tree: dict = {}

    for item in items:
        parts = item.path.split("/")
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {"__children__": {}}
            current = current[part]["__children__"]

        final_part = parts[-1]
        if isinstance(item, RepoFolder):
            if final_part not in current:
                current[final_part] = {"__children__": {}}
        else:
            current[final_part] = {"__item__": item}

    prefix_width = 0
    max_size_width = 0
    max_date_width = 0
    if not quiet:
        for item in items:
            if isinstance(item, RepoFile):
                size_str = _format_size(item.size, human_readable)
                max_size_width = max(max_size_width, len(size_str))
                if item.last_commit is not None:
                    date_str = _format_mtime(item.last_commit.date, human_readable)
                    max_date_width = max(max_date_width, len(date_str))
        if max_size_width > 0:
            prefix_width = max_size_width + 2 + max_date_width

    lines: list[str] = []
    _render_tree(
        tree,
        lines,
        "",
        prefix_width=prefix_width,
        max_size_width=max_size_width,
        human_readable=human_readable,
    )
    return lines


def _render_tree(
    node: dict,
    lines: list[str],
    indent: str,
    prefix_width: int = 0,
    max_size_width: int = 0,
    human_readable: bool = False,
) -> None:
    """Recursively render a tree structure with size+date prefix."""
    items = sorted(node.items())
    for i, (name, value) in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "

        is_dir = "__children__" in value
        children = value.get("__children__", {})

        if prefix_width > 0:
            if is_dir:
                prefix = " " * prefix_width
            else:
                item = value.get("__item__")
                if item is not None:
                    size_str = _format_size(item.size, human_readable)
                    date_str = _format_mtime(
                        item.last_commit.date if item.last_commit else None,
                        human_readable,
                    )
                    prefix = f"{size_str:>{max_size_width}}  {date_str}"
                else:
                    prefix = " " * prefix_width
            lines.append(f"{prefix}  {indent}{connector}{name}{'/' if is_dir else ''}")
        else:
            lines.append(f"{indent}{connector}{name}{'/' if is_dir else ''}")

        if children:
            child_indent = indent + ("    " if is_last else "│   ")
            _render_tree(
                children,
                lines,
                child_indent,
                prefix_width=prefix_width,
                max_size_width=max_size_width,
                human_readable=human_readable,
            )


@repos_cli.command(
    name="list | ls",
    examples=[
        "hf repos list",
        "hf repos list huggingface",
        "hf repos list user/my-model",
        "hf repos list user/my-model -R",
        "hf repos list user/my-model -h",
        "hf repos list user/my-model --tree",
        "hf repos list user/my-model --tree -h",
        "hf repos list hf://datasets/user/my-dataset",
        "hf repos list hf://datasets/user/my-dataset@main",
        "hf repos list user/my-model/sub -R",
    ],
)
def list_cmd(
    argument: Annotated[
        Optional[str],
        typer.Argument(
            help=(
                "Namespace (user or org) to list repos, or repo ID"
                " (namespace/repo_name(/path) or hf://...) to list files."
            ),
        ),
    ] = None,
    human_readable: Annotated[
        bool,
        typer.Option(
            "--human-readable",
            "-h",
            help="Show sizes in human readable format.",
        ),
    ] = False,
    as_tree: Annotated[
        bool,
        typer.Option(
            "--tree",
            help="List files in tree format (only for listing files).",
        ),
    ] = False,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive",
            "-R",
            help="List files recursively (only for listing files).",
        ),
    ] = False,
    format: FormatOpt = OutputFormat.table,
    quiet: QuietOpt = False,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """List repos or files in a repo.

    When called with no argument or a namespace, lists repos.
    When called with a repo ID (namespace/repo_name), lists files in the repo.
    """
    try:
        parsed_type, identifier, revision, path_in_repo = _parse_repo_argument(
            argument or "", repo_type=None if argument and argument.startswith("hf://") else repo_type.value
        )
    except ValueError as e:
        raise typer.BadParameter(str(e))

    # Use parsed type if it came from the handle, otherwise use the --type flag
    effective_type = parsed_type if (argument and argument.startswith("hf://")) else repo_type.value

    is_file_mode = path_in_repo is not None

    if is_file_mode:
        _list_repo_files(
            repo_id=identifier,  # type: ignore[arg-type]
            path_in_repo=path_in_repo or None,
            revision=revision,
            repo_type=effective_type,
            human_readable=human_readable,
            as_tree=as_tree,
            recursive=recursive,
            format=format,
            quiet=quiet,
            token=token,
        )
    else:
        _list_repos(
            namespace=identifier,
            repo_type=effective_type,
            human_readable=human_readable,
            as_tree=as_tree,
            recursive=recursive,
            format=format,
            quiet=quiet,
            token=token,
        )


def _list_repos(
    namespace: Optional[str],
    repo_type: str,
    human_readable: bool,
    as_tree: bool,
    recursive: bool,
    format: OutputFormat,
    quiet: bool,
    token: Optional[str],
) -> None:
    """List repos in a namespace."""
    if as_tree:
        raise typer.BadParameter("Cannot use --tree when listing repos.")
    if recursive:
        raise typer.BadParameter("Cannot use --recursive when listing repos.")

    api = get_hf_api(token=token)

    if repo_type == "model":
        results = [api_object_to_dict(info) for info in api.list_models(author=namespace)]
    elif repo_type == "dataset":
        results = [api_object_to_dict(info) for info in api.list_datasets(author=namespace)]
    elif repo_type == "space":
        results = [api_object_to_dict(info) for info in api.list_spaces(author=namespace)]
    else:
        raise typer.BadParameter(f"Unknown repo type: {repo_type}")

    if not results:
        if not quiet and format != OutputFormat.json:
            resolved_namespace = namespace if namespace is not None else api.whoami()["name"]
            print(f"No {repo_type}s found under namespace '{resolved_namespace}'.")
            return

    print_list_output(results, format=format, quiet=quiet)


def _list_repo_files(
    repo_id: str,
    path_in_repo: Optional[str],
    revision: Optional[str],
    repo_type: str,
    human_readable: bool,
    as_tree: bool,
    recursive: bool,
    format: OutputFormat,
    quiet: bool,
    token: Optional[str],
) -> None:
    """List files in a repo."""
    if as_tree and format == OutputFormat.json:
        raise typer.BadParameter("Cannot use --tree with --format json.")

    api = get_hf_api(token=token)

    items = list(
        api.list_repo_tree(
            repo_id,
            path_in_repo=path_in_repo,
            recursive=recursive,
            expand=True,
            revision=revision,
            repo_type=repo_type,
        )
    )

    if not items:
        print("(empty)")
        return

    has_directories = any(isinstance(item, RepoFolder) for item in items)

    if format == OutputFormat.json:
        results = [api_object_to_dict(item) for item in items]
        print(json.dumps(results, indent=2))
    elif as_tree:
        tree_lines = _build_tree(items, human_readable=human_readable, quiet=quiet)
        for line in tree_lines:
            print(line)
    elif quiet:
        for item in items:
            if isinstance(item, RepoFolder):
                print(f"{item.path}/")
            else:
                print(item.path)
    else:
        for item in items:
            if isinstance(item, RepoFolder):
                date_str = _format_mtime(
                    item.last_commit.date if item.last_commit else None,
                    human_readable,
                )
                print(f"{'':>12}  {date_str:>19}  {item.path}/")
            else:
                size_str = _format_size(item.size, human_readable)
                date_str = _format_mtime(
                    item.last_commit.date if item.last_commit else None,
                    human_readable,
                )
                print(f"{size_str:>12}  {date_str:>19}  {item.path}")

    if not recursive and has_directories:
        StatusLine().done("Use -R to list files recursively.")


class GatedChoices(str, enum.Enum):
    auto = "auto"
    manual = "manual"
    false = "false"


@repos_cli.command(
    "create",
    examples=[
        "hf repos create my-model",
        "hf repos create my-dataset --repo-type dataset --private",
    ],
)
def repo_create(
    repo_id: RepoIdArg,
    repo_type: RepoTypeOpt = RepoType.model,
    space_sdk: Annotated[
        Optional[str],
        typer.Option(
            help="Hugging Face Spaces SDK type. Required when --type is set to 'space'.",
        ),
    ] = None,
    private: PrivateOpt = None,
    token: TokenOpt = None,
    exist_ok: Annotated[
        bool,
        typer.Option(
            help="Do not raise an error if repo already exists.",
        ),
    ] = False,
    resource_group_id: Annotated[
        Optional[str],
        typer.Option(
            help="Resource group in which to create the repo. Resource groups is only available for Enterprise Hub organizations.",
        ),
    ] = None,
) -> None:
    """Create a new repo on the Hub."""
    api = get_hf_api(token=token)
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type.value,
        private=private,
        token=token,
        exist_ok=exist_ok,
        resource_group_id=resource_group_id,
        space_sdk=space_sdk,
    )
    print(f"Successfully created {ANSI.bold(repo_url.repo_id)} on the Hub.")
    print(f"Your repo is now available at {ANSI.bold(repo_url)}")


@repos_cli.command("delete", examples=["hf repos delete my-model"])
def repo_delete(
    repo_id: RepoIdArg,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
    missing_ok: Annotated[
        bool,
        typer.Option(
            help="If set to True, do not raise an error if repo does not exist.",
        ),
    ] = False,
) -> None:
    """Delete a repo from the Hub. This is an irreversible operation."""
    api = get_hf_api(token=token)
    api.delete_repo(
        repo_id=repo_id,
        repo_type=repo_type.value,
        missing_ok=missing_ok,
    )
    print(f"Successfully deleted {ANSI.bold(repo_id)} on the Hub.")


@repos_cli.command("move", examples=["hf repos move old-namespace/my-model new-namespace/my-model"])
def repo_move(
    from_id: RepoIdArg,
    to_id: RepoIdArg,
    token: TokenOpt = None,
    repo_type: RepoTypeOpt = RepoType.model,
) -> None:
    """Move a repository from a namespace to another namespace."""
    api = get_hf_api(token=token)
    api.move_repo(
        from_id=from_id,
        to_id=to_id,
        repo_type=repo_type.value,
    )
    print(f"Successfully moved {ANSI.bold(from_id)} to {ANSI.bold(to_id)} on the Hub.")


@repos_cli.command(
    "settings",
    examples=[
        "hf repos settings my-model --private",
        "hf repos settings my-model --gated auto",
    ],
)
def repo_settings(
    repo_id: RepoIdArg,
    gated: Annotated[
        Optional[GatedChoices],
        typer.Option(
            help="The gated status for the repository.",
        ),
    ] = None,
    private: Annotated[
        Optional[bool],
        typer.Option(
            help="Whether the repository should be private.",
        ),
    ] = None,
    token: TokenOpt = None,
    repo_type: RepoTypeOpt = RepoType.model,
) -> None:
    """Update the settings of a repository."""
    api = get_hf_api(token=token)
    api.update_repo_settings(
        repo_id=repo_id,
        gated=(gated.value if gated else None),  # type: ignore [arg-type]
        private=private,
        repo_type=repo_type.value,
    )
    print(f"Successfully updated the settings of {ANSI.bold(repo_id)} on the Hub.")


@repos_cli.command(
    "delete-files",
    examples=[
        "hf repos delete-files my-model file.txt",
        'hf repos delete-files my-model "*.json"',
        "hf repos delete-files my-model folder/",
    ],
)
def repo_delete_files(
    repo_id: RepoIdArg,
    patterns: Annotated[
        list[str],
        typer.Argument(
            help="Glob patterns to match files to delete. Based on fnmatch, '*' matches files recursively.",
        ),
    ],
    repo_type: RepoTypeOpt = RepoType.model,
    revision: RevisionOpt = None,
    commit_message: Annotated[
        Optional[str],
        typer.Option(
            help="The summary / title / first line of the generated commit.",
        ),
    ] = None,
    commit_description: Annotated[
        Optional[str],
        typer.Option(
            help="The description of the generated commit.",
        ),
    ] = None,
    create_pr: Annotated[
        bool,
        typer.Option(
            help="Whether to create a new Pull Request for these changes.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Delete files from a repo on the Hub."""
    api = get_hf_api(token=token)
    url = api.delete_files(
        delete_patterns=patterns,
        repo_id=repo_id,
        repo_type=repo_type.value,
        revision=revision,
        commit_message=commit_message,
        commit_description=commit_description,
        create_pr=create_pr,
    )
    print(f"Files correctly deleted from repo. Commit: {url}.")


@branch_cli.command(
    "create",
    examples=[
        "hf repos branch create my-model dev",
        "hf repos branch create my-model dev --revision abc123",
    ],
)
def branch_create(
    repo_id: RepoIdArg,
    branch: Annotated[
        str,
        typer.Argument(
            help="The name of the branch to create.",
        ),
    ],
    revision: RevisionOpt = None,
    token: TokenOpt = None,
    repo_type: RepoTypeOpt = RepoType.model,
    exist_ok: Annotated[
        bool,
        typer.Option(
            help="If set to True, do not raise an error if branch already exists.",
        ),
    ] = False,
) -> None:
    """Create a new branch for a repo on the Hub."""
    api = get_hf_api(token=token)
    api.create_branch(
        repo_id=repo_id,
        branch=branch,
        revision=revision,
        repo_type=repo_type.value,
        exist_ok=exist_ok,
    )
    print(f"Successfully created {ANSI.bold(branch)} branch on {repo_type.value} {ANSI.bold(repo_id)}")


@branch_cli.command("delete", examples=["hf repos branch delete my-model dev"])
def branch_delete(
    repo_id: RepoIdArg,
    branch: Annotated[
        str,
        typer.Argument(
            help="The name of the branch to delete.",
        ),
    ],
    token: TokenOpt = None,
    repo_type: RepoTypeOpt = RepoType.model,
) -> None:
    """Delete a branch from a repo on the Hub."""
    api = get_hf_api(token=token)
    api.delete_branch(
        repo_id=repo_id,
        branch=branch,
        repo_type=repo_type.value,
    )
    print(f"Successfully deleted {ANSI.bold(branch)} branch on {repo_type.value} {ANSI.bold(repo_id)}")


@tag_cli.command(
    "create",
    examples=[
        "hf repos tag create my-model v1.0",
        'hf repos tag create my-model v1.0 -m "First release"',
    ],
)
def tag_create(
    repo_id: RepoIdArg,
    tag: Annotated[
        str,
        typer.Argument(
            help="The name of the tag to create.",
        ),
    ],
    message: Annotated[
        Optional[str],
        typer.Option(
            "-m",
            "--message",
            help="The description of the tag to create.",
        ),
    ] = None,
    revision: RevisionOpt = None,
    token: TokenOpt = None,
    repo_type: RepoTypeOpt = RepoType.model,
) -> None:
    """Create a tag for a repo."""
    repo_type_str = repo_type.value
    api = get_hf_api(token=token)
    print(f"You are about to create tag {ANSI.bold(tag)} on {repo_type_str} {ANSI.bold(repo_id)}")
    try:
        api.create_tag(repo_id=repo_id, tag=tag, tag_message=message, revision=revision, repo_type=repo_type_str)
    except RepositoryNotFoundError as e:
        raise CLIError(f"{repo_type_str.capitalize()} '{repo_id}' not found.") from e
    except RevisionNotFoundError as e:
        raise CLIError(f"Revision '{revision}' not found.") from e
    except HfHubHTTPError as e:
        if e.response.status_code == 409:
            raise CLIError(f"Tag '{tag}' already exists on '{repo_id}'.") from e
        raise
    print(f"Tag {ANSI.bold(tag)} created on {ANSI.bold(repo_id)}")


@tag_cli.command("list", examples=["hf repos tag list my-model"])
def tag_list(
    repo_id: RepoIdArg,
    token: TokenOpt = None,
    repo_type: RepoTypeOpt = RepoType.model,
) -> None:
    """List tags for a repo."""
    repo_type_str = repo_type.value
    api = get_hf_api(token=token)
    try:
        refs = api.list_repo_refs(repo_id=repo_id, repo_type=repo_type_str)
    except RepositoryNotFoundError as e:
        raise CLIError(f"{repo_type_str.capitalize()} '{repo_id}' not found.") from e
    if len(refs.tags) == 0:
        print("No tags found")
        raise typer.Exit(code=0)
    print(f"Tags for {repo_type_str} {ANSI.bold(repo_id)}:")
    for t in refs.tags:
        print(t.name)


@tag_cli.command("delete", examples=["hf repos tag delete my-model v1.0"])
def tag_delete(
    repo_id: RepoIdArg,
    tag: Annotated[
        str,
        typer.Argument(
            help="The name of the tag to delete.",
        ),
    ],
    yes: Annotated[
        bool,
        typer.Option(
            "-y",
            "--yes",
            help="Answer Yes to prompt automatically",
        ),
    ] = False,
    token: TokenOpt = None,
    repo_type: RepoTypeOpt = RepoType.model,
) -> None:
    """Delete a tag for a repo."""
    repo_type_str = repo_type.value
    print(f"You are about to delete tag {ANSI.bold(tag)} on {repo_type_str} {ANSI.bold(repo_id)}")
    if not yes:
        choice = input("Proceed? [Y/n] ").lower()
        if choice not in ("", "y", "yes"):
            print("Abort")
            raise typer.Exit()
    api = get_hf_api(token=token)
    try:
        api.delete_tag(repo_id=repo_id, tag=tag, repo_type=repo_type_str)
    except RepositoryNotFoundError as e:
        raise CLIError(f"{repo_type_str.capitalize()} '{repo_id}' not found.") from e
    except RevisionNotFoundError as e:
        raise CLIError(f"Tag '{tag}' not found on '{repo_id}'.") from e
    print(f"Tag {ANSI.bold(tag)} deleted on {ANSI.bold(repo_id)}")
