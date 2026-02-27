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
"""Contains commands to interact with discussions and pull requests on the Hugging Face Hub.

Usage:
    # list open discussions and PRs on a repo
    hf discussions list username/my-model

    # list only pull requests
    hf discussions list username/my-model --kind pull_request

    # view a specific discussion or PR
    hf discussions view username/my-model 5

    # create a new discussion
    hf discussions create username/my-model --title "Bug report"

    # create a new pull request
    hf discussions create username/my-model --title "Fix typo" --pull-request

    # comment on a discussion or PR
    hf discussions comment username/my-model 5 --body "Thanks for reporting!"

    # merge a pull request
    hf discussions merge username/my-model 5

    # show the diff of a pull request
    hf discussions diff username/my-model 5
"""

import enum
import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from huggingface_hub import constants
from huggingface_hub.community import DiscussionComment, DiscussionWithDetails
from huggingface_hub.utils import ANSI

from ._cli_utils import (
    AuthorOpt,
    FormatOpt,
    LimitOpt,
    OutputFormat,
    QuietOpt,
    RepoIdArg,
    RepoType,
    RepoTypeOpt,
    TokenOpt,
    _format_cell,
    api_object_to_dict,
    get_hf_api,
    print_list_output,
    typer_factory,
)


class DiscussionStatus(str, enum.Enum):
    open = "open"
    closed = "closed"
    merged = "merged"
    draft = "draft"
    all = "all"


class DiscussionKind(str, enum.Enum):
    all = "all"
    discussion = "discussion"
    pull_request = "pull_request"


class ViewFormat(str, enum.Enum):
    """Output format for the view command."""

    text = "text"
    json = "json"


# "merged" and "draft" are valid Discussion statuses but the Hub API filter
# (DiscussionStatusFilter) only accepts "all", "open", "closed". When the user
# asks for merged/draft we fetch with api_status=None (i.e. all) and filter
# client-side.
_CLIENT_SIDE_STATUSES = {"merged", "draft"}


DiscussionNumArg = Annotated[
    int,
    typer.Argument(
        help="The discussion or pull request number.",
        min=1,
    ),
]


def _format_status(status: str) -> str:
    if status == "open":
        return ANSI.green("open")
    elif status == "closed":
        return ANSI.red("closed")
    elif status == "merged":
        return ANSI.blue("merged")
    elif status == "draft":
        return ANSI.yellow("draft")
    return status


def _read_body(body: Optional[str], body_file: Optional[Path]) -> Optional[str]:
    """Resolve body text from --body or --body-file (supports '-' for stdin)."""
    if body is not None and body_file is not None:
        raise typer.BadParameter("Cannot use both --body and --body-file.")
    if body_file is not None:
        if str(body_file) == "-":
            return sys.stdin.read()
        return body_file.read_text(encoding="utf-8")
    return body


def _print_discussion_view(details: DiscussionWithDetails, show_comments: bool = False) -> None:
    kind = "Pull Request" if details.is_pull_request else "Discussion"

    print(f"{ANSI.bold(details.title)} {ANSI.gray(f'#{details.num}')}")
    parts = [_format_status(details.status), details.author, details.created_at.strftime("%Y-%m-%d %H:%M")]
    if details.is_pull_request and details.target_branch:
        parts.append(f"into {ANSI.bold(details.target_branch)}")
    print(f"{kind}: {' · '.join(parts)}")

    if details.is_pull_request and details.conflicting_files:
        if details.conflicting_files is True:
            print(ANSI.yellow("Has conflicting files"))
        else:
            print(ANSI.yellow(f"Conflicting files: {', '.join(details.conflicting_files)}"))

    body = None
    comments = []
    for event in details.events:
        if isinstance(event, DiscussionComment) and not event.hidden:
            if body is None:
                body = event
            else:
                comments.append(event)

    if body and body.content.strip():
        print()
        print(body.content.strip())

    if show_comments and comments:
        print()
        print(ANSI.gray("─" * 60))
        for comment in comments:
            print()
            print(f"{ANSI.bold(comment.author)} · {comment.created_at.strftime('%Y-%m-%d %H:%M')}")
            print(comment.content.strip())
    elif comments:
        print()
        print(ANSI.gray(f"{len(comments)} comment{'s' if len(comments) != 1 else ''} (use --comments to show)"))

    print()
    print(f"View on Hub: {ANSI.blue(details.url)}")


discussions_cli = typer_factory(help="Manage discussions and pull requests on the Hub.")


@discussions_cli.command(
    "list | ls",
    examples=[
        "hf discussions list username/my-model",
        "hf discussions list username/my-model --kind pull_request --status merged",
        "hf discussions list username/my-dataset --type dataset --status closed",
        "hf discussions list username/my-model --author alice --format json",
    ],
)
def discussion_list(
    repo_id: RepoIdArg,
    status: Annotated[
        DiscussionStatus,
        typer.Option(
            "-s",
            "--status",
            help="Filter by status (open, closed, merged, draft, all).",
        ),
    ] = DiscussionStatus.open,
    kind: Annotated[
        DiscussionKind,
        typer.Option(
            "-k",
            "--kind",
            help="Filter by kind (discussion, pull_request, all).",
        ),
    ] = DiscussionKind.all,
    author: AuthorOpt = None,
    limit: LimitOpt = 30,
    repo_type: RepoTypeOpt = RepoType.model,
    format: FormatOpt = OutputFormat.table,
    quiet: QuietOpt = False,
    token: TokenOpt = None,
) -> None:
    """List discussions and pull requests on a repo."""
    api = get_hf_api(token=token)

    api_status: Optional[constants.DiscussionStatusFilter]
    if status == DiscussionStatus.open:
        api_status = "open"
    elif status == DiscussionStatus.closed:
        api_status = "closed"
    else:
        api_status = None

    api_discussion_type: Optional[constants.DiscussionTypeFilter]
    if kind == DiscussionKind.all:
        api_discussion_type = None
    else:
        api_discussion_type = kind.value  # type: ignore[assignment]

    discussions = []
    for d in api.get_repo_discussions(
        repo_id=repo_id,
        author=author,
        discussion_type=api_discussion_type,
        discussion_status=api_status,
        repo_type=repo_type.value,
    ):
        if status.value in _CLIENT_SIDE_STATUSES and d.status != status.value:
            continue
        discussions.append(d)
        if len(discussions) >= limit:
            break

    items = [api_object_to_dict(d) for d in discussions]

    print_list_output(
        items,
        format=format,
        quiet=quiet,
        id_key="num",
        headers=["num", "title", "is_pull_request", "status", "author", "created_at"],
        row_fn=lambda item: [
            f"#{item['num']}",
            _format_cell(item.get("title", ""), max_len=50),
            "PR" if item.get("is_pull_request") else "",
            _format_status(str(item.get("status", ""))),
            str(item.get("author", "")),
            _format_cell(item.get("created_at", "")),
        ],
        alignments={"num": "right"},
    )


@discussions_cli.command(
    "view",
    examples=[
        "hf discussions view username/my-model 5",
        "hf discussions view username/my-model 5 --comments",
        "hf discussions view username/my-model 5 --diff",
        "hf discussions view username/my-model 5 --format json",
    ],
)
def discussion_view(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    comments: Annotated[
        bool,
        typer.Option(
            "--comments",
            help="Show all comments.",
        ),
    ] = False,
    diff: Annotated[
        bool,
        typer.Option(
            "--diff",
            help="Show the diff (for pull requests).",
        ),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option(
            "--no-color",
            help="Disable colored output.",
        ),
    ] = False,
    repo_type: RepoTypeOpt = RepoType.model,
    format: Annotated[
        ViewFormat,
        typer.Option(
            help="Output format (text or json).",
        ),
    ] = ViewFormat.text,
    token: TokenOpt = None,
) -> None:
    """View a discussion or pull request."""
    import os

    if no_color:
        os.environ["NO_COLOR"] = "1"

    api = get_hf_api(token=token)
    details = api.get_discussion_details(
        repo_id=repo_id,
        discussion_num=num,
        repo_type=repo_type.value,
    )

    if format == ViewFormat.json:
        result = api_object_to_dict(details)
        if not diff:
            result.pop("diff", None)
        print(json.dumps(result, indent=2))
        return

    _print_discussion_view(details, show_comments=comments)

    if diff and details.diff:
        print()
        print(ANSI.gray("─" * 60))
        print(details.diff)


@discussions_cli.command(
    "create",
    examples=[
        'hf discussions create username/my-model --title "Bug report"',
        'hf discussions create username/my-model --title "Feature request" --body "Please add X"',
        'hf discussions create username/my-model --title "Fix typo" --pull-request',
        'hf discussions create username/my-dataset --type dataset --title "Data quality issue"',
    ],
)
def discussion_create(
    repo_id: RepoIdArg,
    title: Annotated[
        str,
        typer.Option(
            "--title",
            help="The title of the discussion or pull request.",
        ),
    ],
    body: Annotated[
        Optional[str],
        typer.Option(
            "--body",
            help="The description (supports Markdown).",
        ),
    ] = None,
    body_file: Annotated[
        Optional[Path],
        typer.Option(
            "--body-file",
            help="Read the description from a file. Use '-' for stdin.",
        ),
    ] = None,
    pull_request: Annotated[
        bool,
        typer.Option(
            "--pull-request",
            "--pr",
            help="Create a pull request instead of a discussion.",
        ),
    ] = False,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Create a new discussion or pull request on a repo."""
    description = _read_body(body, body_file)
    api = get_hf_api(token=token)
    discussion = api.create_discussion(
        repo_id=repo_id,
        title=title,
        description=description,
        repo_type=repo_type.value,
        pull_request=pull_request,
    )
    kind = "pull request" if pull_request else "discussion"
    print(f"Created {kind} {ANSI.bold(f'#{discussion.num}')} on {ANSI.bold(repo_id)}")
    if pull_request:
        print(f"Push changes to: {ANSI.bold(f'refs/pr/{discussion.num}')}")
    print(f"View on Hub: {ANSI.blue(discussion.url)}")


@discussions_cli.command(
    "comment",
    examples=[
        'hf discussions comment username/my-model 5 --body "Thanks for reporting!"',
        'hf discussions comment username/my-model 5 --body "LGTM!"',
    ],
)
def discussion_comment(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    body: Annotated[
        Optional[str],
        typer.Option(
            "--body",
            help="The comment text (supports Markdown).",
        ),
    ] = None,
    body_file: Annotated[
        Optional[Path],
        typer.Option(
            "--body-file",
            help="Read the comment from a file. Use '-' for stdin.",
        ),
    ] = None,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Comment on a discussion or pull request."""
    comment = _read_body(body, body_file)
    if comment is None:
        raise typer.BadParameter("Either --body or --body-file is required.")
    api = get_hf_api(token=token)
    api.comment_discussion(
        repo_id=repo_id,
        discussion_num=num,
        comment=comment,
        repo_type=repo_type.value,
    )
    print(f"Commented on #{num} in {ANSI.bold(repo_id)}")


@discussions_cli.command(
    "close",
    examples=[
        "hf discussions close username/my-model 5",
        'hf discussions close username/my-model 5 --comment "Closing as resolved."',
    ],
)
def discussion_close(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    comment: Annotated[
        Optional[str],
        typer.Option(
            "--comment",
            help="An optional comment to post when closing.",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Close a discussion or pull request."""
    if not yes:
        confirm = typer.confirm(f"Close #{num} on '{repo_id}'?")
        if not confirm:
            print("Aborted.")
            raise typer.Exit()
    api = get_hf_api(token=token)
    api.change_discussion_status(
        repo_id=repo_id,
        discussion_num=num,
        new_status="closed",
        comment=comment,
        repo_type=repo_type.value,
    )
    print(f"Closed #{num} in {ANSI.bold(repo_id)}")


@discussions_cli.command(
    "reopen",
    examples=[
        "hf discussions reopen username/my-model 5",
        'hf discussions reopen username/my-model 5 --comment "Reopening for further investigation."',
    ],
)
def discussion_reopen(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    comment: Annotated[
        Optional[str],
        typer.Option(
            "--comment",
            help="An optional comment to post when reopening.",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Reopen a closed discussion or pull request."""
    if not yes:
        confirm = typer.confirm(f"Reopen #{num} on '{repo_id}'?")
        if not confirm:
            print("Aborted.")
            raise typer.Exit()
    api = get_hf_api(token=token)
    api.change_discussion_status(
        repo_id=repo_id,
        discussion_num=num,
        new_status="open",
        comment=comment,
        repo_type=repo_type.value,
    )
    print(f"Reopened #{num} in {ANSI.bold(repo_id)}")


@discussions_cli.command(
    "rename",
    examples=[
        'hf discussions rename username/my-model 5 "Updated title"',
    ],
)
def discussion_rename(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    new_title: Annotated[
        str,
        typer.Argument(
            help="The new title.",
        ),
    ],
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Rename a discussion or pull request."""
    api = get_hf_api(token=token)
    api.rename_discussion(
        repo_id=repo_id,
        discussion_num=num,
        new_title=new_title,
        repo_type=repo_type.value,
    )
    print(f"Renamed #{num} to {ANSI.bold(new_title)} in {ANSI.bold(repo_id)}")


@discussions_cli.command(
    "merge",
    examples=[
        "hf discussions merge username/my-model 5",
        'hf discussions merge username/my-model 5 --comment "Merging, thanks!"',
    ],
)
def discussion_merge(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    comment: Annotated[
        Optional[str],
        typer.Option(
            "--comment",
            help="An optional comment to post when merging.",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Merge a pull request."""
    if not yes:
        confirm = typer.confirm(f"Merge #{num} on '{repo_id}'?")
        if not confirm:
            print("Aborted.")
            raise typer.Exit()
    api = get_hf_api(token=token)
    api.merge_pull_request(
        repo_id=repo_id,
        discussion_num=num,
        comment=comment,
        repo_type=repo_type.value,
    )
    print(f"Merged #{num} in {ANSI.bold(repo_id)}")


@discussions_cli.command(
    "diff",
    examples=[
        "hf discussions diff username/my-model 5",
    ],
)
def discussion_diff(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Show the diff of a pull request."""
    api = get_hf_api(token=token)
    details = api.get_discussion_details(
        repo_id=repo_id,
        discussion_num=num,
        repo_type=repo_type.value,
    )
    if details.diff:
        print(details.diff)
    else:
        print("No diff available.")
