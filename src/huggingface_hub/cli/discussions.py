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
    hf discussions list username/my-model --discussion-type pull_request

    # view a specific discussion or PR
    hf discussions view username/my-model 5

    # create a new discussion
    hf discussions create username/my-model --title "Bug report"

    # create a new pull request
    hf discussions create username/my-model --title "Fix typo" --pull-request

    # comment on a discussion or PR
    hf discussions comment username/my-model 5 --comment "Thanks for reporting!"

    # merge a pull request
    hf discussions merge username/my-model 5

    # show the diff of a pull request
    hf discussions diff username/my-model 5
"""

import enum
import json
import webbrowser
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
    get_hf_api,
    print_as_table,
    typer_factory,
)


class DiscussionStatus(str, enum.Enum):
    open = "open"
    closed = "closed"
    merged = "merged"
    draft = "draft"
    all = "all"


class DiscussionType(str, enum.Enum):
    all = "all"
    discussion = "discussion"
    pull_request = "pull_request"


# Statuses that require client-side filtering (not natively supported by the Hub API)
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
        "hf discussions list username/my-model --discussion-type pull_request --status merged",
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
    discussion_type: Annotated[
        DiscussionType,
        typer.Option(
            "--discussion-type",
            help="Filter by type (discussion, pull_request, all).",
        ),
    ] = DiscussionType.all,
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
    if discussion_type == DiscussionType.all:
        api_discussion_type = None
    else:
        api_discussion_type = discussion_type.value  # type: ignore[assignment]

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

    if quiet:
        for d in discussions:
            print(d.num)
        return

    if format == OutputFormat.json:
        items = [
            {
                "num": d.num,
                "title": d.title,
                "status": d.status,
                "author": d.author,
                "isPullRequest": d.is_pull_request,
                "createdAt": d.created_at.isoformat(),
                "url": d.url,
            }
            for d in discussions
        ]
        print(json.dumps(items, indent=2))
        return

    items = [
        {
            "num": d.num,
            "title": d.title,
            "kind": "PR" if d.is_pull_request else "",
            "status": d.status,
            "author": d.author,
            "createdAt": d.created_at.strftime("%Y-%m-%d"),
        }
        for d in discussions
    ]
    print_as_table(
        items,
        headers=["num", "title", "kind", "status", "author", "createdAt"],
        row_fn=lambda item: [
            f"#{item['num']}",
            _format_cell(item["title"], max_len=50),
            item["kind"],
            _format_status(item["status"]),
            item["author"],
            item["createdAt"],
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
            "-c",
            "--comments",
            help="Show all comments.",
        ),
    ] = False,
    diff: Annotated[
        bool,
        typer.Option(
            "-d",
            "--diff",
            help="Show the diff (for pull requests).",
        ),
    ] = False,
    web: Annotated[
        bool,
        typer.Option(
            "-w",
            "--web",
            help="Open in the browser instead of printing to the terminal.",
        ),
    ] = False,
    repo_type: RepoTypeOpt = RepoType.model,
    format: FormatOpt = OutputFormat.table,
    token: TokenOpt = None,
) -> None:
    """View a discussion or pull request."""
    api = get_hf_api(token=token)
    details = api.get_discussion_details(
        repo_id=repo_id,
        discussion_num=num,
        repo_type=repo_type.value,
    )

    if web:
        webbrowser.open(details.url)
        return

    if format == OutputFormat.json:
        events = []
        for event in details.events:
            event_dict: dict = {
                "id": event.id,
                "type": event.type,
                "author": event.author,
                "createdAt": event.created_at.isoformat(),
            }
            if isinstance(event, DiscussionComment):
                event_dict["content"] = event.content
                event_dict["edited"] = event.edited
                event_dict["hidden"] = event.hidden
            events.append(event_dict)

        result: dict = {
            "num": details.num,
            "title": details.title,
            "status": details.status,
            "author": details.author,
            "isPullRequest": details.is_pull_request,
            "createdAt": details.created_at.isoformat(),
            "url": details.url,
            "events": events,
        }
        if details.is_pull_request:
            result["targetBranch"] = details.target_branch
            result["conflictingFiles"] = details.conflicting_files
            result["mergeCommitOid"] = details.merge_commit_oid
        if diff and details.diff:
            result["diff"] = details.diff
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
        'hf discussions create username/my-model --title "Feature request" --description "Please add X"',
        'hf discussions create username/my-model --title "Fix typo" --pull-request',
        'hf discussions create username/my-dataset --type dataset --title "Data quality issue"',
    ],
)
def discussion_create(
    repo_id: RepoIdArg,
    title: Annotated[
        str,
        typer.Option(
            "-t",
            "--title",
            help="The title of the discussion or pull request.",
        ),
    ],
    description: Annotated[
        Optional[str],
        typer.Option(
            help="The description (supports Markdown).",
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
        'hf discussions comment username/my-model 5 --comment "Thanks for reporting!"',
        'hf discussions comment username/my-model 5 --comment "LGTM!"',
    ],
)
def discussion_comment(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    comment: Annotated[
        str,
        typer.Option(
            "-c",
            "--comment",
            help="The comment text (supports Markdown).",
        ),
    ],
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Comment on a discussion or pull request."""
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
            "-c",
            "--comment",
            help="An optional comment to post when closing.",
        ),
    ] = None,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Close a discussion or pull request."""
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
            "-c",
            "--comment",
            help="An optional comment to post when reopening.",
        ),
    ] = None,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Reopen a closed discussion or pull request."""
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
        'hf discussions rename username/my-model 5 --new-title "Updated title"',
    ],
)
def discussion_rename(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    new_title: Annotated[
        str,
        typer.Option(
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
            "-c",
            "--comment",
            help="An optional comment to post when merging.",
        ),
    ] = None,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Merge a pull request."""
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
