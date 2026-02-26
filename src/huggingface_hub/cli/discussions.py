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
"""Contains commands to interact with discussions on the Hugging Face Hub.

Usage:
    # list open discussions on a repo
    hf discussion list username/my-model

    # view a specific discussion
    hf discussion view username/my-model 5

    # create a new discussion
    hf discussion create username/my-model --title "Bug report"

    # comment on a discussion
    hf discussion comment username/my-model 5 --body "Thanks for reporting!"

    # close a discussion
    hf discussion close username/my-model 5
"""

import enum
import json
import webbrowser
from typing import Annotated, Optional

import typer

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


class DiscussionState(str, enum.Enum):
    open = "open"
    closed = "closed"
    all = "all"


DiscussionNumArg = Annotated[
    int,
    typer.Argument(
        help="The discussion number.",
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


discussions_cli = typer_factory(help="Manage discussions on the Hub.")


@discussions_cli.command(
    "list | ls",
    examples=[
        "hf discussion list username/my-model",
        "hf discussion list username/my-dataset --type dataset --state closed",
        "hf discussion list username/my-model --author alice --format json",
    ],
)
def discussion_list(
    repo_id: RepoIdArg,
    state: Annotated[
        DiscussionState,
        typer.Option(
            "-s",
            "--state",
            help="Filter by state (open, closed, all).",
        ),
    ] = DiscussionState.open,
    author: AuthorOpt = None,
    limit: LimitOpt = 30,
    repo_type: RepoTypeOpt = RepoType.model,
    format: FormatOpt = OutputFormat.table,
    quiet: QuietOpt = False,
    token: TokenOpt = None,
) -> None:
    """List discussions on a repo."""
    api = get_hf_api(token=token)
    status_filter = None if state == DiscussionState.all else state.value

    discussions = []
    for d in api.get_repo_discussions(
        repo_id=repo_id,
        author=author,
        discussion_type="discussion",
        discussion_status=status_filter,
        repo_type=repo_type.value,
    ):
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
            "status": d.status,
            "author": d.author,
            "createdAt": d.created_at.strftime("%Y-%m-%d"),
        }
        for d in discussions
    ]
    print_as_table(
        items,
        headers=["num", "title", "status", "author", "createdAt"],
        row_fn=lambda item: [
            f"#{item['num']}",
            _format_cell(item["title"], max_len=50),
            item["status"],
            item["author"],
            item["createdAt"],
        ],
        alignments={"num": "right"},
    )


@discussions_cli.command(
    "view",
    examples=[
        "hf discussion view username/my-model 5",
        "hf discussion view username/my-model 5 --comments",
        "hf discussion view username/my-model 5 --format json",
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

        result = {
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
        print(json.dumps(result, indent=2))
        return

    _print_discussion_view(details, show_comments=comments)


@discussions_cli.command(
    "create",
    examples=[
        'hf discussion create username/my-model --title "Bug report"',
        'hf discussion create username/my-model --title "Feature request" --body "Please add X"',
        'hf discussion create username/my-dataset --type dataset --title "Data quality issue"',
    ],
)
def discussion_create(
    repo_id: RepoIdArg,
    title: Annotated[
        str,
        typer.Option(
            "-t",
            "--title",
            help="The title of the discussion.",
        ),
    ],
    body: Annotated[
        Optional[str],
        typer.Option(
            "-b",
            "--body",
            help="The body/description of the discussion (supports Markdown).",
        ),
    ] = None,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Create a new discussion on a repo."""
    api = get_hf_api(token=token)
    discussion = api.create_discussion(
        repo_id=repo_id,
        title=title,
        description=body,
        repo_type=repo_type.value,
    )
    print(f"Created discussion {ANSI.bold(f'#{discussion.num}')} on {ANSI.bold(repo_id)}")
    print(f"View on Hub: {ANSI.blue(discussion.url)}")


@discussions_cli.command(
    "comment",
    examples=[
        'hf discussion comment username/my-model 5 --body "Thanks for reporting!"',
        'hf discussion comment username/my-model 5 --body "Fixed in latest release."',
    ],
)
def discussion_comment(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    body: Annotated[
        str,
        typer.Option(
            "-b",
            "--body",
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
        comment=body,
        repo_type=repo_type.value,
    )
    print(f"Commented on #{num} in {ANSI.bold(repo_id)}")


@discussions_cli.command(
    "close",
    examples=[
        "hf discussion close username/my-model 5",
        'hf discussion close username/my-model 5 --comment "Closing as resolved."',
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
        "hf discussion reopen username/my-model 5",
        'hf discussion reopen username/my-model 5 --comment "Reopening for further investigation."',
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
        'hf discussion rename username/my-model 5 --title "Updated title"',
    ],
)
def discussion_rename(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    title: Annotated[
        str,
        typer.Option(
            "-t",
            "--title",
            help="The new title for the discussion.",
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
        new_title=title,
        repo_type=repo_type.value,
    )
    print(f"Renamed #{num} to {ANSI.bold(title)} in {ANSI.bold(repo_id)}")
