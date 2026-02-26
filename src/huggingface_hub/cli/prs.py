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
"""Contains commands to interact with pull requests on the Hugging Face Hub.

Usage:
    # list open pull requests on a repo
    hf pr list username/my-model

    # view a specific pull request
    hf pr view username/my-model 5

    # create a new pull request
    hf pr create username/my-model --title "Fix typo in config"

    # merge a pull request
    hf pr merge username/my-model 5

    # show the diff of a pull request
    hf pr diff username/my-model 5
"""

import enum
import json
import webbrowser
from typing import Annotated, Optional

import typer

from huggingface_hub.community import DiscussionComment
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
from .discussions import DiscussionNumArg, _format_status, _print_discussion_view


class PRState(str, enum.Enum):
    open = "open"
    closed = "closed"
    merged = "merged"
    draft = "draft"
    all = "all"


# States that require client-side filtering (not supported by the API filter)
_CLIENT_SIDE_STATES = {"merged", "draft"}


prs_cli = typer_factory(help="Manage pull requests on the Hub.")


@prs_cli.command(
    "list | ls",
    examples=[
        "hf pr list username/my-model",
        "hf pr list username/my-model --state merged",
        "hf pr list username/my-dataset --type dataset --author alice --format json",
    ],
)
def pr_list(
    repo_id: RepoIdArg,
    state: Annotated[
        PRState,
        typer.Option(
            "-s",
            "--state",
            help="Filter by state (open, closed, merged, draft, all).",
        ),
    ] = PRState.open,
    author: AuthorOpt = None,
    limit: LimitOpt = 30,
    repo_type: RepoTypeOpt = RepoType.model,
    format: FormatOpt = OutputFormat.table,
    quiet: QuietOpt = False,
    token: TokenOpt = None,
) -> None:
    """List pull requests on a repo."""
    api = get_hf_api(token=token)

    api_status: Optional[str]
    if state == PRState.open:
        api_status = "open"
    elif state == PRState.closed:
        api_status = "closed"
    else:
        api_status = None

    prs = []
    for d in api.get_repo_discussions(
        repo_id=repo_id,
        author=author,
        discussion_type="pull_request",
        discussion_status=api_status,
        repo_type=repo_type.value,
    ):
        if state.value in _CLIENT_SIDE_STATES and d.status != state.value:
            continue
        prs.append(d)
        if len(prs) >= limit:
            break

    if quiet:
        for pr in prs:
            print(pr.num)
        return

    if format == OutputFormat.json:
        items = [
            {
                "num": pr.num,
                "title": pr.title,
                "status": pr.status,
                "author": pr.author,
                "createdAt": pr.created_at.isoformat(),
                "url": pr.url,
            }
            for pr in prs
        ]
        print(json.dumps(items, indent=2))
        return

    items = [
        {
            "num": pr.num,
            "title": pr.title,
            "status": pr.status,
            "author": pr.author,
            "createdAt": pr.created_at.strftime("%Y-%m-%d"),
        }
        for pr in prs
    ]
    print_as_table(
        items,
        headers=["num", "title", "status", "author", "createdAt"],
        row_fn=lambda item: [
            f"#{item['num']}",
            _format_cell(item["title"], max_len=50),
            _format_status(item["status"]),
            item["author"],
            item["createdAt"],
        ],
        alignments={"num": "right"},
    )


@prs_cli.command(
    "view",
    examples=[
        "hf pr view username/my-model 5",
        "hf pr view username/my-model 5 --comments",
        "hf pr view username/my-model 5 --diff",
        "hf pr view username/my-model 5 --format json",
    ],
)
def pr_view(
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
            help="Show the diff.",
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
    """View a pull request."""
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
            "targetBranch": details.target_branch,
            "conflictingFiles": details.conflicting_files,
            "mergeCommitOid": details.merge_commit_oid,
            "events": events,
        }
        if diff and details.diff:
            result["diff"] = details.diff
        print(json.dumps(result, indent=2))
        return

    _print_discussion_view(details, show_comments=comments)

    if diff and details.diff:
        print()
        print(ANSI.gray("─" * 60))
        print(details.diff)


@prs_cli.command(
    "create",
    examples=[
        'hf pr create username/my-model --title "Fix typo in config"',
        'hf pr create username/my-model --title "Update README" --body "Improved documentation"',
        'hf pr create username/my-dataset --type dataset --title "Add new samples"',
    ],
)
def pr_create(
    repo_id: RepoIdArg,
    title: Annotated[
        str,
        typer.Option(
            "-t",
            "--title",
            help="The title of the pull request.",
        ),
    ],
    body: Annotated[
        Optional[str],
        typer.Option(
            "-b",
            "--body",
            help="The body/description of the pull request (supports Markdown).",
        ),
    ] = None,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Create a new pull request on a repo.

    The pull request will be created in draft status. Push commits to the
    `refs/pr/<num>` git reference to add changes.
    """
    api = get_hf_api(token=token)
    pr = api.create_pull_request(
        repo_id=repo_id,
        title=title,
        description=body,
        repo_type=repo_type.value,
    )
    print(f"Created pull request {ANSI.bold(f'#{pr.num}')} on {ANSI.bold(repo_id)}")
    print(f"Push changes to: {ANSI.bold(f'refs/pr/{pr.num}')}")
    print(f"View on Hub: {ANSI.blue(pr.url)}")


@prs_cli.command(
    "comment",
    examples=[
        'hf pr comment username/my-model 5 --body "LGTM!"',
        'hf pr comment username/my-model 5 --body "Please update the config file."',
    ],
)
def pr_comment(
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
    """Comment on a pull request."""
    api = get_hf_api(token=token)
    api.comment_discussion(
        repo_id=repo_id,
        discussion_num=num,
        comment=body,
        repo_type=repo_type.value,
    )
    print(f"Commented on #{num} in {ANSI.bold(repo_id)}")


@prs_cli.command(
    "close",
    examples=[
        "hf pr close username/my-model 5",
        'hf pr close username/my-model 5 --comment "Superseded by #6."',
    ],
)
def pr_close(
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
    """Close a pull request without merging."""
    api = get_hf_api(token=token)
    api.change_discussion_status(
        repo_id=repo_id,
        discussion_num=num,
        new_status="closed",
        comment=comment,
        repo_type=repo_type.value,
    )
    print(f"Closed #{num} in {ANSI.bold(repo_id)}")


@prs_cli.command(
    "reopen",
    examples=[
        "hf pr reopen username/my-model 5",
    ],
)
def pr_reopen(
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
    """Reopen a closed pull request."""
    api = get_hf_api(token=token)
    api.change_discussion_status(
        repo_id=repo_id,
        discussion_num=num,
        new_status="open",
        comment=comment,
        repo_type=repo_type.value,
    )
    print(f"Reopened #{num} in {ANSI.bold(repo_id)}")


@prs_cli.command(
    "merge",
    examples=[
        "hf pr merge username/my-model 5",
        'hf pr merge username/my-model 5 --comment "Merging, thanks!"',
    ],
)
def pr_merge(
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


@prs_cli.command(
    "diff",
    examples=[
        "hf pr diff username/my-model 5",
    ],
)
def pr_diff(
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
