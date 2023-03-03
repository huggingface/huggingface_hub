# coding=utf-8
# Copyright 2023-present, the HuggingFace Inc. team.
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
"""Contains utilities to chunked commits (i.e. push changes in multiple commits on a PR)."""
import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, List, Optional, Set

from ._commit_api import CommitOperation, CommitOperationAdd, CommitOperationDelete
from .community import DiscussionComment, DiscussionWithDetails
from .constants import DEFAULT_REVISION
from .utils import BadRequestError, logging
from .utils._cache_manager import _format_size


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from .hf_api import HfApi


class ChunkedCommitException(Exception):
    """Base exception for any exception happening while doing a chunked commit."""


PR_DESCRIPTION_TEMPLATE = """
## {commit_message}

{commit_description}

**Chunked commit ID:** {chunked_commit_id}

Scheduled chunks:

{chunked_commit_strategy}

_This is a PR opened using the `huggingface_hub` library in the context of a chunked commit. PR can be commented as a usual PR. However, please be aware that manually updating the PR description, changing the PR status, or pushing new commits, is not recommended as it might corrupt the commit process. Learn more about chunked commits [in this guide](https://huggingface.co/docs/huggingface_hub/main/guides/upload)._
"""

PR_COMPLETION_COMMENT_TEMPLATE = """
Chunked commit is now completed! You can ping the repo owner to review the changes. This PR can now be commented or modified without risking to corrupt it.

_This is a comment posted using the `huggingface_hub` library in the context of a chunked commit. Learn more about chunked commits [in this guide](https://huggingface.co/docs/huggingface_hub/main/guides/upload)._
"""

PR_CLOSING_COMMENT_TEMPLATE = """
`create_pr=False` has been passed so PR is automatically merged.

_This is a comment posted using the `huggingface_hub` library in the context of a chunked commit. Learn more about chunked commits [in this guide](https://huggingface.co/docs/huggingface_hub/main/guides/upload)._
"""

PR_CLOSE_COMMENT_FAILURE_NO_CHANGES_TEMPLATE = """
Cannot merge Pull Requests as no changes are associated. This PR will be closed automatically.

_This is a comment posted using the `huggingface_hub` library in the context of a chunked commit. Learn more about chunked commits [in this guide](https://huggingface.co/docs/huggingface_hub/main/guides/upload)._
"""

PR_CLOSE_COMMENT_FAILURE_BAD_REQUEST_TEMPLATE = """
An error occurred while trying to merge the Pull Request: `{error_message}`.

_This is a comment posted using the `huggingface_hub` library in the context of a chunked commit. Learn more about chunked commits [in this guide](https://huggingface.co/docs/huggingface_hub/main/guides/upload)._
"""


STEP_ID_REGEX = re.compile(r"- \[(?P<completed>[ |x])\].*(?P<step_id>[a-fA-F0-9]{64})", flags=re.MULTILINE)


@dataclass
class ChunkedCommitStep:
    """Dataclass containing a list of CommitOperation to commit at once.

    A [`ChunkedCommitStep`] is one atomic part of a [`ChunkedCommitStrategy`].
    Each step is identified by its own deterministic ID based on the list of commit operations (hexadecimal sha256).
    ID is persistent between re-runs if the list of commits is kept the same.
    """

    num: int  # order of the step in the strategy # TODO: this is ridiculous, remove it ASAP
    operations: List[CommitOperation]

    id: str = field(init=False)
    completed: bool = False

    def __post_init__(self) -> None:
        if len(self.operations) == 0:
            raise ValueError("A ChunkedCommitStep must have at least 1 commit operation, got 0.")

        # Generate step id
        sha = sha256()
        for op in self.operations:
            if isinstance(op, CommitOperationAdd):
                sha.update(op.path_in_repo.encode())
                sha.update(op.upload_info.sha256)
            elif isinstance(op, CommitOperationDelete):
                sha.update(op.path_in_repo.encode())
                sha.update(str(op.is_folder).encode())
            else:
                NotImplementedError()
        self.id = sha.hexdigest()

    def __str__(self) -> str:
        """Format a step for PR description.

        Formatting can be changed in the future as long as it is single line, start with `- [ ]`/`- [x]` and contains
        `self.id`. Must be able to match `STEP_ID_REGEX`.
        """
        additions = [op for op in self.operations if isinstance(op, CommitOperationAdd)]
        file_deletions = [op for op in self.operations if isinstance(op, CommitOperationDelete) and not op.is_folder]
        folder_deletions = [op for op in self.operations if isinstance(op, CommitOperationDelete) and op.is_folder]
        return (
            f"- [{'x' if self.completed else ' '}] Upload {len(additions)} file(s) "
            f"totalling {_format_size(sum(add.upload_info.size for add in additions))}, "
            f"delete {len(file_deletions)} file(s) and {len(folder_deletions)} folder(s)"
            f" ({self.id})"
        )


@dataclass
class ChunkedCommitStrategy:
    """Dataclass containing a list of [`ChunkedCommitStep`] to commit.

    A strategy is identified by its own deterministic ID based on the list of its steps (hexadecimal sha256).
    ID is persistent between re-runs if the list of operations is kept the same.
    """

    steps: List[ChunkedCommitStep]

    id: str = field(init=False)

    def __post_init__(self) -> None:
        if len(self.steps) == 0:
            raise ValueError("A ChunkedCommitStrategy must have at least 1 step, got 0.")

        # Generate strategy id
        sha = sha256()
        for step in self.steps:
            sha.update("new step".encode())
            sha.update(step.id.encode())
        self.id = sha.hexdigest()


def commit_in_chunks(
    *,
    api: "HfApi",
    repo_id: str,
    operations: List[CommitOperation],
    commit_message: str,
    commit_description: Optional[str] = None,
    token: Optional[str] = None,
    repo_type: Optional[str] = None,
    create_pr: Optional[bool] = None,
    num_threads: int = 5,
    verbose: bool = True,
    # TODO Cannot revision: Optional[str] = None,
    # TODO Cannot parent_commit: Optional[str] = None,
) -> None:
    """Push changes to the Hub in multiple commits.

    Commits are chained and pushed to a draft PR branch. If the upload fails or gets interrupted, it can be resumed.
    Progress is tracked in the PR description. If `create_pr=False` is passed, the PR is merged automatically at the
    end of the process.

    A strategy (see [`ChunkedCommitStrategy`]) is planned to schedule that consists in a list of [`ChunkedCommitStep`].
    Each [`ChunkedCommitStep`] is a list of commit operations to commit together. The process to split the commit
    operations into steps is automatically done. TODO: explain in more details.

    TODO warning: operations must be a list
    TODO maybe take directly the strategy as input
    TODO maybe separate from `create_commit` (avoid problems with the return type)
    TODO maybe add a "split_strategy" to `upload_folder` ("auto", "1 by 1", "100 by 100", "not more than 10GB per commit",...)
    TODO what to do with `revision`?
    TODO what to do with `parent_commit`?
    """
    if verbose:
        logger.setLevel("INFO")

    # 1. Define strategy (e.g. split the operations in multiple commits)
    logger.info(f"Got {len(operations)} operations to commit.")
    strategy = _plan_commits(operations)
    logger.info(
        f"Commit strategy has been determined. Will consists in {len(strategy.steps)} step(s) (ID: {strategy.id})."
    )

    # 2. Check if an existing PR is doing the same
    for discussion in api.get_repo_discussions(repo_id=repo_id, repo_type=repo_type, token=token):
        # search for a draft PR with strategy ID
        if discussion.is_pull_request and discussion.status == "draft" and strategy.id in discussion.title:
            pr = api.get_discussion_details(
                repo_id=repo_id, discussion_num=discussion.num, repo_type=repo_type, token=token
            )
            logger.info(f"PR already exists: {pr.url}. Will resume process where it stopped.")
            break
    else:
        # did not find a PR matching the strategy ID
        pr = _create_pull_request(
            api,
            repo_id=repo_id,
            commit_message=commit_message,
            commit_description=commit_description,
            strategy=strategy,
            token=token,
            repo_type=repo_type,
        )
        logger.info(f"New PR created: {pr.url}")

    # 3. Parse PR description to check consistency with strategy (e.g. same steps are scheduled)
    for event in pr.events:
        if isinstance(event, DiscussionComment):
            pr_comment = event
            break
    else:
        raise ChunkedCommitException(f"PR #{pr.num} must have at least 1 comment")

    description_steps: Set[str] = {match[1] for match in STEP_ID_REGEX.findall(pr_comment.content)}
    if len(description_steps) != len(strategy.steps):
        raise ChunkedCommitException(
            f"Corrupted chunked commit PR #{pr.num}: got {len(description_steps)} steps in"
            f" description but {len(strategy.steps)} in strategy."
        )
    for step in strategy.steps:
        if step.id not in description_steps:
            raise ChunkedCommitException(
                f"Corrupted chunked commit PR #{pr.num}: expected step {step.id} but didn't find"
                f" it (have {', '.join(description_steps)})."
            )

    # 4. Retrieve commit history (and check consistency)
    commits_on_main_branch = {
        commit.commit_id
        for commit in api.list_repo_commits(
            repo_id=repo_id, repo_type=repo_type, token=token, revision=DEFAULT_REVISION
        )
    }
    pr_commits = list(
        reversed(
            [
                commit
                for commit in api.list_repo_commits(
                    repo_id=repo_id, repo_type=repo_type, token=token, revision=pr.git_reference
                )
                if commit.commit_id not in commits_on_main_branch
            ]
        )
    )
    if len(pr_commits) > 0:
        logger.info(f"Found {len(pr_commits)} existing commits on the PR.")

    # At this point `pr_commits` is a list of commits pushed to the PR, sorted by oldest commit first.
    # We except all of these commits (if any) to have a step_id as title. Raise exception if an unexpected commit has
    # been pushed or if the commit order is not respected.
    # TODO: relax this rule if we want parallel commits to work as well
    if len(pr_commits) > len(strategy.steps):
        raise ChunkedCommitException(
            f"Corrupted chunked commit PR #{pr.num}: scheduled {len(strategy.steps)} steps but"
            f" {len(pr_commits)} commits have already been pushed to the PR."
        )

    # Check completed steps
    for commit, step in zip(pr_commits, strategy.steps):
        if commit.title != step.id:
            if commit.title in (step.id for step in strategy.steps):  # means wrong order
                raise ChunkedCommitException(
                    f"Corrupted chunked commit PR #{pr.num}: expected commit '{step.id}' but got"
                    f" '{commit.title}'. Commits have been pushed to the PR in a wrong order."
                )
            else:  # means commit not part of the strategy
                raise ChunkedCommitException(
                    f"Corrupted chunked commit PR #{pr.num}: unexpected commit '{commit.title}'"
                    f" is not part of the strategy (expected commit '{step.id}')."
                )

        # Commit is in history => means it has been completed
        step.completed = True
        logger.info(f"  ({step.num+1}/{len(strategy.steps)}) step {step.id}: completed.")

    # 5. Push remaining commits to the PR + update description
    parent_commit = pr_commits[-1].commit_id if len(pr_commits) > 0 else None
    for step in strategy.steps:
        if step.completed:  # skip pushed commits
            continue

        # Push new commit
        parent_commit = api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message=step.id,
            revision=pr.git_reference,
            chunked_commits=False,
            num_threads=num_threads,
            operations=step.operations,
            create_pr=False,
            parent_commit=parent_commit,
        ).oid
        step.completed = True
        logger.info(f"  ({step.num+1}/{len(strategy.steps)}) step {step.id}: completed.")

        # Update PR description
        api.edit_discussion_comment(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            discussion_num=pr.num,
            comment_id=pr_comment.id,
            new_content=_generate_comment(
                commit_message=commit_message, commit_description=commit_description, strategy=strategy
            ),
        )
    logger.info("All steps have been pushed.")

    # 6. Update PR (and merge)
    api.rename_discussion(
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        discussion_num=pr.num,
        new_title=commit_message,
    )
    api.change_discussion_status(
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        discussion_num=pr.num,
        new_status="open",
        comment=PR_COMPLETION_COMMENT_TEMPLATE,
    )
    logger.info("PR has been renamed and set an 'open' for reviews.")

    if not create_pr:  # User don't want a PR => merge it
        try:
            api.merge_pull_request(
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                discussion_num=pr.num,
                comment=PR_CLOSING_COMMENT_TEMPLATE,
            )
            logger.info("PR has been automatically merged.")
        except BadRequestError as error:
            if error.server_message is not None and "no associated changes" in error.server_message:
                # PR cannot be merged as no changes are associated. We close the PR without merging with a comment to
                # explain.
                api.change_discussion_status(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token,
                    discussion_num=pr.num,
                    comment=PR_CLOSE_COMMENT_FAILURE_NO_CHANGES_TEMPLATE,
                    new_status="closed",
                )
                logger.warning("Couldn't merge the PR: no associated changes.")
            else:
                # PR cannot be merged for another reason (conflicting files for example). We comment the PR to explain
                # and re-raise the exception.
                api.comment_discussion(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token,
                    discussion_num=pr.num,
                    comment=PR_CLOSE_COMMENT_FAILURE_BAD_REQUEST_TEMPLATE.format(error_message=error.server_message),
                )
                raise ChunkedCommitException(
                    f"Couldn't merge Pull Request in chunked commit: {error.server_message}"
                ) from error


def _create_pull_request(
    api: "HfApi",
    repo_id: str,
    commit_message: str,
    commit_description: Optional[str],
    strategy: ChunkedCommitStrategy,
    token: Optional[str],
    repo_type: Optional[str],
) -> DiscussionWithDetails:
    return api.create_pull_request(
        repo_id=repo_id,
        title=f"[WIP] {commit_message} (chunked commit {strategy.id})",
        description=_generate_comment(
            commit_message=commit_message, commit_description=commit_description, strategy=strategy
        ),
        token=token,
        repo_type=repo_type,
    )


def _generate_comment(
    commit_message: str,
    commit_description: Optional[str],
    strategy: ChunkedCommitStrategy,
) -> str:
    return PR_DESCRIPTION_TEMPLATE.format(
        commit_message=commit_message,
        commit_description=commit_description or "",
        chunked_commit_id=strategy.id,
        chunked_commit_strategy="\n".join(str(step) for step in strategy.steps),
    )


def _plan_commits(operations: List[CommitOperation]) -> ChunkedCommitStrategy:
    # Dumb strategy for now:
    # - 1 step for all deletions
    # - 1 step every 10 added files
    # - order is not preserved
    # TODO: review strategy
    # TODO: maybe as first-class citizen
    # TODO: should we preserve order? I'd say deletes first, then additions
    steps = []

    # Start by deleting everything
    delete_operations = [op for op in operations if isinstance(op, CommitOperationDelete)]
    if len(delete_operations) > 0:
        steps.append(ChunkedCommitStep(num=0, operations=delete_operations))

    # Then upload 10 files by 10 files
    step_ops = []
    for op in operations:
        if isinstance(op, CommitOperationAdd):
            step_ops.append(op)
        if len(step_ops) >= 10:
            steps.append(ChunkedCommitStep(num=len(steps), operations=step_ops))
            step_ops = []
    if len(step_ops) > 0:
        steps.append(ChunkedCommitStep(num=len(steps), operations=step_ops))
    return ChunkedCommitStrategy(steps=steps)
