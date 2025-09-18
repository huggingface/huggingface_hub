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
    hf repo create my-cool-dataset --repo-type=dataset

    # create a private model repo on the Hub
    hf repo create my-cool-model --private
"""

from typing import Annotated, Optional

import typer

from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.utils import logging

from ._cli_utils import (
    ANSI,
    PrivateOpt,
    RepoIdArg,
    RepoType,
    RepoTypeOpt,
    RevisionOpt,
    TokenOpt,
    get_hf_api,
    typer_factory,
)


logger = logging.get_logger(__name__)

repo_cli = typer_factory(help="Manage repos on the Hub.")
tag_app = typer_factory(help="Manage tags for a repo on the Hub.")
repo_cli.add_typer(tag_app, name="tag")


@repo_cli.command("create", help="Create a new repo on the Hub.")
def repo_create(
    repo_id: RepoIdArg,
    repo_type: RepoTypeOpt = RepoType.model,
    space_sdk: Annotated[
        Optional[str],
        typer.Option(
            help="Hugging Face Spaces SDK type. Required when --type is set to 'space'.",
        ),
    ] = None,
    private: PrivateOpt = False,
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


@tag_app.command("create", help="Create a tag for a repo.")
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
    repo_type_str = repo_type.value
    api = get_hf_api(token=token)
    print(f"You are about to create tag {ANSI.bold(tag)} on {repo_type_str} {ANSI.bold(repo_id)}")
    try:
        api.create_tag(repo_id=repo_id, tag=tag, tag_message=message, revision=revision, repo_type=repo_type_str)
    except RepositoryNotFoundError:
        print(f"{repo_type_str.capitalize()} {ANSI.bold(repo_id)} not found.")
        raise typer.Exit(code=1)
    except RevisionNotFoundError:
        print(f"Revision {ANSI.bold(str(revision))} not found.")
        raise typer.Exit(code=1)
    except HfHubHTTPError as e:
        if e.response.status_code == 409:
            print(f"Tag {ANSI.bold(tag)} already exists on {ANSI.bold(repo_id)}")
            raise typer.Exit(code=1)
        raise e
    print(f"Tag {ANSI.bold(tag)} created on {ANSI.bold(repo_id)}")


@tag_app.command("list", help="List tags for a repo.")
def tag_list(
    repo_id: RepoIdArg,
    token: TokenOpt = None,
    repo_type: RepoTypeOpt = RepoType.model,
) -> None:
    repo_type_str = repo_type.value
    api = get_hf_api(token=token)
    try:
        refs = api.list_repo_refs(repo_id=repo_id, repo_type=repo_type_str)
    except RepositoryNotFoundError:
        print(f"{repo_type_str.capitalize()} {ANSI.bold(repo_id)} not found.")
        raise typer.Exit(code=1)
    except HfHubHTTPError as e:
        print(e)
        print(ANSI.red(e.response.text))
        raise typer.Exit(code=1)
    if len(refs.tags) == 0:
        print("No tags found")
        raise typer.Exit(code=0)
    print(f"Tags for {repo_type_str} {ANSI.bold(repo_id)}:")
    for t in refs.tags:
        print(t.name)


@tag_app.command("delete", help="Delete a tag for a repo.")
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
    except RepositoryNotFoundError:
        print(f"{repo_type_str.capitalize()} {ANSI.bold(repo_id)} not found.")
        raise typer.Exit(code=1)
    except RevisionNotFoundError:
        print(f"Tag {ANSI.bold(tag)} not found on {ANSI.bold(repo_id)}")
        raise typer.Exit(code=1)
    print(f"Tag {ANSI.bold(tag)} deleted on {ANSI.bold(repo_id)}")
