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
"""Contains commands to interact with models on the Hugging Face Hub.

Usage:
    # list models on the Hub
    hf models ls

    # list models with a search query
    hf models ls --search "llama"

    # get info about a model
    hf models info Lightricks/LTX-2
"""

import enum
from typing import Annotated, get_args

import typer

from huggingface_hub.errors import CLIError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.hf_api import ExpandModelProperty_T, ModelSort_T
from huggingface_hub.repocard import ModelCard

from ._cli_utils import (
    REPO_LIST_DEFAULT_LIMIT,
    AuthorOpt,
    FilterOpt,
    LimitOpt,
    RevisionOpt,
    SearchOpt,
    TokenOpt,
    api_object_to_dict,
    get_hf_api,
    make_expand_properties_parser,
    typer_factory,
)
from ._file_listing import list_repo_files_cmd
from ._output import out


_EXPAND_PROPERTIES = sorted(get_args(ExpandModelProperty_T))
_SORT_OPTIONS = get_args(ModelSort_T)
ModelSortEnum = enum.Enum("ModelSortEnum", {s: s for s in _SORT_OPTIONS}, type=str)  # type: ignore[misc]


ExpandOpt = Annotated[
    str | None,
    typer.Option(
        help=f"Comma-separated properties to return. When used, only the listed properties (and id) are returned. Example: '--expand=downloads,likes,tags'. Valid: {', '.join(_EXPAND_PROPERTIES)}.",
        callback=make_expand_properties_parser(_EXPAND_PROPERTIES),
    ),
]


models_cli = typer_factory(help="Interact with models on the Hub.")


@models_cli.command(
    "list | ls",
    examples=[
        "hf models ls --sort downloads --limit 10",
        'hf models ls --search "llama" --author meta-llama',
        "hf models ls --num-parameters min:6B,max:128B --sort likes",
        "hf models ls meta-llama/Llama-3.2-1B-Instruct",
        "hf models ls meta-llama/Llama-3.2-1B-Instruct -R",
        "hf models ls meta-llama/Llama-3.2-1B-Instruct --tree -h",
    ],
)
def models_ls(
    repo_id: Annotated[
        str | None,
        typer.Argument(help="Model ID (e.g. `username/repo-name`) to list files from. If omitted, lists models."),
    ] = None,
    search: SearchOpt = None,
    author: AuthorOpt = None,
    filter: FilterOpt = None,
    num_parameters: Annotated[
        str | None,
        typer.Option(help="Filter by parameter count, e.g. 'min:6B,max:128B'."),
    ] = None,
    sort: Annotated[
        ModelSortEnum | None,
        typer.Option(help="Sort results."),
    ] = None,
    limit: LimitOpt = REPO_LIST_DEFAULT_LIMIT,
    expand: ExpandOpt = None,
    human_readable: Annotated[
        bool,
        typer.Option("--human-readable", "-h", help="Show sizes in human readable format (only for listing files)."),
    ] = False,
    as_tree: Annotated[
        bool,
        typer.Option("--tree", help="List files in tree format (only for listing files)."),
    ] = False,
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-R", help="List files recursively (only for listing files)."),
    ] = False,
    revision: RevisionOpt = None,
    token: TokenOpt = None,
) -> None:
    """List models on the Hub, or files in a model repo.

    When called with no argument, lists models on the Hub.
    When called with a model ID, lists files in that model repo.
    """
    if repo_id is not None:
        if search is not None:
            raise typer.BadParameter("Cannot use --search when listing files.")
        if author is not None:
            raise typer.BadParameter("Cannot use --author when listing files.")
        if filter is not None:
            raise typer.BadParameter("Cannot use --filter when listing files.")
        if num_parameters is not None:
            raise typer.BadParameter("Cannot use --num-parameters when listing files.")
        if sort is not None:
            raise typer.BadParameter("Cannot use --sort when listing files.")
        if limit != REPO_LIST_DEFAULT_LIMIT:
            raise typer.BadParameter("Cannot use --limit when listing files.")
        if expand is not None:
            raise typer.BadParameter("Cannot use --expand when listing files.")
        return list_repo_files_cmd(
            repo_id=repo_id,
            repo_type="model",
            human_readable=human_readable,
            as_tree=as_tree,
            recursive=recursive,
            revision=revision,
            token=token,
        )

    if as_tree:
        raise typer.BadParameter("Cannot use --tree when listing models.")
    if recursive:
        raise typer.BadParameter("Cannot use --recursive when listing models.")
    if human_readable:
        raise typer.BadParameter("Cannot use --human-readable when listing models.")
    if revision is not None:
        raise typer.BadParameter("Cannot use --revision when listing models.")
    api = get_hf_api(token=token)
    sort_key = sort.value if sort else None
    results = [
        api_object_to_dict(model_info)
        for model_info in api.list_models(
            filter=filter,
            author=author,
            search=search,
            num_parameters=num_parameters,
            sort=sort_key,
            limit=limit,
            expand=expand,  # type: ignore
        )
    ]
    out.table(results)


@models_cli.command(
    "info",
    examples=[
        "hf models info meta-llama/Llama-3.2-1B-Instruct",
        "hf models info Qwen/Qwen3.5-9B --expand downloads,likes,tags",
    ],
)
def models_info(
    model_id: Annotated[str, typer.Argument(help="The model ID (e.g. `username/repo-name`).")],
    revision: RevisionOpt = None,
    expand: ExpandOpt = None,
    token: TokenOpt = None,
) -> None:
    """Get info about a model on the Hub."""
    api = get_hf_api(token=token)
    try:
        info = api.model_info(repo_id=model_id, revision=revision, expand=expand)  # type: ignore
    except RepositoryNotFoundError as e:
        raise CLIError(f"Model '{model_id}' not found.") from e
    except RevisionNotFoundError as e:
        raise CLIError(f"Revision '{revision}' not found on '{model_id}'.") from e
    out.dict(info)


@models_cli.command(
    "card",
    examples=[
        "hf models card google/gemma-4-31B-it",
        "hf models card google/gemma-4-31B-it --metadata",
        "hf models card google/gemma-4-31B-it --metadata --format json",
        "hf models card google/gemma-4-31B-it --text",
    ],
)
def models_card(
    model_id: Annotated[str, typer.Argument(help="The model ID (e.g. `username/repo-name`).")],
    metadata: Annotated[bool, typer.Option("--metadata", help="Output only the metadata from the card.")] = False,
    text: Annotated[bool, typer.Option("--text", help="Output only the text body (no metadata).")] = False,
    token: TokenOpt = None,
) -> None:
    """Get the model card (README) for a model on the Hub."""
    if metadata and text:
        raise CLIError("--metadata and --text are mutually exclusive.")
    card = ModelCard.load(model_id, token=token)
    if metadata:
        out.dict(card.data.to_dict())
    elif text:
        out.text(card.text)
    else:
        out.text(card.content)
        out.hint(f"Use `hf models card {model_id} --metadata` to extract only the card metadata.")
