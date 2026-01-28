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
"""Contains commands to interact with datasets on the Hugging Face Hub.

Usage:
    # list datasets on the Hub
    hf datasets ls

    # list datasets with a search query
    hf datasets ls --search "code"

    # get info about a dataset
    hf datasets info HuggingFaceFW/fineweb
"""

import enum
import json
from typing import Annotated, Any, Optional, get_args

import typer

from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.hf_api import DatasetSort_T, ExpandDatasetProperty_T
from huggingface_hub.utils import ANSI

from ._cli_utils import (
    AuthorOpt,
    FilterOpt,
    FormatOpt,
    LimitOpt,
    OutputFormat,
    QuietOpt,
    RevisionOpt,
    SearchOpt,
    TokenOpt,
    api_object_to_dict,
    get_hf_api,
    make_expand_properties_parser,
    print_list_output,
    typer_factory,
)


_EXPAND_PROPERTIES = sorted(get_args(ExpandDatasetProperty_T))
_SORT_OPTIONS = get_args(DatasetSort_T)
DatasetSortEnum = enum.Enum("DatasetSortEnum", {s: s for s in _SORT_OPTIONS}, type=str)  # type: ignore[misc]


ExpandOpt = Annotated[
    Optional[str],
    typer.Option(
        help=f"Comma-separated properties to expand. Example: '--expand=downloads,likes,tags'. Valid: {', '.join(_EXPAND_PROPERTIES)}.",
        callback=make_expand_properties_parser(_EXPAND_PROPERTIES),
    ),
]


datasets_cli = typer_factory(help="Interact with datasets on the Hub.")


@datasets_cli.command("ls")
def datasets_ls(
    search: SearchOpt = None,
    author: AuthorOpt = None,
    filter: FilterOpt = None,
    sort: Annotated[
        Optional[DatasetSortEnum],
        typer.Option(help="Sort results."),
    ] = None,
    limit: LimitOpt = 10,
    expand: ExpandOpt = None,
    format: FormatOpt = OutputFormat.table,
    quiet: QuietOpt = False,
    token: TokenOpt = None,
) -> None:
    """List datasets on the Hub."""
    api = get_hf_api(token=token)
    sort_key = sort.value if sort else None
    results = [
        api_object_to_dict(dataset_info)
        for dataset_info in api.list_datasets(
            filter=filter, author=author, search=search, sort=sort_key, limit=limit, expand=expand
        )
    ]

    def row_fn(item: dict[str, Any]) -> list[str]:
        repo_id = str(item.get("id", ""))
        author = str(item.get("author", "")) or (repo_id.split("/")[0] if "/" in repo_id else "")
        return [
            repo_id,
            author,
            str(item.get("downloads", "") or ""),
            str(item.get("likes", "") or ""),
        ]

    print_list_output(
        items=results,
        format=format,
        quiet=quiet,
        id_key="id",
        headers=["ID", "AUTHOR", "DOWNLOADS", "LIKES"],
        row_fn=row_fn,
    )


@datasets_cli.command("info")
def datasets_info(
    dataset_id: Annotated[str, typer.Argument(help="The dataset ID (e.g. `username/repo-name`).")],
    revision: RevisionOpt = None,
    expand: ExpandOpt = None,
    token: TokenOpt = None,
) -> None:
    """Get info about a dataset on the Hub."""
    api = get_hf_api(token=token)
    try:
        info = api.dataset_info(repo_id=dataset_id, revision=revision, expand=expand)  # type: ignore[arg-type]
    except RepositoryNotFoundError:
        print(f"Dataset {ANSI.bold(dataset_id)} not found.")
        raise typer.Exit(code=1)
    except RevisionNotFoundError:
        print(f"Revision {ANSI.bold(str(revision))} not found on {ANSI.bold(dataset_id)}.")
        raise typer.Exit(code=1)
    print(json.dumps(api_object_to_dict(info), indent=2))
