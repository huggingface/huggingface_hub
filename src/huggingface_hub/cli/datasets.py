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
from typing import Annotated, Optional, Union, get_args

import typer

from huggingface_hub._datasets_parquet import list_dataset_parquet_entries
from huggingface_hub._datasets_sql import execute_raw_sql_query, format_sql_result
from huggingface_hub.errors import CLIError, EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.hf_api import DatasetSort_T, ExpandDatasetProperty_T
from huggingface_hub.utils import tabulate

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


@datasets_cli.command(
    "ls",
    examples=[
        "hf datasets ls",
        "hf datasets ls --sort downloads --limit 10",
        'hf datasets ls --search "code"',
    ],
)
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
    print_list_output(results, format=format, quiet=quiet)


@datasets_cli.command(
    "info",
    examples=[
        "hf datasets info HuggingFaceFW/fineweb",
        "hf datasets info my-dataset --expand downloads,likes,tags",
    ],
)
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
    except RepositoryNotFoundError as e:
        raise CLIError(f"Dataset '{dataset_id}' not found.") from e
    except RevisionNotFoundError as e:
        raise CLIError(f"Revision '{revision}' not found on '{dataset_id}'.") from e
    print(json.dumps(api_object_to_dict(info), indent=2))


@datasets_cli.command(
    "parquet",
    examples=[
        "hf datasets parquet cfahlgren1/hub-stats",
        "hf datasets parquet cfahlgren1/hub-stats --subset models",
        "hf datasets parquet cfahlgren1/hub-stats --split train",
        "hf datasets parquet cfahlgren1/hub-stats --format json",
    ],
)
def datasets_parquet(
    dataset_id: Annotated[str, typer.Argument(help="The dataset ID (e.g. `username/repo-name`).")],
    subset: Annotated[Optional[str], typer.Option("--subset", help="Filter parquet entries by subset/config.")] = None,
    split: Annotated[Optional[str], typer.Option(help="Filter parquet entries by split.")] = None,
    format: Annotated[
        OutputFormat,
        typer.Option(help="Output format.", case_sensitive=False),
    ] = OutputFormat.table,
    token: TokenOpt = None,
) -> None:
    """List parquet file URLs available for a dataset."""
    api = get_hf_api(token=token)
    effective_token = api.token

    try:
        entries = list_dataset_parquet_entries(repo_id=dataset_id, token=effective_token, config=subset, split=split)
    except RepositoryNotFoundError as e:
        raise CLIError(f"Dataset '{dataset_id}' not found.") from e
    except EntryNotFoundError as e:
        raise CLIError(str(e)) from e
    rows: list[list[Union[str, int]]] = [[entry.config, entry.split, entry.url] for entry in entries]

    if format == OutputFormat.table:
        typer.echo(tabulate(rows=rows, headers=["SUBSET", "SPLIT", "URL"]))
        return

    if format == OutputFormat.json:
        typer.echo(
            json.dumps(
                [{"subset": entry.config, "split": entry.split, "url": entry.url} for entry in entries],
                indent=2,
            )
        )
        return


@datasets_cli.command(
    "sql",
    examples=[
        "hf datasets sql \"SELECT COUNT(*) AS rows FROM read_parquet('https://huggingface.co/api/datasets/cfahlgren1/hub-stats/parquet/models/train/0.parquet')\"",
        "hf datasets sql \"SELECT * FROM read_parquet('https://huggingface.co/api/datasets/cfahlgren1/hub-stats/parquet/models/train/0.parquet') LIMIT 5\" --format json",
    ],
)
def datasets_sql(
    sql: Annotated[str, typer.Argument(help="Raw SQL query to execute.")],
    format: Annotated[
        OutputFormat,
        typer.Option(help="Output format.", case_sensitive=False),
    ] = OutputFormat.table,
    token: TokenOpt = None,
) -> None:
    """Execute a raw SQL query with DuckDB against dataset parquet URLs."""
    api = get_hf_api(token=token)
    effective_token = api.token
    try:
        result = execute_raw_sql_query(sql_query=sql, token=effective_token, output_format=format.value)
    except (ImportError, ValueError) as e:
        raise CLIError(str(e)) from e
    typer.echo(format_sql_result(result=result, output_format=format.value))
