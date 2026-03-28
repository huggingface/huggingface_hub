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
"""Contains commands to interact with papers on the Hugging Face Hub.

Usage:
    # list daily papers (most recently submitted)
    hf papers ls

    # list trending papers
    hf papers ls --sort=trending

    # list papers from a specific date, ordered by upvotes
    hf papers ls --date=2025-01-23

    # list today's papers, ordered by upvotes
    hf papers ls --date=today

    # list papers from a specific week
    hf papers ls --week=2025-W09

    # list papers by a specific submitter
    hf papers ls --submitter=someuser

    # search papers
    hf papers search "vision language"

    # get info about a paper
    hf papers info 2502.08025

    # read a paper as markdown
    hf papers read 2502.08025
"""

import datetime
import enum
import json
from typing import Annotated, Optional, get_args

import typer

from huggingface_hub.errors import CLIError, HfHubHTTPError
from huggingface_hub.hf_api import DailyPapersSort_T

from ._cli_utils import (
    FormatOpt,
    LimitOpt,
    OutputFormat,
    QuietOpt,
    TokenOpt,
    _format_cell,
    api_object_to_dict,
    get_hf_api,
    print_list_output,
    typer_factory,
)


_SORT_OPTIONS = get_args(DailyPapersSort_T)
PaperSortEnum = enum.Enum("PaperSortEnum", {s: s for s in _SORT_OPTIONS}, type=str)  # type: ignore[misc]


def _parse_date(value: Optional[str]) -> Optional[str]:
    """Parse date option, converting 'today' to current date."""
    if value is None:
        return None
    if value.lower() == "today":
        return datetime.date.today().isoformat()
    return value


papers_cli = typer_factory(help="Interact with papers on the Hub.")


@papers_cli.command(
    "list | ls",
    examples=[
        "hf papers ls",
        "hf papers ls --sort trending",
        "hf papers ls --date 2025-01-23",
        "hf papers ls --week 2025-W09",
        "hf papers ls --submitter akhaliq",
        "hf papers ls --format json",
    ],
)
def papers_ls(
    date: Annotated[
        Optional[str],
        typer.Option(
            help="Date in ISO format (YYYY-MM-DD) or 'today'.",
            callback=_parse_date,
        ),
    ] = None,
    week: Annotated[
        Optional[str],
        typer.Option(help="ISO week to filter by, e.g. '2025-W09'."),
    ] = None,
    month: Annotated[
        Optional[str],
        typer.Option(help="Month to filter by in ISO format (YYYY-MM), e.g. '2025-02'."),
    ] = None,
    submitter: Annotated[
        Optional[str],
        typer.Option(help="Filter by username of the submitter."),
    ] = None,
    sort: Annotated[
        Optional[PaperSortEnum],
        typer.Option(help="Sort results."),
    ] = None,
    limit: LimitOpt = 50,
    format: FormatOpt = OutputFormat.table,
    quiet: QuietOpt = False,
    token: TokenOpt = None,
) -> None:
    """List daily papers on the Hub."""
    api = get_hf_api(token=token)
    sort_key = sort.value if sort else None
    results = [
        api_object_to_dict(paper_info)
        for paper_info in api.list_daily_papers(
            date=date,
            week=week,
            month=month,
            submitter=submitter,
            sort=sort_key,
            limit=limit,
        )
    ]
    _HEADERS = ["id", "title", "upvotes", "comments", "published_at", "submitted_by"]

    def _paper_row(item: dict) -> list[str]:
        submitted_by = item.get("submitted_by") or {}
        submitter_name = submitted_by.get("fullname") or submitted_by.get("username") or ""
        return [
            item.get("id", ""),
            _format_cell(item.get("title", ""), max_len=60),
            str(item.get("upvotes", "")),
            str(item.get("comments", "")),
            _format_cell(item.get("published_at", "")),
            submitter_name,
        ]

    print_list_output(
        results,
        format=format,
        quiet=quiet,
        id_key="id",
        headers=_HEADERS,
        row_fn=_paper_row,
        alignments={"upvotes": "right", "comments": "right"},
    )


@papers_cli.command(
    "search",
    examples=[
        'hf papers search "vision language"',
        'hf papers search "attention mechanism" --limit 10',
        'hf papers search "diffusion" --format json',
    ],
)
def papers_search(
    query: Annotated[str, typer.Argument(help="Search query string.")],
    limit: LimitOpt = 20,
    format: FormatOpt = OutputFormat.table,
    quiet: QuietOpt = False,
    token: TokenOpt = None,
) -> None:
    """Search papers on the Hub."""
    api = get_hf_api(token=token)
    results = [api_object_to_dict(paper_info) for paper_info in api.list_papers(query=query, limit=limit)]
    _HEADERS = ["id", "title", "upvotes", "published_at"]

    def _paper_row(item: dict) -> list[str]:
        return [
            item.get("id", ""),
            _format_cell(item.get("title", ""), max_len=70),
            str(item.get("upvotes", "")),
            _format_cell(item.get("published_at", "")),
        ]

    print_list_output(
        results,
        format=format,
        quiet=quiet,
        id_key="id",
        headers=_HEADERS,
        row_fn=_paper_row,
        alignments={"upvotes": "right"},
    )


@papers_cli.command(
    "info",
    examples=[
        "hf papers info 2601.15621",
    ],
)
def papers_info(
    paper_id: Annotated[str, typer.Argument(help="The arXiv paper ID (e.g. '2502.08025').")],
    token: TokenOpt = None,
) -> None:
    """Get info about a paper on the Hub. Output is in JSON format."""
    api = get_hf_api(token=token)
    try:
        info = api.paper_info(id=paper_id)
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            raise CLIError(f"Paper '{paper_id}' not found on the Hub.") from e
        raise
    print(json.dumps(api_object_to_dict(info), indent=2))


@papers_cli.command(
    "read",
    examples=[
        "hf papers read 2601.15621",
    ],
)
def papers_read(
    paper_id: Annotated[str, typer.Argument(help="The arXiv paper ID (e.g. '2502.08025').")],
    token: TokenOpt = None,
) -> None:
    """Read a paper as markdown."""
    api = get_hf_api(token=token)
    try:
        content = api.read_paper(id=paper_id)
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            raise CLIError(f"Paper '{paper_id}' not found on the Hub.") from e
        raise
    print(content)
