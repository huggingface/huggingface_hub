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
import json
from typing import Annotated, Optional, get_args

import typer

from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.hf_api import ExpandModelProperty_T, ModelSort_T
from huggingface_hub.utils import ANSI

from ._cli_utils import (
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


_EXPAND_PROPERTIES = sorted(get_args(ExpandModelProperty_T))
_SORT_OPTIONS = get_args(ModelSort_T)
ModelSortEnum = enum.Enum("ModelSortEnum", {s: s for s in _SORT_OPTIONS}, type=str)  # type: ignore[misc]


ExpandOpt = Annotated[
    Optional[str],
    typer.Option(
        help=f"Comma-separated properties to expand. Example: '--expand=downloads,likes,tags'. Valid: {', '.join(_EXPAND_PROPERTIES)}.",
        callback=make_expand_properties_parser(_EXPAND_PROPERTIES),
    ),
]


models_cli = typer_factory(help="Interact with models on the Hub.")


_MODELS_EPILOG = """\
EXAMPLES
  $ hf models ls --sort downloads --limit 10
  $ hf models ls --search "llama" --author meta-llama
  $ hf models info meta-llama/Llama-3.2-1B-Instruct
  $ hf models info gpt2 --expand downloads,likes,tags

LEARN MORE
  Use `hf <command> --help` for more information about a command.
  Read the documentation at https://huggingface.co/docs/huggingface_hub/en/guides/cli#hf-models
"""


@models_cli.callback(epilog=_MODELS_EPILOG, invoke_without_command=True)
def models_callback(ctx: typer.Context) -> None:
    """Interact with models on the Hub."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@models_cli.command("ls")
def models_ls(
    search: SearchOpt = None,
    author: AuthorOpt = None,
    filter: FilterOpt = None,
    sort: Annotated[
        Optional[ModelSortEnum],
        typer.Option(help="Sort results."),
    ] = None,
    limit: LimitOpt = 10,
    expand: ExpandOpt = None,
    token: TokenOpt = None,
) -> None:
    """List models on the Hub."""
    api = get_hf_api(token=token)
    sort_key = sort.value if sort else None
    results = [
        api_object_to_dict(model_info)
        for model_info in api.list_models(
            filter=filter, author=author, search=search, sort=sort_key, limit=limit, expand=expand
        )
    ]
    print(json.dumps(results, indent=2))


@models_cli.command("info")
def models_info(
    model_id: Annotated[str, typer.Argument(help="The model ID (e.g. `username/repo-name`).")],
    revision: RevisionOpt = None,
    expand: ExpandOpt = None,
    token: TokenOpt = None,
) -> None:
    """Get info about a model on the Hub."""
    api = get_hf_api(token=token)
    try:
        info = api.model_info(repo_id=model_id, revision=revision, expand=expand)  # type: ignore[arg-type]
    except RepositoryNotFoundError:
        print(f"Model {ANSI.bold(model_id)} not found.")
        raise typer.Exit(code=1)
    except RevisionNotFoundError:
        print(f"Revision {ANSI.bold(str(revision))} not found on {ANSI.bold(model_id)}.")
        raise typer.Exit(code=1)
    print(json.dumps(api_object_to_dict(info), indent=2))
