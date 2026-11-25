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
"""Contains commands to manage resource groups on the Hugging Face Hub."""

import enum
import re
from typing import Annotated

import click

from huggingface_hub.errors import CLIError, HfHubHTTPError, HFValidationError

from ._cli_utils import TokenOpt, get_hf_api, typer_factory
from ._errors import format_known_exception
from ._framework import Argument, Option
from ._output import out


class ResourceGroupItemType(str, enum.Enum):
    model = "model"
    dataset = "dataset"
    space = "space"
    bucket = "bucket"
    container = "container"
    collection = "collection"
    blog = "blog"


ItemsArg = Annotated[
    list[str],
    Argument(
        help="The items to update: repo IDs (e.g. `my-org/my-model`), collection slugs or blog IDs (`my-org/my-post`).",
    ),
]
ItemTypeOpt = Annotated[
    ResourceGroupItemType,
    Option("--type", help="The type of the items. All items of one call share the same type."),
]

resource_groups_cli = typer_factory(help="Manage resource groups of an organization on the Hub.")


@resource_groups_cli.command(
    "assign",
    examples=[
        "hf resource-groups assign 66980ecfc1e12a49c8f0e42d my-org/my-model",
        "hf resource-groups assign 66980ecfc1e12a49c8f0e42d my-org/data-1 my-org/data-2 --type dataset",
        "hf resource-groups assign 66980ecfc1e12a49c8f0e42d my-org/my-collection-64f9a55bb3115b4f513ec026 --type collection",
    ],
)
def assign(
    resource_group_id: Annotated[
        str,
        Argument(help="The 24-character hexadecimal ID of the resource group (last part of its settings URL)."),
    ],
    items: ItemsArg,
    item_type: ItemTypeOpt = ResourceGroupItemType.model,
    token: TokenOpt = None,
) -> None:
    """Assign items to a resource group. An item already in another group is moved to this one."""
    if not re.fullmatch(r"[0-9a-f]{24}", resource_group_id):
        raise CLIError(
            f"Invalid resource group ID '{resource_group_id}': expected a 24-character hexadecimal ID, as found at the"
            " end of the group's URL in your organization settings."
        )
    _update_items(items, resource_group_id, item_type, token, message="Assigned to resource group")


@resource_groups_cli.command(
    "unassign",
    examples=[
        "hf resource-groups unassign my-org/my-model",
        "hf resource-groups unassign my-org/data-1 my-org/data-2 --type dataset",
    ],
)
def unassign(
    items: ItemsArg,
    item_type: ItemTypeOpt = ResourceGroupItemType.model,
    token: TokenOpt = None,
) -> None:
    """Remove items from their resource group. An item belongs to at most one group, so no group is needed."""
    _update_items(items, None, item_type, token, message="Removed from resource group")


def _update_items(
    items: list[str],
    resource_group_id: str | None,
    item_type: ResourceGroupItemType,
    token: str | None,
    message: str,
) -> None:
    """Update each item with one API call. Keeps going on failure and exits with code 1 if any item failed."""
    api = get_hf_api(token=token)
    num_failed = 0
    for item in items:
        try:
            if item_type == ResourceGroupItemType.collection:
                api.update_collection_resource_group(item, resource_group_id)
            elif item_type == ResourceGroupItemType.blog:
                api.update_blog_resource_group(item, resource_group_id)
            else:
                api.update_repo_resource_group(item, resource_group_id, repo_type=item_type.value)
        except (HfHubHTTPError, HFValidationError) as error:
            num_failed += 1
            out.error(f"{item}: {format_known_exception(error) or error}")
            continue
        out.result(message, item=item, type=item_type.value, resource_group_id=resource_group_id)
    if num_failed > 0:
        raise click.exceptions.Exit(code=1)
