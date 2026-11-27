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

from huggingface_hub.errors import CLIError

from ._cli_utils import TokenOpt, get_hf_api, typer_factory
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


ItemArg = Annotated[
    str,
    Argument(
        help="The item to update: a repo ID (e.g. `my-org/my-model`), a collection slug or a blog ID (`my-org/my-post`).",
    ),
]
ItemTypeOpt = Annotated[ResourceGroupItemType, Option("--type", help="The type of the item.")]

resource_groups_cli = typer_factory(help="Manage resource groups of an organization on the Hub.")


@resource_groups_cli.command(
    "assign",
    examples=[
        "hf resource-groups assign 66980ecfc1e12a49c8f0e42d my-org/my-model",
        "hf resource-groups assign 66980ecfc1e12a49c8f0e42d my-org/my-data --type dataset",
        "hf resource-groups assign 66980ecfc1e12a49c8f0e42d my-org/my-collection-64f9a55bb3115b4f513ec026 --type collection",
    ],
)
def assign(
    resource_group_id: Annotated[
        str,
        Argument(help="The 24-character hexadecimal ID of the resource group (last part of its settings URL)."),
    ],
    item: ItemArg,
    item_type: ItemTypeOpt = ResourceGroupItemType.model,
    token: TokenOpt = None,
) -> None:
    """Assign an item to a resource group. An item already in another group is moved to this one."""
    if not re.fullmatch(r"[0-9a-f]{24}", resource_group_id):
        raise CLIError(
            f"Invalid resource group ID '{resource_group_id}': expected a 24-character hexadecimal ID, as found at the"
            " end of the group's URL in your organization settings."
        )
    get_hf_api(token=token).assign_resource_group_item(resource_group_id, item, item_type=item_type.value)
    out.result("Assigned to resource group", item=item, type=item_type.value, resource_group_id=resource_group_id)


@resource_groups_cli.command(
    "unassign",
    examples=[
        "hf resource-groups unassign my-org/my-model",
        "hf resource-groups unassign my-org/my-data --type dataset",
    ],
)
def unassign(
    item: ItemArg,
    item_type: ItemTypeOpt = ResourceGroupItemType.model,
    token: TokenOpt = None,
) -> None:
    """Remove an item from its resource group. An item belongs to at most one group, so no group is needed."""
    get_hf_api(token=token).unassign_resource_group_item(item, item_type=item_type.value)
    out.result("Removed from resource group", item=item, type=item_type.value, resource_group_id=None)
