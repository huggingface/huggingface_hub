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
from typing import TYPE_CHECKING, Annotated, cast, get_args

import click

from huggingface_hub.errors import CLIError, HfHubHTTPError, HFValidationError
from huggingface_hub.hf_api import ResourceGroupAutoJoinScope_T, ResourceGroupRole_T, ResourceGroupSpendLimit_T

from ._cli_utils import TokenOpt, get_hf_api, typer_factory
from ._errors import format_known_exception
from ._framework import Argument, Option
from ._output import out


if TYPE_CHECKING:
    from huggingface_hub import HfApi


class ResourceGroupItemType(str, enum.Enum):
    model = "model"
    dataset = "dataset"
    space = "space"
    bucket = "bucket"
    container = "container"
    collection = "collection"
    blog = "blog"


# Build enums dynamically from Literal types to avoid duplication
ResourceGroupRole = enum.Enum("ResourceGroupRole", {r: r for r in get_args(ResourceGroupRole_T)}, type=str)  # type: ignore[misc]
AutoJoinScope = enum.Enum("AutoJoinScope", {s: s for s in get_args(ResourceGroupAutoJoinScope_T)}, type=str)  # type: ignore[misc]
_SPEND_LIMIT_KEYS = get_args(ResourceGroupSpendLimit_T)
_GROUP_ID_PATTERN = re.compile(r"[0-9a-f]{24}")

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

GroupArg = Annotated[str, Argument(help="Name or 24-character hexadecimal ID of the resource group.")]
OrgOpt = Annotated[str, Option("--org", help="Name of the organization.")]

resource_groups_cli = typer_factory(help="Manage resource groups of an organization on the Hub.")


@resource_groups_cli.command(
    "assign",
    examples=[
        "hf resource-groups assign 66980ecfc1e12a49c8f0e42d my-org/my-model",
        "hf resource-groups assign research my-org/my-model",
        "hf resource-groups assign 66980ecfc1e12a49c8f0e42d my-org/data-1 my-org/data-2 --type dataset",
        "hf resource-groups assign 66980ecfc1e12a49c8f0e42d my-org/my-collection-64f9a55bb3115b4f513ec026 --type collection",
    ],
)
def assign(
    group: Annotated[
        str,
        Argument(
            help="Name or 24-character hexadecimal ID of the resource group. A name is looked up in the organization"
            " that owns the items."
        ),
    ],
    items: ItemsArg,
    item_type: ItemTypeOpt = ResourceGroupItemType.model,
    token: TokenOpt = None,
) -> None:
    """Assign items to a resource group. An item already in another group is moved to this one."""
    api = get_hf_api(token=token)
    if not _GROUP_ID_PATTERN.fullmatch(group):
        namespaces = {item.split("/")[0] for item in items}
        if len(namespaces) > 1:
            raise CLIError(
                f"Cannot resolve resource group '{group}': the items belong to several namespaces"
                f" ({', '.join(sorted(namespaces))}). Pass the group ID instead."
            )
        group = _resolve_group_id(api, namespaces.pop(), group)
    _update_items(api, items, group, item_type, message="Assigned to resource group")


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
    _update_items(get_hf_api(token=token), items, None, item_type, message="Removed from resource group")


@resource_groups_cli.command(
    "list | ls",
    examples=["hf resource-groups ls --org my-org", "hf resource-groups ls --org my-org -q"],
)
def list_cmd(organization: OrgOpt, token: TokenOpt = None) -> None:
    """List the resource groups of an organization.

    Only the groups you administer are listed, or all of them if you are an admin of the organization.
    """
    api = get_hf_api(token=token)
    out.table(
        [
            {
                "id": group.id,
                "name": group.name,
                "description": group.description,
                "users": len(group.users),
                "resources": len(group.resources),
            }
            for group in api.list_resource_groups(organization)
        ]
    )


@resource_groups_cli.command(
    "info",
    examples=[
        "hf resource-groups info research --org my-org",
        "hf resource-groups info 66980ecfc1e12a49c8f0e42d --org my-org",
    ],
)
def info(group: GroupArg, organization: OrgOpt, token: TokenOpt = None) -> None:
    """Show a resource group.

    Prints its description, auto-join setting, spend limits, members and resources.
    Requires being an admin of the group or of the organization.
    """
    api = get_hf_api(token=token)
    out.dict(api.get_resource_group(organization, _resolve_group_id(api, organization, group)))


@resource_groups_cli.command(
    "create",
    examples=[
        "hf resource-groups create research --org my-org",
        'hf resource-groups create research --org my-org --description "Research team" --user alice:admin --user bob:write',
        "hf resource-groups create research --org my-org --repo my-org/my-model --repo datasets/my-org/my-data",
        "hf resource-groups create research --org my-org --auto-join read --auto-join-scope read_plus",
    ],
)
def create(
    name: Annotated[str, Argument(help="Name of the resource group.")],
    organization: OrgOpt,
    description: Annotated[str | None, Option(help="Description of the resource group.")] = None,
    users: Annotated[
        list[str] | None,
        Option(
            "--user",
            help="Member to add, as `username:role` (role: admin, write, contributor, read, no_access or inherit)."
            " Can be used multiple times.",
        ),
    ] = None,
    repos: Annotated[
        list[str] | None,
        Option(
            "--repo",
            help="Repo to assign to the group. Prefix non-model repos by their type (e.g. `datasets/my-org/my-data`,"
            " `spaces/...`, `buckets/...`, `containers/...`). A repo already in another group is moved."
            " Can be used multiple times.",
        ),
    ] = None,
    auto_join: Annotated[
        ResourceGroupRole | None,  # type: ignore[valid-type]
        Option("--auto-join", help="Make organization members join the group automatically, with this role."),
    ] = None,
    auto_join_scope: Annotated[
        AutoJoinScope | None,  # type: ignore[valid-type]
        Option(help="Which members auto-join: all of them, or those with at least the read role (read_plus)."),
    ] = None,
    token: TokenOpt = None,
) -> None:
    """Create a resource group in an organization."""
    if auto_join_scope is not None and auto_join is None:
        raise CLIError("--auto-join-scope requires --auto-join.")
    api = get_hf_api(token=token)
    group = api.create_resource_group(
        organization,
        name,
        description=description,
        users=_parse_users(users),
        repos=repos,
        auto_join_role=auto_join.value if auto_join is not None else None,
        auto_join_scope=auto_join_scope.value if auto_join_scope is not None else None,
    )
    out.result("Resource group created", id=group.id, name=group.name, organization=organization)


@resource_groups_cli.command(
    "update",
    examples=[
        'hf resource-groups update research --org my-org --name ml-research --description "ML research team"',
        "hf resource-groups update research --org my-org --spend-limit total=50000 --spend-limit jobs=10000",
        "hf resource-groups update research --org my-org --no-spend-limit jobs",
    ],
)
def update(
    group: GroupArg,
    organization: OrgOpt,
    name: Annotated[str | None, Option(help="New name of the resource group.")] = None,
    description: Annotated[str | None, Option(help="New description of the resource group.")] = None,
    spend_limits: Annotated[
        list[str] | None,
        Option(
            "--spend-limit",
            help=f"Spend limit to set, in cents, as `key=value` (key: {', '.join(_SPEND_LIMIT_KEYS)})."
            " Can be used multiple times.",
        ),
    ] = None,
    no_spend_limits: Annotated[
        list[str] | None,
        Option("--no-spend-limit", help="Spend limit to remove (same keys). Can be used multiple times."),
    ] = None,
    token: TokenOpt = None,
) -> None:
    """Update the name, description or spend limits of a resource group."""
    limits = _parse_spend_limits(spend_limits, no_spend_limits)
    if name is None and description is None and not limits:
        raise CLIError("Nothing to update: pass --name, --description, --spend-limit or --no-spend-limit.")
    api = get_hf_api(token=token)
    updated = api.update_resource_group(
        organization,
        _resolve_group_id(api, organization, group),
        name=name,
        description=description,
        spend_limits=limits or None,
    )
    out.result("Resource group updated", id=updated.id, name=updated.name, organization=organization)


@resource_groups_cli.command(
    "delete",
    examples=[
        "hf resource-groups delete research --org my-org",
        "hf resource-groups delete research --org my-org --yes",
    ],
)
def delete(
    group: GroupArg,
    organization: OrgOpt,
    yes: Annotated[bool, Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
    token: TokenOpt = None,
) -> None:
    """Delete a resource group. Its items stay in the organization, without a group."""
    api = get_hf_api(token=token)
    group_id = _resolve_group_id(api, organization, group)
    out.confirm(f"Delete resource group '{group}' ({group_id}) of '{organization}'?", yes=yes)
    api.delete_resource_group(organization, group_id)
    out.result("Resource group deleted", id=group_id, organization=organization)


def _update_items(
    api: "HfApi",
    items: list[str],
    resource_group_id: str | None,
    item_type: ResourceGroupItemType,
    message: str,
) -> None:
    """Update each item with one API call. Keeps going on failure and exits with code 1 if any item failed."""
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


def _resolve_group_id(api: "HfApi", organization: str, group: str) -> str:
    """Return the ID of a resource group given by name or ID (a 24-character hex string is taken as an ID)."""
    if _GROUP_ID_PATTERN.fullmatch(group):
        return group
    matches = [g.id for g in api.list_resource_groups(organization) if g.name == group]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise CLIError(
            f"Several resource groups are named '{group}' in '{organization}' ({', '.join(matches)}). Pass the ID instead."
        )
    raise CLIError(
        f"No resource group named '{group}' in '{organization}'. Run `hf resource-groups ls --org {organization}` to"
        " list the groups you can see (only the groups you administer, unless you are an organization admin)."
    )


def _parse_users(values: list[str] | None) -> dict[str, ResourceGroupRole_T] | None:
    if not values:
        return None
    roles = get_args(ResourceGroupRole_T)
    users: dict[str, ResourceGroupRole_T] = {}
    for value in values:
        user, _, role = value.rpartition(":")
        if not user or role not in roles:
            raise click.BadParameter(f"Expected `username:role` with role in {', '.join(roles)}, got '{value}'.")
        users[user] = cast(ResourceGroupRole_T, role)
    return users


def _parse_spend_limits(
    values: list[str] | None, removed: list[str] | None
) -> dict[ResourceGroupSpendLimit_T, int | None]:
    limits: dict[ResourceGroupSpendLimit_T, int | None] = {}
    for value in values or []:
        key, _, amount = value.partition("=")
        if key not in _SPEND_LIMIT_KEYS or not amount.isdigit():
            raise click.BadParameter(
                f"Expected `key=cents` with key in {', '.join(_SPEND_LIMIT_KEYS)}, got '{value}'."
            )
        limits[cast(ResourceGroupSpendLimit_T, key)] = int(amount)
    for key in removed or []:
        if key not in _SPEND_LIMIT_KEYS:
            raise click.BadParameter(f"Invalid spend limit '{key}', expected one of {', '.join(_SPEND_LIMIT_KEYS)}.")
        if key in limits:
            raise click.BadParameter(f"Spend limit '{key}' is both set and removed.")
        limits[cast(ResourceGroupSpendLimit_T, key)] = None
    return limits
