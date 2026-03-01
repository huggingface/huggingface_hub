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
"""Contains commands to manage webhooks on the Hugging Face Hub.

Usage:
    # list all webhooks
    hf webhooks ls

    # show details of a single webhook
    hf webhooks info <webhook_id>

    # create a new webhook
    hf webhooks create --url https://example.com/hook --watch model:bert-base-uncased

    # create a webhook watching multiple items and domains
    hf webhooks create --url https://example.com/hook --watch org:HuggingFace --watch model:gpt2 --domain repo

    # update a webhook
    hf webhooks update <webhook_id> --url https://new-url.com/hook

    # enable / disable a webhook
    hf webhooks enable <webhook_id>
    hf webhooks disable <webhook_id>

    # delete a webhook
    hf webhooks delete <webhook_id>
"""

import enum
import json
from typing import Annotated, Optional, get_args, get_type_hints

import typer

from huggingface_hub.constants import WEBHOOK_DOMAIN_T
from huggingface_hub.hf_api import WebhookWatchedItem

from ._cli_utils import (
    FormatOpt,
    OutputFormat,
    QuietOpt,
    TokenOpt,
    api_object_to_dict,
    get_hf_api,
    print_list_output,
    typer_factory,
)


# Build enums dynamically from Literal types to avoid duplication
_WATCHED_TYPES = get_args(get_type_hints(WebhookWatchedItem)["type"])
WatchedItemType = enum.Enum("WatchedItemType", {t: t for t in _WATCHED_TYPES}, type=str)  # type: ignore[misc]

_DOMAIN_TYPES = get_args(WEBHOOK_DOMAIN_T)
WebhookDomain = enum.Enum("WebhookDomain", {d: d for d in _DOMAIN_TYPES}, type=str)  # type: ignore[misc]


def _parse_watch(values: list[str]) -> list[WebhookWatchedItem]:
    """Parse 'type:name' strings into WebhookWatchedItem objects.

    Args:
        values: List of strings in the format 'type:name'
            (e.g., 'model:bert-base-uncased', 'org:HuggingFace').

    Returns:
        List of WebhookWatchedItem objects.

    Raises:
        typer.BadParameter: If any value doesn't match the expected format.
    """
    items = []
    valid_types = tuple(_WATCHED_TYPES)
    for v in values:
        if ":" not in v:
            raise typer.BadParameter(
                f"Expected format 'type:name' (e.g. 'model:bert-base-uncased'), got '{v}'."
                f" Valid types: {', '.join(valid_types)}."
            )
        kind, name = v.split(":", 1)
        if kind not in valid_types:
            raise typer.BadParameter(
                f"Invalid type '{kind}'. Valid types: {', '.join(valid_types)}."
            )
        items.append(WebhookWatchedItem(type=kind, name=name))  # type: ignore[arg-type]
    return items


webhooks_cli = typer_factory(help="Manage webhooks on the Hub.")


@webhooks_cli.command(
    "ls",
    examples=[
        "hf webhooks ls",
        "hf webhooks ls --format json",
        "hf webhooks ls -q",
    ],
)
def webhooks_ls(
    format: FormatOpt = OutputFormat.table,
    quiet: QuietOpt = False,
    token: TokenOpt = None,
) -> None:
    """List all webhooks for the current user."""
    api = get_hf_api(token=token)
    results = [api_object_to_dict(w) for w in api.list_webhooks()]
    print_list_output(
        results,
        format=format,
        quiet=quiet,
        headers=["id", "url", "disabled", "domains", "watched"],
        row_fn=lambda item: [
            item.get("id", ""),
            item.get("url") or "(job)",
            str(item.get("disabled", False)),
            ", ".join(item.get("domains") or []),
            ", ".join(
                f"{w['type']}:{w['name']}" if isinstance(w, dict) else str(w)
                for w in (item.get("watched") or [])
            ),
        ],
    )


@webhooks_cli.command(
    "info",
    examples=[
        "hf webhooks info abc123",
    ],
)
def webhooks_info(
    webhook_id: Annotated[str, typer.Argument(help="The ID of the webhook.")],
    token: TokenOpt = None,
) -> None:
    """Show full details for a single webhook as JSON."""
    api = get_hf_api(token=token)
    webhook = api.get_webhook(webhook_id)
    print(json.dumps(api_object_to_dict(webhook), indent=2))


@webhooks_cli.command(
    "create",
    examples=[
        "hf webhooks create --url https://example.com/hook --watch model:bert-base-uncased",
        "hf webhooks create --url https://example.com/hook --watch org:HuggingFace --watch model:gpt2 --domain repo",
        "hf webhooks create --url https://example.com/hook --watch user:julien-c --secret mysecret",
    ],
)
def webhooks_create(
    url: Annotated[str, typer.Option(help="URL to send webhook payloads to.")],
    watch: Annotated[
        list[str],
        typer.Option(
            "--watch",
            help="Item to watch, in 'type:name' format (e.g. 'model:bert-base-uncased'). Repeatable.",
        ),
    ],
    domain: Annotated[
        Optional[list[WebhookDomain]],
        typer.Option(
            "--domain",
            help="Domain to watch: 'repo' or 'discussions'. Repeatable. Defaults to all domains.",
        ),
    ] = None,
    secret: Annotated[
        Optional[str],
        typer.Option(help="Optional secret used to sign webhook payloads."),
    ] = None,
    token: TokenOpt = None,
) -> None:
    """Create a new webhook."""
    api = get_hf_api(token=token)
    watched_items = _parse_watch(watch)
    domains = [d.value for d in domain] if domain else None
    webhook = api.create_webhook(url=url, watched=watched_items, domains=domains, secret=secret)
    print(f"Webhook created: {webhook.id}")
    print(json.dumps(api_object_to_dict(webhook), indent=2))


@webhooks_cli.command(
    "update",
    examples=[
        "hf webhooks update abc123 --url https://new-url.com/hook",
        "hf webhooks update abc123 --watch model:gpt2 --domain repo",
        "hf webhooks update abc123 --secret newsecret",
    ],
)
def webhooks_update(
    webhook_id: Annotated[str, typer.Argument(help="The ID of the webhook to update.")],
    url: Annotated[
        Optional[str],
        typer.Option(help="New URL to send webhook payloads to."),
    ] = None,
    watch: Annotated[
        Optional[list[str]],
        typer.Option(
            "--watch",
            help=(
                "New list of items to watch, in 'type:name' format. "
                "Repeatable. Replaces the entire existing watched list."
            ),
        ),
    ] = None,
    domain: Annotated[
        Optional[list[WebhookDomain]],
        typer.Option(
            "--domain",
            help="New list of domains to watch: 'repo' or 'discussions'. Repeatable.",
        ),
    ] = None,
    secret: Annotated[
        Optional[str],
        typer.Option(help="New secret used to sign webhook payloads."),
    ] = None,
    token: TokenOpt = None,
) -> None:
    """Update an existing webhook. Only provided options are changed."""
    api = get_hf_api(token=token)
    watched_items = _parse_watch(watch) if watch else None
    domains = [d.value for d in domain] if domain else None
    webhook = api.update_webhook(webhook_id, url=url, watched=watched_items, domains=domains, secret=secret)
    print(f"Webhook updated: {webhook.id}")
    print(json.dumps(api_object_to_dict(webhook), indent=2))


@webhooks_cli.command(
    "enable",
    examples=[
        "hf webhooks enable abc123",
    ],
)
def webhooks_enable(
    webhook_id: Annotated[str, typer.Argument(help="The ID of the webhook to enable.")],
    token: TokenOpt = None,
) -> None:
    """Enable a disabled webhook."""
    api = get_hf_api(token=token)
    webhook = api.enable_webhook(webhook_id)
    print(f"Webhook enabled: {webhook.id}")


@webhooks_cli.command(
    "disable",
    examples=[
        "hf webhooks disable abc123",
    ],
)
def webhooks_disable(
    webhook_id: Annotated[str, typer.Argument(help="The ID of the webhook to disable.")],
    token: TokenOpt = None,
) -> None:
    """Disable an active webhook."""
    api = get_hf_api(token=token)
    webhook = api.disable_webhook(webhook_id)
    print(f"Webhook disabled: {webhook.id}")


@webhooks_cli.command(
    "delete",
    examples=[
        "hf webhooks delete abc123",
    ],
)
def webhooks_delete(
    webhook_id: Annotated[str, typer.Argument(help="The ID of the webhook to delete.")],
    token: TokenOpt = None,
) -> None:
    """Delete a webhook permanently."""
    api = get_hf_api(token=token)
    api.delete_webhook(webhook_id)
    print(f"Webhook deleted: {webhook_id}")
