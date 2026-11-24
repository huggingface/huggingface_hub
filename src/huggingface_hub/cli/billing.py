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
"""Contains commands to manage billing (credits) on the Hugging Face Hub."""

import time
from typing import Annotated

from huggingface_hub.errors import CLIError

from ._cli_utils import TokenOpt, get_hf_api, typer_factory
from ._framework import Argument, Option
from ._output import OutputFormat, out


# Polling settings for `hf billing credits purchase-link --wait`.
_WAIT_POLL_INTERVAL = 10  # seconds
_WAIT_TIMEOUT = 15 * 60  # seconds

billing_cli = typer_factory(help="Manage billing on the Hub.")
credits_cli = typer_factory(help="Check credits balance and purchase credits.")
billing_cli.add_group(credits_cli, name="credits")


NamespaceOpt = Annotated[
    str | None,
    Option(help="The user or organization to use. Defaults to the authenticated user."),
]


@credits_cli.command(
    "info",
    examples=[
        "hf billing credits info",
        "hf billing credits info --namespace my-org",
        "hf billing credits info --format json",
    ],
)
def credits_info(namespace: NamespaceOpt = None, token: TokenOpt = None) -> None:
    """Show the credits balance of a user or an organization.

    With `--quiet`, only the total balance (in USD) is printed.
    """
    credits = get_hf_api(token=token).get_credits(namespace=namespace)
    if out.mode == OutputFormat.json:
        out.dict(credits)
        return
    out.result("Credits", balance_usd=credits.balance_usd, billing_mode=credits.billing_mode)
    if out.is_quiet():
        return
    out.table(
        [
            {
                "type": balance.type,
                "name": balance.name,
                "balance_usd": balance.balance_usd,
                "expires_at": balance.expires_at,
            }
            for balance in credits.balances
        ]
    )
    namespace_opt = f" --namespace {namespace}" if namespace else ""
    out.hint(f"Use `hf billing credits purchase-link 50{namespace_opt}` to ask for a $50 credits purchase.")


@credits_cli.command(
    "purchase-link",
    examples=[
        "hf billing credits purchase-link 50",
        "hf billing credits purchase-link 50 --namespace my-org",
        "hf billing credits purchase-link 50 --wait",
    ],
)
def credits_purchase_link(
    amount_usd: Annotated[int, Argument(help="Amount of credits to purchase, in USD.")],
    namespace: NamespaceOpt = None,
    wait: Annotated[
        bool,
        Option(
            "--wait",
            help=(
                "Block until the credits balance increases (checked every 10s, for up to 15 minutes)."
                " Exits with a non-zero code on timeout."
            ),
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Create a link to purchase credits. Nothing is charged until the purchase is confirmed in a browser.

    The link is also sent by email to the token owner. Useful for agents that need a human to add credits.
    With `--quiet`, only the URL is printed.
    """
    api = get_hf_api(token=token)
    # Read the balance before creating the link so that a fast purchase isn't missed.
    initial_balance = api.get_credits(namespace=namespace).balance_usd if wait else 0.0
    link = api.create_credits_purchase_link(amount_usd, namespace=namespace)
    out.result("Purchase link created. Nothing was charged and the link was also sent by email.", url=link.url)

    namespace_opt = f" --namespace {namespace}" if namespace else ""
    if not wait:
        out.hint(
            "Open the link in a browser to confirm the purchase."
            f" Run `hf billing credits info{namespace_opt}` to check the balance once confirmed."
        )
        return

    # Any increase means credits were added (usage only lowers the balance): this also catches credits added
    # from the billing page, a coupon or auto-recharge instead of this link.
    status = out.status("Waiting for the purchase to be confirmed in a browser...")
    deadline = time.monotonic() + _WAIT_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(_WAIT_POLL_INTERVAL)
        balance = api.get_credits(namespace=namespace).balance_usd
        if balance > initial_balance:
            status.done("Purchase confirmed.")
            out.result("Credits added", balance_usd=balance)
            return
    status.done("Timed out.")
    raise CLIError(
        f"No credits added after {_WAIT_TIMEOUT // 60} minutes. Run `hf billing credits info{namespace_opt}`"
        " to check the balance once the purchase is confirmed."
    )
