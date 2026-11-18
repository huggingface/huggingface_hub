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
"""Commands for billing and prepaid credits."""

from typing import Annotated

from ._cli_utils import TokenOpt, get_hf_api, typer_factory
from ._framework import Argument, Option
from ._output import OutputFormat, _dataclass_to_dict, out


billing_cli = typer_factory(help="Manage billing and prepaid credits.")
credits_cli = typer_factory(help="View and manage prepaid credits.")
credits_cli.name = "credits"
billing_cli.add_command(credits_cli, name="credits")

NamespaceOpt = Annotated[
    str | None,
    Option(help="Organization namespace. Defaults to your personal account."),
]


@credits_cli.command("info", examples=["hf billing credits info", "hf billing credits info --format json"])
def credits_info(namespace: NamespaceOpt = None, token: TokenOpt = None) -> None:
    """Show the credit balance and billing mode."""
    credits = get_hf_api(token=token).get_credits(namespace=namespace)
    if out.mode == OutputFormat.human:
        out.result("Credits balance", balance_usd=credits.balance_usd, billing_mode=credits.billing_mode)
        out.table([_dataclass_to_dict(balance) for balance in credits.balances])
    elif out.mode == OutputFormat.quiet:
        out.result("Credits balance", balance_usd=credits.balance_usd)
    else:
        out.dict(_dataclass_to_dict(credits))
    out.hint("Create a purchase link with `hf billing credits purchase-link 50` to add credits.")


@credits_cli.command(
    "purchase-link",
    examples=[
        "hf billing credits purchase-link 50",
        "hf billing credits purchase-link 50 --namespace my-org",
    ],
)
def credits_purchase_link(
    amount_usd: Annotated[int, Argument(help="Amount in USD to purchase.")],
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Create a link to confirm a prepaid credits purchase."""
    link = get_hf_api(token=token).create_credits_purchase_link(amount_usd, namespace=namespace)
    out.result("Credits purchase link created", url=link.url)
    out.hint(
        "Nothing was charged. The link was also sent by email; open it to confirm the purchase, then run "
        "`hf billing credits info` to check your balance."
    )
