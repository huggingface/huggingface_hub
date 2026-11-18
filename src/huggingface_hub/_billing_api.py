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
"""Data structures used by billing-related API methods."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .utils import parse_datetime


CreditBalanceType = Literal["included", "purchased", "granted"]
BillingMode = Literal["prepaid", "postpaid"]


@dataclass
class CreditBalance:
    """A credit balance from one source.

    Args:
        type (`str`): Source of the credits: `included`, `purchased` or `granted`.
        balance_usd (`float`): Amount remaining in USD.
        name (`str` or `None`): Name of the balance, if provided by the Hub.
        expires_at (`datetime` or `None`): Expiration date, if the balance expires.
    """

    type: CreditBalanceType
    balance_usd: float
    name: str | None = None
    expires_at: datetime | None = None

    def __init__(self, data: dict) -> None:
        self.type = data["type"]
        self.balance_usd = data["balanceUsd"]
        self.name = data.get("name")
        self.expires_at = parse_datetime(expires_at) if (expires_at := data.get("expiresAt")) is not None else None


@dataclass
class CreditsInfo:
    """Credit balance and billing mode for a user or organization.

    Args:
        balance_usd (`float`): Total credit balance in USD.
        billing_mode (`str`): Billing mode, either `prepaid` or `postpaid`.
        balances (`list` of [`CreditBalance`]): Individual balances by source.
    """

    balance_usd: float
    billing_mode: BillingMode
    balances: list[CreditBalance]

    def __init__(self, data: dict) -> None:
        self.balance_usd = data["balanceUsd"]
        self.billing_mode = data["billingMode"]
        self.balances = [CreditBalance(balance) for balance in data["balances"]]


@dataclass
class CreditsPurchaseLink:
    """A link where the user can confirm a credits purchase.

    Creating the link does not charge the account. The link is also sent by email to the token owner.

    Args:
        url (`str`): Link to confirm the purchase.
    """

    url: str
