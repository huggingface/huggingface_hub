#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License
"""Utility helpers to handle progress bars in `huggingface_hub`.

Usage:
    1. Use `huggingface_hub.utils.tqdm` as you would use `tqdm.tqdm` or `tqdm.auto.tqdm`.
    2. To disable progress bars, either use `disable_progress_bars()` helper or set the
       environement variable `HF_HUB_DISABLE_PROGRESS_BARS` to 1.
    3. To re-enable progress bars, use `enable_progress_bars()`.
    4. To check weither progress bars are disabled, use `are_progress_bars_disabled()`.

NOTE: Environment variable `HF_HUB_DISABLE_PROGRESS_BARS` has the priority.

Example:
    ```py
    from huggingface_hub.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
        tqdm,
    )

    # Disable progress bars globally
    disable_progress_bars()

    # Use as normal `tqdm`
    for _ in tqdm(range(5)):
       do_something()

    # Still not showing progress bars, as `disable=False` is overwritten to `True`.
    for _ in tqdm(range(5), disable=False):
       do_something()

    are_progress_bars_disabled() # True

    # Re-enable progress bars globally
    enable_progress_bars()

    # Progress bar will be shown !
    for _ in tqdm(range(5)):
       do_something()
    ```
"""
import warnings

from tqdm.auto import tqdm as old_tqdm

from ..constants import HF_HUB_DISABLE_PROGRESS_BARS


# `HF_HUB_DISABLE_PROGRESS_BARS` is `Optional[bool]` while `_hf_hub_progress_bars_disabled`
# is a `bool`. If `HF_HUB_DISABLE_PROGRESS_BARS` is set to True or False, it has priority.
# If `HF_HUB_DISABLE_PROGRESS_BARS` is None, it means the user have not set the
# environment variable and is free to enable/disable progress bars programmatically.
# TL;DR: env variable has priority over code.
#
# By default, progress bars are enabled.
_hf_hub_progress_bars_disabled: bool = HF_HUB_DISABLE_PROGRESS_BARS or False


def disable_progress_bars() -> None:
    """
    Disable globally progress bars used in `huggingface_hub` except if
    `HF_HUB_DISABLE_PROGRESS_BARS` environement variable has been set.
    """
    if HF_HUB_DISABLE_PROGRESS_BARS is False:
        warnings.warn(
            "Cannot disable progress bars: environment variable"
            " `HF_HUB_DISABLE_PROGRESS_BARS=0` is set and has priority."
        )
        return
    global _hf_hub_progress_bars_disabled
    _hf_hub_progress_bars_disabled = True


def enable_progress_bars() -> None:
    """
    Enable globally progress bars used in `huggingface_hub` except if
    `HF_HUB_DISABLE_PROGRESS_BARS` environement variable has been set.
    """
    if HF_HUB_DISABLE_PROGRESS_BARS is True:
        warnings.warn(
            "Cannot enable progress bars: environment variable"
            " `HF_HUB_DISABLE_PROGRESS_BARS=1` is set and has priority."
        )
        return
    global _hf_hub_progress_bars_disabled
    _hf_hub_progress_bars_disabled = False


def are_progress_bars_disabled() -> bool:
    """Return weither progress bars are globally disabled or not."""
    global _hf_hub_progress_bars_disabled
    return _hf_hub_progress_bars_disabled


class tqdm(old_tqdm):
    """
    Class to override `disable` argument in case progress bars are globally disabled.

    Taken from https://github.com/tqdm/tqdm/issues/619#issuecomment-619639324.
    """

    def __init__(self, *args, **kwargs):
        if are_progress_bars_disabled():
            kwargs["disable"] = True
        super().__init__(*args, **kwargs)
