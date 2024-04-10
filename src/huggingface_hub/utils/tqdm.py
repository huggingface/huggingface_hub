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

Example:
    1. Use `huggingface_hub.utils.tqdm` as you would use `tqdm.tqdm` or `tqdm.auto.tqdm`.
    2. To disable progress bars, either use `disable_progress_bars()` helper or set the
       environment variable `HF_HUB_DISABLE_PROGRESS_BARS` to 1.
    3. To re-enable progress bars, use `enable_progress_bars()`.
    4. To check whether progress bars are disabled, use `are_progress_bars_disabled()`.

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

import io
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

from tqdm.auto import tqdm as old_tqdm

from ..constants import HF_HUB_DISABLE_PROGRESS_BARS


# The `HF_HUB_DISABLE_PROGRESS_BARS` environment variable can be True, False, or not set (None),
# allowing for control over progress bar visibility. When set, this variable takes precedence
# over programmatic settings, dictating whether progress bars should be shown or hidden globally.
# Essentially, the environment variable's setting overrides any code-based configurations.
#
# If `HF_HUB_DISABLE_PROGRESS_BARS` is not defined (None), it implies that users can manage
# progress bar visibility through code. By default, progress bars are turned on.


progress_bar_states = {}


def _set_progress_bar_state(name: Optional[str], enabled: bool) -> None:
    """
    Set the state of a progress bar group, creating the hierarchy as needed.

    Args:
        name (`str`, *optional*):
            The name of the progress bar group.
        enabled (`bool`):
            Whether the progress bar group should be enabled or not.


    """
    if name is None:
        name = ""
    parts = name.split(".")
    current_level = progress_bar_states

    for part in parts[:-1]:
        if part not in current_level:
            current_level[part] = {}
        current_level = current_level[part]

    current_level[parts[-1]] = enabled


def _get_progress_bar_state(name: Optional[str]) -> bool:
    """
    Get the state of a progress bar group.

    Args:
        name (`str`, *optional*):
            The name of the progress bar group.

    Returns:
        `bool`: Whether the progress bar group is enabled or not.
    """
    if name is None:
        name = ""
    parts = name.split(".")
    current_level = progress_bar_states

    for part in parts:
        if part in current_level:
            if isinstance(current_level[part], dict):
                current_level = current_level[part]
            else:
                return current_level[part]
        else:
            break

    return True


def disable_progress_bars(name: Optional[str] = None) -> None:
    """
    Disable progress bars globally or for a specific group in `huggingface_hub`, except when overridden by
    the `HF_HUB_DISABLE_PROGRESS_BARS` environment variable.

    Args:
        name (`str`, *optional*):
            Specific group to disable progress bars for. If None, disables globally.
    """
    if HF_HUB_DISABLE_PROGRESS_BARS is False:
        warnings.warn(
            "Cannot disable progress bars: environment variable `HF_HUB_DISABLE_PROGRESS_BARS=0` is set and has priority."
        )
        return

    _set_progress_bar_state(name, False)


def enable_progress_bars(name: Optional[str] = None) -> None:
    """
    Enable progress bars globally or for a specific group in `huggingface_hub`, except when overridden by
    the `HF_HUB_DISABLE_PROGRESS_BARS` environment variable.

    Args:
        name (Optional[str]):
            Specific group to enable progress bars for. If None, enables globally.
    """
    if HF_HUB_DISABLE_PROGRESS_BARS is True:
        warnings.warn(
            "Cannot enable progress bars: environment variable `HF_HUB_DISABLE_PROGRESS_BARS=1` is set and has priority."
        )
        return

    _set_progress_bar_state(name, True)


def are_progress_bars_disabled(name: Optional[str] = None) -> bool:
    """
    Check if progress bars are disabled globally or for a specific group.

    This respects the `HF_HUB_DISABLE_PROGRESS_BARS` environment variable first, then checks
    programmatic settings. If a group name is provided, it checks for that specific group's
    setting, unless overridden by the environment variable.

    Args:
        name (`str`, *optional*): Group name to check; if None, checks the global setting.

    Returns:
        `bool`: True if disabled, False otherwise.
    """
    if HF_HUB_DISABLE_PROGRESS_BARS is True:
        return True
    return not _get_progress_bar_state(name)


class tqdm(old_tqdm):
    """
    Class to override `disable` argument in case progress bars are globally disabled.

    Taken from https://github.com/tqdm/tqdm/issues/619#issuecomment-619639324.
    """

    def __init__(self, *args, **kwargs):
        if are_progress_bars_disabled():
            kwargs["disable"] = True
        super().__init__(*args, **kwargs)

    def __delattr__(self, attr: str) -> None:
        """Fix for https://github.com/huggingface/huggingface_hub/issues/1603"""
        try:
            super().__delattr__(attr)
        except AttributeError:
            if attr != "_lock":
                raise


@contextmanager
def tqdm_stream_file(path: Union[Path, str]) -> Iterator[io.BufferedReader]:
    """
    Open a file as binary and wrap the `read` method to display a progress bar when it's streamed.

    First implemented in `transformers` in 2019 but removed when switched to git-lfs. Used in `huggingface_hub` to show
    progress bar when uploading an LFS file to the Hub. See github.com/huggingface/transformers/pull/2078#discussion_r354739608
    for implementation details.

    Note: currently implementation handles only files stored on disk as it is the most common use case. Could be
          extended to stream any `BinaryIO` object but we might have to debug some corner cases.

    Example:
    ```py
    >>> with tqdm_stream_file("config.json") as f:
    >>>     requests.put(url, data=f)
    config.json: 100%|█████████████████████████| 8.19k/8.19k [00:02<00:00, 3.72kB/s]
    ```
    """
    if isinstance(path, str):
        path = Path(path)

    with path.open("rb") as f:
        total_size = path.stat().st_size
        pbar = tqdm(
            unit="B",
            unit_scale=True,
            total=total_size,
            initial=0,
            desc=path.name,
        )

        f_read = f.read

        def _inner_read(size: Optional[int] = -1) -> bytes:
            data = f_read(size)
            pbar.update(len(data))
            return data

        f.read = _inner_read  # type: ignore

        yield f

        pbar.close()
