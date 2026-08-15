# Copyright 2022-present, the HuggingFace Inc. team.
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
"""Contains a utility to iterate by chunks over an iterator."""

import itertools
from collections.abc import Iterable
from typing import TypeVar


T = TypeVar("T")


def chunk_iterable(iterable: Iterable[T], chunk_size: int) -> Iterable[list[T]]:
    """Iterates over an iterator chunk by chunk.

    Args:
        iterable (`Iterable`):
            The iterable on which we want to iterate.
        chunk_size (`int`):
            Size of the chunks. Must be a strictly positive integer (e.g. >0).

    Example:

    ```python
    >>> from huggingface_hub.utils import chunk_iterable

    >>> for items in chunk_iterable(range(17), chunk_size=8):
    ...     print(items)
    # [0, 1, 2, 3, 4, 5, 6, 7]
    # [8, 9, 10, 11, 12, 13, 14, 15]
    # [16] # smaller last chunk
    ```

    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If `chunk_size` <= 0.

    > [!WARNING]
    > The last chunk can be smaller than `chunk_size`.
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("`chunk_size` must be a strictly positive integer (>0).")

    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk
