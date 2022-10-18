"""Miscellaneous / general utilities"""
from typing import Any, Iterable, Iterator, List


def chunk_iterable(
    iterable: Iterable[Any],
    chunk_size: int,
) -> Iterator[List[Any]]:
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

    <Tip warning={true}>
        The last chunk can be smaller than `chunk_size`.
    </Tip>
    """

    def _chunk_iter(itr: Iterable[Any]) -> Iterator[List[Any]]:
        while True:
            chunk = [x for _, x in zip(range(chunk_size), itr)]
            if not chunk:
                break
            yield chunk

    if not chunk_size > 0:
        raise ValueError("chunk_size must be a strictly positive (>0) integer")
    return _chunk_iter(iter(iterable))
