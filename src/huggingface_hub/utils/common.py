from multiprocessing.sharedctypes import Value
from typing import Any, Iterable, Iterator, List


def chunk_iterable(
    iterable: Iterable[Any],
    chunk_size: int,
) -> Iterator[List[Any]]:
    """
    Returns an iterator over iterable in chunks of size chunk_size.

    ```python
    >>> iterable = range(128)
    >>> chunked_iterable = chunk_iterable(iterable, chunk_size=8)
    >>> next(chunked_iterable)
    # [0, 1, 2, 3, 4, 5, 6, 7]
    >>> next(chunked_iterable)
    # [8, 9, 10, 11, 12, 13, 14, 15]
    ```
    """

    def _chunk_iter(iter: Iterable[Any]) -> Iterator[List[Any]]:
        chunk = []
        for elem in iterable:
            chunk.append(elem)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if len(chunk):
            yield chunk

    if not chunk_size > 0:
        raise ValueError("chunk_size must be a strictly positive (>0) integer")
    return _chunk_iter(iterable)
