"""Utilities to efficiently compute the SHA 256 hash of a bunch of bytes"""

from functools import partial
from hashlib import sha256
from typing import BinaryIO, Iterable, Optional


def iter_fileobj(
    fileobj: BinaryIO, chunk_size: Optional[int] = None
) -> Iterable[bytes]:
    """Returns an iterator over the content of ``fileobj`` in chunks of ``chunk_size``"""
    chunk_size = chunk_size or -1
    return iter(partial(fileobj.read, chunk_size), b"")


def sha_iter(iterable: Iterable[bytes]):
    sha = sha256()
    for chunk in iterable:
        sha.update(chunk)
    return sha.digest()


def sha_fileobj(fileobj: BinaryIO, chunk_size: Optional[int] = None) -> bytes:
    """
    Computes the sha256 hash of the given file object, by chunks of size `chunk_size`.

    Args:
        fileobj (file-like object):
            The File object to compute sha256 for, typically obtained with `open(path, "rb")`
        chunk_size (`int`, *optional*):
            The number of bytes to read from `fileobj` at once, defaults to 512

    Returns:
        `bytes`: `fileobj`'s sha256 hash as bytes
    """
    chunk_size = chunk_size if chunk_size is not None else 512
    return sha_iter(iter_fileobj(fileobj))
