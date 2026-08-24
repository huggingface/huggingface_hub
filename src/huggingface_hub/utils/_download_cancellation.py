"""Cooperative cancellation for the downloads owned by one high-level operation.

Helpers like [`snapshot_download`] fan files out over a `ThreadPoolExecutor`. A `KeyboardInterrupt` is
delivered to the main thread only, so worker threads never observe it: they keep running, and the
executor's `shutdown(wait=True)` blocks until every one of them finishes. Ctrl+C then looks like a hang.

This module lets the main thread stop those workers from the outside. The controller is published on a
`ContextVar` that each worker enters, so a cancellation only ever reaches the downloads of the operation
that created it — a parallel download running in another thread is left alone.

Workers cooperate in two ways:
- Xet downloads register their `hf_xet` download group, which `cancel()` aborts (see `xet_get`).
- HTTP downloads poll [`DownloadCancellation.is_cancelled`] between chunks (see `http_get`).
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import Protocol

from ..errors import DownloadCancelledError


class _AbortableDownloadGroup(Protocol):
    def abort(self) -> None: ...


class DownloadCancellation:
    """Cancel only the downloads owned by one high-level operation.

    A group must stay registered for as long as a worker can be blocked inside it. `hf_xet`'s
    `wait_to_finish()` (called by the group's `__exit__`) only polls for `KeyboardInterrupt` on the main
    thread, so a worker blocked there can be woken up by nothing but an external `abort()`. Registration
    is therefore a plain `register`/`unregister` pair rather than a context manager: a `with` block
    nested inside the group's own `with` would unwind first and leave exactly that window untracked.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._groups: dict[int, _AbortableDownloadGroup] = {}

    @property
    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested. Polled by loops that can stop between iterations."""
        with self._lock:
            return self._cancelled

    def raise_if_cancelled(self) -> None:
        """Reject work that starts, or resumes, after cancellation was requested."""
        with self._lock:
            if self._cancelled:
                raise DownloadCancelledError("Download was cancelled")

    def register(self, group: _AbortableDownloadGroup) -> None:
        """Track a group unless cancellation has already been requested.

        Raises [`DownloadCancelledError`] if it has, so the caller leaves the group's `with` block
        through the exception path: `hf_xet` then aborts the group instead of waiting on it.
        """
        with self._lock:
            if self._cancelled:
                raise DownloadCancelledError("Download was cancelled")
            self._groups[id(group)] = group

    def unregister(self, group: _AbortableDownloadGroup) -> None:
        """Stop tracking a group that can no longer block its worker."""
        with self._lock:
            self._groups.pop(id(group), None)

    def cancel(self) -> None:
        """Prevent new downloads from starting and abort all currently tracked groups."""
        with self._lock:
            self._cancelled = True
            groups = list(self._groups.values())

        for group in groups:
            # `abort()` crosses into Rust: a PyO3 panic surfaces as `pyo3_runtime.PanicException`, which
            # derives from `BaseException`. Suppressing only `Exception` would let one bad group skip the
            # abort of every group after it - and those workers would stay blocked. A group that finished
            # concurrently is also fair game here; aborting it is a no-op.
            with suppress(BaseException):
                group.abort()


_DOWNLOAD_CANCELLATION = ContextVar[DownloadCancellation | None]("huggingface_hub_download_cancellation", default=None)


@contextmanager
def download_cancellation_scope(cancellation: DownloadCancellation) -> Iterator[None]:
    """Make a cancellation controller visible to the downloads in the current execution context.

    `ContextVar`s are not inherited by `ThreadPoolExecutor` workers, so this must be entered *inside* the
    mapped function rather than around `hf_thread_map`.
    """
    token = _DOWNLOAD_CANCELLATION.set(cancellation)
    try:
        yield
    finally:
        _DOWNLOAD_CANCELLATION.reset(token)


def get_download_cancellation() -> DownloadCancellation | None:
    """Return the cancellation controller for the current high-level operation, if any."""
    return _DOWNLOAD_CANCELLATION.get()
