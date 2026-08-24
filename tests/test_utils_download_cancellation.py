import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from huggingface_hub.errors import DownloadCancelledError
from huggingface_hub.utils._download_cancellation import (
    DownloadCancellation,
    download_cancellation_scope,
    get_download_cancellation,
)


def test_aborts_only_its_own_groups():
    first_cancellation = DownloadCancellation()
    second_cancellation = DownloadCancellation()
    first_group = MagicMock()
    unrelated_group = MagicMock()

    first_cancellation.register(first_group)
    second_cancellation.register(unrelated_group)
    first_cancellation.cancel()

    first_group.abort.assert_called_once_with()
    unrelated_group.abort.assert_not_called()


def test_rejects_late_groups():
    cancellation = DownloadCancellation()
    cancellation.cancel()

    with pytest.raises(DownloadCancelledError, match="Download was cancelled"):
        cancellation.register(MagicMock())


def test_ignores_unregistered_groups():
    cancellation = DownloadCancellation()
    group = MagicMock()

    cancellation.register(group)
    cancellation.unregister(group)
    cancellation.unregister(group)  # unregistering twice must not raise
    cancellation.cancel()

    group.abort.assert_not_called()


def test_keeps_aborting_after_a_group_raises():
    """`abort()` reaches Rust: a PyO3 panic is a `BaseException`, not an `Exception`."""
    cancellation = DownloadCancellation()
    panicking_group = MagicMock()
    panicking_group.abort.side_effect = BaseException("panic from Rust code")
    healthy_group = MagicMock()

    cancellation.register(panicking_group)
    cancellation.register(healthy_group)
    cancellation.cancel()

    healthy_group.abort.assert_called_once_with()


def test_is_cancelled_flips_once_cancelled():
    cancellation = DownloadCancellation()
    assert not cancellation.is_cancelled

    cancellation.raise_if_cancelled()  # no-op while running
    cancellation.cancel()

    assert cancellation.is_cancelled
    with pytest.raises(DownloadCancelledError, match="Download was cancelled"):
        cancellation.raise_if_cancelled()


def test_scope_is_restored_and_isolated_per_thread():
    cancellation = DownloadCancellation()
    assert get_download_cancellation() is None

    with download_cancellation_scope(cancellation):
        assert get_download_cancellation() is cancellation

        # `ContextVar`s are not inherited by pool workers: the scope must be entered inside the mapped
        # function, which is why `snapshot_download` wraps the worker rather than `hf_thread_map`.
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(get_download_cancellation).result() is None

    assert get_download_cancellation() is None


def test_scopes_nest():
    outer = DownloadCancellation()
    inner = DownloadCancellation()

    with download_cancellation_scope(outer):
        with download_cancellation_scope(inner):
            assert get_download_cancellation() is inner
        assert get_download_cancellation() is outer


def test_cancel_is_idempotent_and_thread_safe():
    cancellation = DownloadCancellation()
    group = MagicMock()
    cancellation.register(group)

    barrier = threading.Barrier(4)

    def cancel_concurrently():
        barrier.wait(timeout=5)
        cancellation.cancel()

    threads = [threading.Thread(target=cancel_concurrently) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    # Each `cancel()` aborts the groups it snapshots; what matters is that none of them raised and the
    # controller stayed consistent.
    assert cancellation.is_cancelled
    assert group.abort.call_count >= 1
