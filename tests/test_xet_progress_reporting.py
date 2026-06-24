import logging
from types import SimpleNamespace

from huggingface_hub.utils._xet_progress_reporting import (
    XET_TRANSFER_BAR_MIN_TOTAL,
    XetDownloadProgressReporter,
    _finish_transfer_bar,
    _set_monotonic_total,
    _update_transfer_bar,
    make_xet_aggregated_progress_proxy,
)


def _group_report(
    *,
    bytes_completed: int = 0,
    transfer_completed: int = 0,
    total_bytes: int = 100,
) -> SimpleNamespace:
    return SimpleNamespace(
        total_bytes_completed=bytes_completed,
        total_transfer_bytes_completed=transfer_completed,
        total_bytes=total_bytes,
        total_bytes_completion_rate=1_000_000.0,
        total_transfer_bytes_completion_rate=2_000_000.0,
    )


class _RecordingBar:
    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total")
        self.updates: list[int] = []
        self.n = 0
        self.postfixes: list[str] = []
        self.closed = False

    def update(self, n: int) -> None:
        self.updates.append(n)
        self.n += n

    def refresh(self) -> None:
        pass

    def set_postfix_str(self, postfix: str, refresh: bool = False) -> None:
        self.postfixes.append(postfix)

    def close(self) -> None:
        self.closed = True


class TestXetDownloadProgressReporter:
    def test_updates_both_bars_independently(self):
        reconstruction_bar = _RecordingBar(total=100)
        transfer_bar = _RecordingBar(total=100)

        reporter = XetDownloadProgressReporter(
            reconstruction_desc="reconstructing",
            transfer_desc="downloading",
            total=100,
            log_level=logging.CRITICAL,
            external_reconstruction_bar=reconstruction_bar,
        )
        reporter.transfer_bar = transfer_bar
        reporter._owns_transfer_bar = False

        reporter.update_progress(_group_report(bytes_completed=0, transfer_completed=50))
        assert transfer_bar.updates == [50]
        assert reconstruction_bar.updates == []

        reporter.update_progress(_group_report(bytes_completed=30, transfer_completed=80))
        assert reconstruction_bar.updates == [30]
        assert transfer_bar.updates == [50, 30]

    def test_aggregated_bar_routes_transfer_updates(self):
        reconstruction = _RecordingBar(total=0)
        transfer = _RecordingBar(total=0)
        reporter = XetDownloadProgressReporter(
            reconstruction_desc="reconstructing",
            total=100,
            log_level=logging.CRITICAL,
            tqdm_class=make_xet_aggregated_progress_proxy(reconstruction, transfer),
        )

        reporter.update_progress(_group_report(transfer_completed=42))
        assert transfer.updates == [42]
        assert reconstruction.updates == []

    def test_aggregated_reporter_close_does_not_close_parent_bars(self):
        reconstruction = _RecordingBar(total=0)
        transfer = _RecordingBar(total=0)
        reporter = XetDownloadProgressReporter(
            reconstruction_desc="reconstructing",
            total=100,
            log_level=logging.CRITICAL,
            tqdm_class=make_xet_aggregated_progress_proxy(reconstruction, transfer),
        )

        reporter.close()

        assert reconstruction.closed is False
        assert transfer.closed is False

    def test_aggregated_proxy_seeds_both_bar_totals(self):
        reconstruction = _RecordingBar(total=0)
        transfer = _RecordingBar(total=None)
        proxy = make_xet_aggregated_progress_proxy(reconstruction, transfer)(total=100)

        assert reconstruction.total == 100
        assert transfer.total == 100
        proxy.update_transfer(50)
        assert transfer.n == 50
        assert transfer.total == 100

    def test_negative_increments_are_ignored(self):
        reconstruction_bar = _RecordingBar(total=100)
        transfer_bar = _RecordingBar(total=100)

        reporter = XetDownloadProgressReporter(
            reconstruction_desc="reconstructing",
            total=100,
            log_level=logging.CRITICAL,
            external_reconstruction_bar=reconstruction_bar,
        )
        reporter.transfer_bar = transfer_bar
        reporter._owns_transfer_bar = False

        reporter.update_progress(_group_report(bytes_completed=50, transfer_completed=50))
        reporter.update_progress(_group_report(bytes_completed=10, transfer_completed=10))
        assert reconstruction_bar.updates == [50]
        assert transfer_bar.updates == [50]

    def test_aggregated_bar_supports_postfix(self):
        reconstruction = _RecordingBar(total=0)
        transfer = _RecordingBar(total=0)
        reporter = XetDownloadProgressReporter(
            reconstruction_desc="reconstructing",
            total=100,
            log_level=logging.CRITICAL,
            tqdm_class=make_xet_aggregated_progress_proxy(reconstruction, transfer),
        )
        reporter.update_progress(_group_report(bytes_completed=10, transfer_completed=20))
        assert reconstruction.updates == [10]
        assert transfer.updates == [20]
        assert len(reconstruction.postfixes) == 1
        assert len(transfer.postfixes) == 1

    def test_set_monotonic_total_never_decreases(self):
        bar = _RecordingBar(total=100)
        _set_monotonic_total(bar, 80)
        assert bar.total == 100
        _set_monotonic_total(bar, 150)
        assert bar.total == 150

    def test_update_transfer_bar_skips_growth_when_total_already_seeded(self):
        bar = _RecordingBar(total=100)
        _update_transfer_bar(bar, 50)
        assert bar.n == 50
        assert bar.total == 100

    def test_update_transfer_bar_grows_hidden_total(self):
        bar = _RecordingBar(total=XET_TRANSFER_BAR_MIN_TOTAL)
        _update_transfer_bar(bar, 2_000_000)
        assert bar.n == 2_000_000
        assert bar.total > 2_000_000

    def test_finish_transfer_bar_marks_complete(self):
        bar = _RecordingBar(total=10_000_000)
        bar.n = 2_000_000
        _finish_transfer_bar(bar)
        assert bar.total == 2_000_000
