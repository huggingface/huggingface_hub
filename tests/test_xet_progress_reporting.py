import logging
from types import SimpleNamespace

from huggingface_hub.utils._xet_progress_reporting import XetDownloadProgressReporter


def _group_report(*, bytes_completed: int = 0, transfer_completed: int = 0, total_bytes: int = 100) -> SimpleNamespace:
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

    def update(self, n: int) -> None:
        self.updates.append(n)

    def set_postfix_str(self, postfix: str, refresh: bool = False) -> None:
        pass


class _AggregatedBar(_RecordingBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transfer_updates: list[int] = []

    def update_transfer(self, n: int) -> None:
        self.transfer_updates.append(n)

    def set_transfer_postfix_str(self, postfix: str, refresh: bool = False) -> None:
        pass


class TestXetDownloadProgressReporter:
    def test_updates_both_bars_independently(self):
        reconstruction_bar = _RecordingBar(total=100)
        transfer_bar = _RecordingBar()

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
        reporter = XetDownloadProgressReporter(
            reconstruction_desc="reconstructing",
            total=100,
            log_level=logging.CRITICAL,
            tqdm_class=_AggregatedBar,
        )
        aggregated = reporter.reconstruction_bar
        assert reporter.transfer_bar is aggregated

        reporter.update_progress(_group_report(transfer_completed=42))
        assert aggregated.transfer_updates == [42]
        assert aggregated.updates == []

    def test_negative_increments_are_ignored(self):
        reconstruction_bar = _RecordingBar(total=100)
        transfer_bar = _RecordingBar()

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
