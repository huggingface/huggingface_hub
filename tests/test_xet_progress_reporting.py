import logging
from types import SimpleNamespace

from huggingface_hub.utils._xet_progress_reporting import (
    XetDownloadProgressReporter,
    _finish_transfer_bar,
    _format_speed_postfix,
    _set_aggregate_rate_postfix,
    _set_monotonic_total,
    _update_transfer_bar,
)


class _RecordingBar:
    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total")
        self.n = 0

    def update(self, n: int) -> None:
        self.n += n

    def refresh(self) -> None:
        pass


class _FakeTqdm:
    """Stub covering the whole bar surface the reporter drives, recording every instance created."""

    created: list["_FakeTqdm"] = []

    def __init__(self, **kwargs):
        self.total = kwargs.get("total")
        self.n = 0
        self.closed = False
        _FakeTqdm.created.append(self)

    def update(self, n: int) -> None:
        self.n += n

    def set_postfix_str(self, postfix: str, refresh: bool = False) -> None:
        self.postfix = postfix

    def refresh(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _RateBar:
    """Stub bar exposing a tqdm-like ``format_dict['rate']`` and recording its postfix."""

    def __init__(self, rate):
        self._rate = rate
        self.postfix = None

    @property
    def format_dict(self):
        return {"rate": self._rate}

    def set_postfix_str(self, postfix: str, refresh: bool = False) -> None:
        self.postfix = postfix


class TestXetProgressBarHelpers:
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
        bar = _RecordingBar(total=0)
        _update_transfer_bar(bar, 2_000_000)
        assert bar.n == 2_000_000
        assert bar.total > 2_000_000

    def test_finish_transfer_bar_marks_complete(self):
        bar = _RecordingBar(total=10_000_000)
        bar.n = 2_000_000
        _finish_transfer_bar(bar)
        assert bar.total == 2_000_000

    def test_aggregate_rate_postfix_reports_bar_own_summed_rate(self):
        # Regression: shared snapshot bar must show its own aggregated throughput, not a per-file rate.
        # https://github.com/huggingface/huggingface_hub/issues/4519
        bar = _RateBar(rate=234_000_000)  # bytes/s summed across all files
        _set_aggregate_rate_postfix(bar)
        assert "MB/s" in bar.postfix
        assert bar.postfix == _format_speed_postfix(234_000_000)

    def test_aggregate_rate_postfix_handles_unknown_rate(self):
        bar = _RateBar(rate=None)
        _set_aggregate_rate_postfix(bar)
        assert "???" in bar.postfix


class TestXetDownloadProgressReporter:
    def test_custom_tqdm_class_receives_both_download_bars(self):
        # Regression: the transfer bar was hardcoded to the built-in tqdm, so a caller-supplied
        # `tqdm_class` drove only the reconstruction bar and Xet downloads opened a second bar
        # the caller had no handle on.
        _FakeTqdm.created.clear()
        reporter = XetDownloadProgressReporter(
            reconstruction_desc="reconstructing file",
            transfer_desc="downloading bytes",
            total=100,
            log_level=logging.INFO,
            name="huggingface_hub.test",
            tqdm_class=_FakeTqdm,
        )
        reporter.update_progress(
            SimpleNamespace(
                total_bytes_completed=10,
                total_transfer_bytes_completed=40,
                total_bytes_completion_rate=None,
                total_transfer_bytes_completion_rate=None,
                total_bytes=100,
            )
        )
        reporter.close()

        assert _FakeTqdm.created == [reporter.reconstruction_bar, reporter.transfer_bar]
        assert reporter.reconstruction_bar.n == 10
        assert reporter.transfer_bar.n == 40
        assert all(bar.closed for bar in _FakeTqdm.created)
