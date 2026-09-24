from types import SimpleNamespace
from unittest.mock import patch

import huggingface_hub.utils._xet_progress_reporting as xet_progress
from huggingface_hub.utils._xet_progress_reporting import (
    XetUploadProgressReporter,
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


class _PositionBar:
    @staticmethod
    def format_sizeof(value):
        return str(value)

    def __init__(self, *args, **kwargs):
        self.pos = -kwargs["position"]
        self.n = kwargs.get("initial", 0)
        self.total = kwargs.get("total", 0)
        self.cleared = False

    def clear(self):
        self.cleared = True

    def refresh(self):
        pass

    def update(self, n):
        self.n += n

    def set_description(self, *args, **kwargs):
        pass

    def set_postfix_str(self, *args, **kwargs):
        pass

    def close(self):
        pass


def test_validating_bar_starts_with_shards_and_moves_before_next_file_bar():
    def report(total_shards):
        return SimpleNamespace(
            total_bytes=20,
            total_bytes_completed=10,
            total_bytes_completion_rate=None,
            total_transfer_bytes=20,
            total_transfer_bytes_completed=10,
            total_transfer_bytes_completion_rate=None,
            shard=SimpleNamespace(
                total_shards=total_shards,
                total_shards_completed=0,
                total_shard_validation_entries=0,
                total_shard_validation_entries_completed=0,
            ),
        )

    def item(name):
        return SimpleNamespace(item_name=name, bytes_completed=5, total_bytes=10)

    with patch.object(xet_progress, "tqdm", _PositionBar):
        reporter = XetUploadProgressReporter(n_lines=2)
        reporter.per_file_progress = True
        reporter.update_progress(report(0), {"a": item("a")})
        assert reporter.validating_bar is None

        reporter.update_progress(report(1), {"a": item("a")})
        assert reporter.validating_bar.pos == -3

        reporter.update_progress(report(1), {"a": item("a"), "b": item("b")})
        assert reporter.validating_bar.cleared
        assert reporter.validating_bar.pos == -4
        assert reporter.current_bars[1].pos == -3


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
