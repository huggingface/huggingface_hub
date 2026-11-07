from huggingface_hub.utils._xet_progress_reporting import (
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
