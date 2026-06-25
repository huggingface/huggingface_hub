from huggingface_hub.utils._xet_progress_reporting import (
    _finish_transfer_bar,
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
