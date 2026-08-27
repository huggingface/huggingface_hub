# coding=utf-8
# Copyright 2026-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline tests for continuous bucket sync (`hf buckets sync --continuous`)."""

import os
import re
import time
from unittest.mock import MagicMock, patch

import pytest

from huggingface_hub._buckets import (
    CALM_LARGE_SIZE,
    CALM_SMALL_SIZE,
    DEFAULT_CALM_MAX,
    DEFAULT_CALM_MIN,
    HIGH_ETA_SECONDS,
    FilterMatcher,
    StabilityTracker,
    SyncOperation,
    SyncPlan,
    WatchDisplay,
    WatchRow,
    _bar,
    _calm_period_for_size,
    _churn_meter,
    _compute_sync_plan,
    _format_duration,
    _sync_continuous_loop,
    sync_bucket_internal,
)


BUCKET = "hf://buckets/user/my-bucket"


@pytest.fixture
def sync_dir(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    return str(tmp_path)


@pytest.fixture
def mock_api():
    return MagicMock()


class TestContinuousValidation:
    """`--continuous` is refused in combinations that are meaningless or unsafe."""

    def test_continuous_with_delete_is_refused(self, mock_api, sync_dir):
        # Deliberate: an unattended loop must never propagate deletions to the remote.
        with pytest.raises(ValueError, match="Cannot specify delete with continuous"):
            sync_bucket_internal(sync_dir, BUCKET, api=mock_api, continuous=True, delete=True)

    def test_continuous_with_plan_is_refused(self, mock_api, sync_dir):
        with pytest.raises(ValueError, match="Cannot specify continuous with plan or dry_run"):
            sync_bucket_internal(sync_dir, BUCKET, api=mock_api, continuous=True, plan="p.jsonl")

    def test_continuous_with_dry_run_is_refused(self, mock_api, sync_dir):
        with pytest.raises(ValueError, match="Cannot specify continuous with plan or dry_run"):
            sync_bucket_internal(sync_dir, BUCKET, api=mock_api, continuous=True, dry_run=True)

    def test_continuous_with_apply_is_refused(self, mock_api):
        with pytest.raises(ValueError, match="Cannot specify continuous when using apply"):
            sync_bucket_internal(api=mock_api, continuous=True, apply="p.jsonl")

    @pytest.mark.parametrize("interval", [0, -1])
    def test_non_positive_interval_is_refused(self, mock_api, sync_dir, interval):
        with pytest.raises(ValueError, match="interval must be greater than 0"):
            sync_bucket_internal(sync_dir, BUCKET, api=mock_api, continuous=True, interval=interval)

    def test_negative_calm_time_is_refused(self, mock_api, sync_dir):
        with pytest.raises(ValueError, match="calm_time cannot be negative"):
            sync_bucket_internal(sync_dir, BUCKET, api=mock_api, continuous=True, calm_time=-1)

    def test_inverted_calm_range_is_refused(self, mock_api, sync_dir):
        with pytest.raises(ValueError, match="calm_max cannot be smaller than calm_min"):
            sync_bucket_internal(sync_dir, BUCKET, api=mock_api, continuous=True, calm_min=100, calm_max=10)


class TestCalmPeriodRamp:
    """Calm period scales with file size: flat, ramp, flat."""

    def test_small_files_use_the_floor(self):
        assert _calm_period_for_size(4096) == DEFAULT_CALM_MIN
        assert _calm_period_for_size(CALM_SMALL_SIZE) == DEFAULT_CALM_MIN

    def test_large_files_use_the_ceiling(self):
        assert _calm_period_for_size(CALM_LARGE_SIZE) == DEFAULT_CALM_MAX
        assert _calm_period_for_size(CALM_LARGE_SIZE * 100) == DEFAULT_CALM_MAX

    def test_ramp_is_monotonic_between_anchors(self):
        sizes = [CALM_SMALL_SIZE * m for m in (1, 2, 4, 8, 10)]
        periods = [_calm_period_for_size(sz) for sz in sizes]
        assert periods == sorted(periods)
        assert DEFAULT_CALM_MIN < periods[2] < DEFAULT_CALM_MAX

    def test_custom_anchors_are_honoured(self):
        assert _calm_period_for_size(1024, calm_min=5, calm_max=50) == 5
        assert _calm_period_for_size(CALM_LARGE_SIZE, calm_min=5, calm_max=50) == 50


class TestStabilityTracker:
    """Files that are still changing are held back; quiet files go through."""

    def test_static_file_is_calm_immediately(self):
        # mtime an hour old: nothing to wait for, even on the very first pass.
        tracker = StabilityTracker()
        old_mtime_ms = (time.time() - 3600) * 1000
        assert tracker.is_calm("a.txt", 1000, old_mtime_ms)

    def test_freshly_written_file_is_held_back(self):
        tracker = StabilityTracker()
        verdict = tracker.verdict("a.txt", 1000, time.time() * 1000)
        assert not verdict.calm
        assert "needs" in verdict.reason

    def test_rapidly_changing_file_is_held_back_while_it_churns(self):
        # A file rewritten every second never accumulates the required quiet time.
        tracker = StabilityTracker(calm_min=60.0)
        now = time.time()
        for i in range(5):
            verdict = tracker.verdict("log.txt", 100 * (i + 1), (now + i) * 1000, now=now + i)
        assert tracker._files["log.txt"].is_rapid
        assert not verdict.calm

    def test_file_that_churned_then_settled_is_eventually_uploaded(self):
        """Regression: historical churn must not block a file forever once it goes quiet."""
        tracker = StabilityTracker(calm_min=60.0)
        now = time.time()
        for i in range(5):
            tracker.verdict("log.txt", 100 * (i + 1), (now + i) * 1000, now=now + i)
        assert tracker._files["log.txt"].is_rapid  # still remembers the churn
        # Writing stopped at now+4; ask again well past the calm period with an unchanged file.
        verdict = tracker.verdict("log.txt", 500, (now + 4) * 1000, now=now + 400)
        assert verdict.calm, f"still blocked despite being quiet: {verdict.reason}"

    def test_calm_override_of_zero_disables_checks(self):
        tracker = StabilityTracker(calm_override=0)
        assert tracker.is_calm("a.txt", 1000, time.time() * 1000)

    def test_mean_interval_needs_two_samples(self):
        tracker = StabilityTracker()
        now = time.time()
        tracker.observe("a.txt", 1, now * 1000, now=now)
        tracker.observe("a.txt", 2, (now + 5) * 1000, now=now + 5)
        assert tracker._files["a.txt"].mean_interval is None  # one change == no interval yet
        tracker.observe("a.txt", 3, (now + 10) * 1000, now=now + 10)
        assert tracker._files["a.txt"].mean_interval == pytest.approx(5.0)

    def test_forget_drops_vanished_files(self):
        tracker = StabilityTracker()
        tracker.observe("a.txt", 1, time.time() * 1000)
        tracker.observe("b.txt", 1, time.time() * 1000)
        tracker.forget({"a.txt"})
        assert set(tracker._files) == {"a.txt"}

    def test_local_filter_with_delete_is_refused(self, sync_dir):
        # A held-back file is invisible to the comparison, so it would look like a remote deletion.
        with pytest.raises(ValueError, match="local_filter cannot be combined with delete"):
            _compute_sync_plan(
                source=sync_dir, dest=BUCKET, api=MagicMock(), delete=True, local_filter=lambda *a: True
            )

    def test_plan_excludes_held_back_files(self, sync_dir):
        tracker = StabilityTracker()  # sync_dir files were just written -> not calm
        with patch("huggingface_hub._buckets._list_remote_files", return_value=iter([])):
            plan = _compute_sync_plan(
                source=sync_dir,
                dest=BUCKET,
                api=MagicMock(),
                filter_matcher=FilterMatcher(),
                local_filter=tracker.is_calm,
            )
        assert plan.summary()["uploads"] == 0

    def test_plan_includes_settled_files(self, sync_dir):
        old = time.time() - 7200
        os.utime(os.path.join(sync_dir, "a.txt"), (old, old))
        with patch("huggingface_hub._buckets._list_remote_files", return_value=iter([])):
            plan = _compute_sync_plan(
                source=sync_dir,
                dest=BUCKET,
                api=MagicMock(),
                filter_matcher=FilterMatcher(),
                local_filter=StabilityTracker().is_calm,
            )
        assert plan.summary()["uploads"] == 1


class TestThrashAbort:
    """An expensive upload is refused when the file would change out from under it."""

    def _tracker_changing_every(self, seconds, calm_min=1.0):
        tracker = StabilityTracker(calm_min=calm_min)
        now = time.time()
        for i in range(4):
            tracker.observe("big.bin", 1000 * (i + 1), (now + i * seconds) * 1000, now=now + i * seconds)
        return tracker, now + 4 * seconds

    def test_slow_upload_of_churning_file_is_refused(self):
        # Changes every 30s (not "rapid"), but a huge file at 1 B/s gives an enormous ETA.
        tracker, now = self._tracker_changing_every(30.0)
        tracker._throughput_bps = 1.0
        tracker._measured = True
        verdict = tracker.verdict("big.bin", 10_000, (now - 600) * 1000, now=now + 600)
        assert not verdict.calm
        assert "overtaken" in verdict.reason
        assert verdict.predicted_changes >= 2

    def test_fast_upload_of_same_file_is_allowed(self):
        # Same churn, but throughput makes the upload near-instant, so there is nothing to lose.
        tracker, now = self._tracker_changing_every(30.0)
        tracker._throughput_bps = 10**9
        tracker._measured = True
        verdict = tracker.verdict("big.bin", 10_000, (now - 600) * 1000, now=now + 600)
        assert verdict.calm
        assert verdict.eta < HIGH_ETA_SECONDS

    def test_record_transfer_updates_throughput(self):
        tracker = StabilityTracker()
        tracker.record_transfer(100 * 1024**2, 1.0)
        assert tracker._throughput_bps == pytest.approx(100 * 1024**2)
        # Later measurements are smoothed, not replaced outright.
        tracker.record_transfer(0, 1.0)
        assert tracker._throughput_bps == pytest.approx(100 * 1024**2)

    def test_zero_duration_transfer_is_ignored(self):
        tracker = StabilityTracker()
        before = tracker._throughput_bps
        tracker.record_transfer(1000, 0)
        assert tracker._throughput_bps == before


class TestFormatDuration:
    @pytest.mark.parametrize(
        "seconds,expected",
        [(0, "0s"), (9.4, "9s"), (59, "59s"), (60, "1m00s"), (130, "2m10s"), (3600, "1h00m"), (3900, "1h05m")],
    )
    def test_formats(self, seconds, expected):
        assert _format_duration(seconds) == expected

    def test_negative_clamps_to_zero(self):
        assert _format_duration(-5) == "0s"


class TestContinuousLoop:
    """The loop re-plans each pass, survives failures, and exits cleanly on Ctrl-C."""

    def _run(self, compute_side_effect, execute=None, passes_before_stop=3, **kwargs):
        """Run the loop, raising KeyboardInterrupt from sleep after N passes."""
        calls = {"sleep": 0}

        def fake_sleep(_seconds):
            calls["sleep"] += 1
            if calls["sleep"] >= passes_before_stop:
                raise KeyboardInterrupt

        with (
            patch("huggingface_hub._buckets._compute_sync_plan", side_effect=compute_side_effect) as compute,
            patch("huggingface_hub._buckets._execute_plan", side_effect=execute) as execute_mock,
            patch("huggingface_hub._buckets.time.sleep", side_effect=fake_sleep),
        ):
            plan = _sync_continuous_loop(
                source="/tmp/x",
                dest=BUCKET,
                api=MagicMock(),
                filter_matcher=FilterMatcher(),
                interval=kwargs.pop("interval", 1.0),
                tracker=kwargs.pop("tracker", StabilityTracker(calm_override=0)),
                ignore_times=False,
                ignore_sizes=False,
                existing=False,
                ignore_existing=False,
                verbose=False,
                quiet=True,
            )
        return plan, compute, execute_mock

    @staticmethod
    def _plan_with(uploads):
        """A real SyncPlan -- the loop iterates `operations`, so a MagicMock will not do."""
        return SyncPlan(
            source="/tmp/x",
            dest=BUCKET,
            timestamp="now",
            operations=[
                SyncOperation(action="upload", path=f"f{i}.bin", size=10, reason="new file") for i in range(uploads)
            ],
        )

    def test_replans_every_pass(self):
        _, compute, _ = self._run(lambda **kw: self._plan_with(0), passes_before_stop=3)
        assert compute.call_count == 3

    def test_keyboard_interrupt_exits_cleanly(self):
        # No exception should escape: Ctrl-C is the documented way to stop.
        plan, _, _ = self._run(lambda **kw: self._plan_with(0), passes_before_stop=2)
        assert plan is not None

    def test_executes_only_when_there_is_work(self):
        plans = [self._plan_with(0), self._plan_with(2), self._plan_with(0)]
        _, _, execute = self._run(lambda **kw: plans.pop(0), passes_before_stop=3)
        assert execute.call_count == 1

    def test_never_deletes(self):
        _, compute, _ = self._run(lambda **kw: self._plan_with(0), passes_before_stop=2)
        assert all(call.kwargs["delete"] is False for call in compute.call_args_list)

    def test_local_filter_is_passed_through(self):
        _, compute, _ = self._run(lambda **kw: self._plan_with(0), passes_before_stop=2)
        assert all(callable(call.kwargs["local_filter"]) for call in compute.call_args_list)

    def test_failed_pass_does_not_stop_the_loop(self):
        results = [RuntimeError("hub down"), self._plan_with(1), self._plan_with(0)]

        def compute(**kw):
            item = results.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        _, compute_mock, execute = self._run(compute, passes_before_stop=3)
        # The loop kept going after the failure and did real work on a later pass.
        assert compute_mock.call_count == 3
        assert execute.call_count == 1


class TestWatchDisplay:
    """The watch panel repaints in place on a TTY and degrades to plain lines elsewhere."""

    def _tty_display(self, monkeypatch):
        monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
        return WatchDisplay("./data", BUCKET, "60s")

    def test_plain_mode_when_not_a_tty(self, capsys):
        # stdout is captured (not a TTY) under pytest, so this is the default path here.
        display = WatchDisplay("./data", BUCKET, "60s")
        assert not display.tui
        display.start()
        out = capsys.readouterr().out
        assert "watching::" in out
        assert "lazy mode active" in out
        assert "\033[" not in out  # no cursor movement in a log file

    def test_plain_mode_reports_a_file_only_when_its_status_changes(self, capsys):
        display = WatchDisplay("./data", BUCKET, "60s")
        display.start()
        capsys.readouterr()
        row = WatchRow("a.log", 3, 60, "waiting", 2.0)
        display.render([row])
        first = capsys.readouterr().out
        display.render([row])  # identical status -> silence
        second = capsys.readouterr().out
        assert "a.log" in first
        assert second == ""

    def test_plain_mode_reprints_when_status_changes(self, capsys):
        display = WatchDisplay("./data", BUCKET, "60s")
        display.render([WatchRow("a.log", 3, 60, "waiting", 2.0)])
        capsys.readouterr()
        display.render([WatchRow("a.log", 30, 60, "waiting", 2.0)])
        assert "a.log" in capsys.readouterr().out

    def test_quiet_suppresses_everything(self, capsys):
        display = WatchDisplay("./data", BUCKET, "60s", quiet=True)
        display.start()
        display.render([WatchRow("a.log", 3, 60, "waiting", 2.0)])
        display.close("done")
        assert capsys.readouterr().out == ""

    def test_tui_moves_the_cursor_back_over_its_own_block(self, monkeypatch, capsys):
        display = self._tty_display(monkeypatch)
        assert display.tui
        display.render([WatchRow("a.log", 3, 60, "waiting", 2.0)])
        first = capsys.readouterr().out
        painted = display._painted
        display.render([WatchRow("a.log", 8, 60, "waiting", 2.0)])
        second = capsys.readouterr().out
        assert not re.search(r"\033\[\d+A", first)  # first frame does not rewind: nothing painted yet
        assert f"\033[{painted}A" in second  # second frame rewinds exactly as far as it painted

    def test_tui_clears_leftovers_when_the_frame_shrinks(self, monkeypatch, capsys):
        display = self._tty_display(monkeypatch)
        display.render([WatchRow(f"f{i}.log", 3, 60, "waiting") for i in range(4)])
        tall = display._painted
        capsys.readouterr()
        display.render([])
        out = capsys.readouterr().out
        assert display._painted < tall
        assert "\033[2K" in out  # leftover rows are erased, not left on screen

    def test_row_shows_progress_and_eta(self, monkeypatch, capsys):
        display = self._tty_display(monkeypatch)
        display.render(
            [
                WatchRow("waiting.log", 30, 60, "waiting", 2.0),
                WatchRow("ready.bin", 0, 60, "ready", None, 3.0),
                WatchRow("held.bin", 60, 60, "deferred", 45.0, 250.0, 3.0),
            ]
        )
        out = capsys.readouterr().out
        assert "3 files changed" in out
        assert "30s/1m00s" in out and "ready in 30s" in out
        assert "eta 3s" in out
        assert "overtaken ~3x" in out

    def test_singular_and_empty_wording(self, monkeypatch, capsys):
        display = self._tty_display(monkeypatch)
        display.render([WatchRow("a.log", 3, 60, "waiting")])
        assert "1 file changed" in capsys.readouterr().out
        display.render([])
        assert "no changes" in capsys.readouterr().out

    def test_long_names_are_truncated_from_the_left(self, monkeypatch, capsys):
        display = self._tty_display(monkeypatch)
        display.render([WatchRow("a/very/deeply/nested/path/to/some/checkpoint-000123.safetensors", 3, 60, "waiting")])
        out = capsys.readouterr().out
        assert "\u2026" in out
        assert "checkpoint-000123.safetensors" in out  # the informative tail survives


class TestMeters:
    def test_bar_endpoints(self):
        assert _bar(0.0) == "\u2591" * 8
        assert _bar(1.0) == "\u2588" * 8
        assert _bar(0.5) == "\u2588" * 4 + "\u2591" * 4

    def test_bar_clamps_out_of_range(self):
        assert _bar(-1.0) == "\u2591" * 8
        assert _bar(5.0) == "\u2588" * 8

    def test_churn_meter_is_blank_without_samples(self):
        assert _churn_meter(None).strip() == ""

    def test_churn_meter_grows_with_rate(self):
        rapid = _churn_meter(1.0).strip()
        moderate = _churn_meter(4.0).strip()
        calm = _churn_meter(60.0).strip()
        assert len(rapid) > len(moderate) >= len(calm)
