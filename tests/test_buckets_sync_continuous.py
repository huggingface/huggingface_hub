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
import time
from unittest.mock import MagicMock, patch

import pytest

from huggingface_hub._buckets import (
    FilterMatcher,
    _compute_sync_plan,
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

    def test_negative_settle_time_is_refused(self, mock_api, sync_dir):
        with pytest.raises(ValueError, match="settle_time cannot be negative"):
            sync_bucket_internal(sync_dir, BUCKET, api=mock_api, continuous=True, settle_time=-1)


class TestSettleTime:
    """`min_age_seconds` keeps files that are still being written out of the plan."""

    def _plan(self, sync_dir, min_age_seconds):
        with patch("huggingface_hub._buckets._list_remote_files", return_value=iter([])):
            return _compute_sync_plan(
                source=sync_dir,
                dest=BUCKET,
                api=MagicMock(),
                filter_matcher=FilterMatcher(),
                min_age_seconds=min_age_seconds,
            )

    def test_fresh_file_is_skipped(self, sync_dir):
        # a.txt was just written, so a settle window of an hour must exclude it.
        plan = self._plan(sync_dir, min_age_seconds=3600)
        assert plan.summary()["uploads"] == 0

    def test_settled_file_is_uploaded(self, sync_dir):
        old = time.time() - 3600
        os.utime(os.path.join(sync_dir, "a.txt"), (old, old))
        plan = self._plan(sync_dir, min_age_seconds=5)
        assert plan.summary()["uploads"] == 1

    def test_disabled_settle_uploads_everything(self, sync_dir):
        plan = self._plan(sync_dir, min_age_seconds=0)
        assert plan.summary()["uploads"] == 1

    def test_settle_with_delete_is_refused(self, sync_dir):
        # An unsettled file is invisible to the comparison, so it would look like a remote deletion.
        with pytest.raises(ValueError, match="min_age_seconds cannot be combined with delete"):
            _compute_sync_plan(source=sync_dir, dest=BUCKET, api=MagicMock(), delete=True, min_age_seconds=5)


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
                settle_time=kwargs.pop("settle_time", 5.0),
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
        plan = MagicMock()
        plan.summary.return_value = {"uploads": uploads, "downloads": 0, "deletes": 0, "skips": 0, "total_size": 0}
        return plan

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

    def test_settle_time_is_passed_through(self):
        _, compute, _ = self._run(lambda **kw: self._plan_with(0), passes_before_stop=2, settle_time=42.0)
        assert all(call.kwargs["min_age_seconds"] == 42.0 for call in compute.call_args_list)

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
