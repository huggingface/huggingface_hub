# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Unit tests for `XetProgressReporter` (no network / no xet binding required)."""

from types import SimpleNamespace

import pytest

from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
from huggingface_hub.utils._xet_progress_reporting import XetProgressReporter


@pytest.fixture(autouse=True)
def _silence_progress_bars():
    # Keep test output clean and TTY-independent; the reporter objects still behave normally.
    disable_progress_bars()
    yield
    enable_progress_bars()


def _group(total_bytes, bytes_completed, transfer_total, transfer_completed):
    return SimpleNamespace(
        total_bytes=total_bytes,
        total_bytes_completed=bytes_completed,
        total_bytes_completion_rate=1.0,
        total_transfer_bytes=transfer_total,
        total_transfer_bytes_completed=transfer_completed,
        total_transfer_bytes_completion_rate=1.0,
    )


def _item(name, bytes_completed, total_bytes):
    return SimpleNamespace(item_name=name, bytes_completed=bytes_completed, total_bytes=total_bytes)


class TestFinalizingIndicator:
    def test_no_finalize_line_while_transfer_in_progress(self):
        reporter = XetProgressReporter()
        try:
            # Processing/transfer still in flight: no finalization indicator yet.
            reporter.update_progress(_group(1000, 400, 800, 300), {"a.bin": _item("a.bin", 400, 1000)})
            assert getattr(reporter, "_finalize_bar", None) is None
        finally:
            reporter.close()

    def test_finalize_line_appears_once_transfer_completes(self):
        reporter = XetProgressReporter()
        try:
            # All data processed and all unique bytes transferred -> finalizing (metadata upload).
            reporter.update_progress(_group(1000, 1000, 800, 800), {"a.bin": _item("a.bin", 1000, 1000)})
            assert getattr(reporter, "_finalize_bar", None) is not None
        finally:
            reporter.close()

    def test_finalize_line_appears_when_fully_deduplicated(self):
        # Nothing new to transfer (everything deduplicated) still finalizes shards/metadata.
        reporter = XetProgressReporter()
        try:
            reporter.update_progress(_group(1000, 1000, 0, 0), {"a.bin": _item("a.bin", 1000, 1000)})
            assert getattr(reporter, "_finalize_bar", None) is not None
        finally:
            reporter.close()

    def test_reset_for_next_commit_clears_finalize_state(self):
        reporter = XetProgressReporter()
        try:
            reporter.update_progress(_group(1000, 1000, 800, 800), {"a.bin": _item("a.bin", 1000, 1000)})
            assert getattr(reporter, "_finalize_bar", None) is not None
            reporter.reset_for_next_commit()
            assert getattr(reporter, "_finalize_bar", None) is None
            assert getattr(reporter, "_finalize_start", None) is None
        finally:
            reporter.close()
