# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import os
from unittest import mock

from huggingface_hub.cli.jobs import _tabulate, _truncate_to_width


class TestJobsTabulate:
    def test_truncate_to_width_handles_narrow_columns(self) -> None:
        """Regression: ``col_width < 3`` must not produce output longer than the value.

        The previous ``value[: col_width - 3] + "..."`` form turned into a negative
        slice for ``col_width`` in (0, 1, 2), dropping trailing chars and then
        overflowing the column with the ellipsis. The result was longer than the
        original value and longer than the column, corrupting the table layout.
        """
        value = "averylongvalue"
        for col_width in range(0, 8):
            truncated = _truncate_to_width(value, col_width)
            assert len(truncated) <= col_width
            assert truncated == truncated.strip("\n")  # no stray newlines

    def test_truncate_to_width_keeps_short_values(self) -> None:
        assert _truncate_to_width("abc", 5) == "abc"
        assert _truncate_to_width("abc", 3) == "abc"

    def test_truncate_to_width_appends_ellipsis_when_room(self) -> None:
        assert _truncate_to_width("abcdef", 5) == "ab..."
        assert _truncate_to_width("abcdef", 4) == "a..."

    def test_tabulate_does_not_overflow_on_narrow_terminal(self) -> None:
        """When the terminal-fit loop compresses columns below 3, no cell may exceed its width."""
        rows: list[list[str | int]] = [
            ["owner-ns/job-1234567890", "RUNNING", "2026-07-20"],
            ["really-long-job-name-here", "DONE", "2026-07-19"],
        ]
        headers = ["JOB ID", "STATUS", "CREATED"]
        with mock.patch("huggingface_hub.cli.jobs.shutil.get_terminal_size") as mocked:
            mocked.return_value = os.terminal_size((25, 24))
            output = _tabulate(rows, headers=headers)

        # Header and data lines should all share the same width (no ragged overflow).
        line_lengths = {len(line) for line in output.split("\n")}
        assert len(line_lengths) == 1
        # No cell value should appear un-truncated in a compressed column.
        assert "owner-ns/job-1234567890" not in output
