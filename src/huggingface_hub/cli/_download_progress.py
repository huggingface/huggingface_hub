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

import threading
import time
from collections.abc import Iterable, Iterator
from typing import Any, cast

from tqdm.auto import tqdm as base_tqdm

from huggingface_hub.errors import CLIError


def get_rich_progress_tqdm(name: str | None = None) -> type[base_tqdm]:
    try:
        from rich.console import Console, Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError as error:
        raise CLIError("`--rich-progress` requires the `rich` package to be installed.") from error

    class RichDownloadProgress:
        _lock = threading.RLock()
        _default_name = name
        _minimum_speed_sample_bytes = 1024 * 1024

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.iterable: Iterable[Any] | None = args[0] if args else None
            self.unit = kwargs.pop("unit", None)
            desc = kwargs.pop("desc", None)
            self.display_name = str(self._default_name or desc or "").strip()
            self.has_explicit_name = self._default_name is not None
            self.location: str | None = None
            self.enabled = self.unit == "B"
            self.total: int | None = kwargs.pop("total", None)
            self.n = int(kwargs.pop("initial", 0) or 0)
            self.display_n = float(self.n)
            self.current_file: str | None = None
            self.current_file_total: int | None = None
            self.current_file_downloaded = 0
            self.current_file_index: int | None = None
            self.total_files: int | None = None
            self.start_time = time.monotonic()
            self.current_file_started_at = self.start_time
            self.last_update_time = self.start_time
            self.last_render_time = self.start_time
            self.speed_samples_mb: list[float] = []
            self.final_status: str | None = None
            self.console = Console(stderr=True)
            self.live = (
                Live(self, console=self.console, refresh_per_second=20, auto_refresh=False, transient=True)
                if self.enabled
                else None
            )
            self.refresh_interval = 0.05
            self.refresh_stop_event = threading.Event()
            self.refresh_thread: threading.Thread | None = None
            self.started = False
            self.closed = False
            self._start()

        @classmethod
        def get_lock(cls) -> threading.RLock:
            return cls._lock

        @classmethod
        def set_lock(cls, lock: threading.RLock) -> None:
            cls._lock = lock

        def __enter__(self) -> "RichDownloadProgress":
            self._start()
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
            self.close(failed=exc_type is not None)

        def _start(self) -> None:
            if self.live is not None and not self.started and not self.closed:
                self.live.start(refresh=True)
                self.started = True
                self.refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
                self.refresh_thread.start()

        def _refresh_loop(self) -> None:
            try:
                while not self.refresh_stop_event.wait(self.refresh_interval):
                    self.refresh()
            except BaseException:
                self.refresh_stop_event.set()

        def __iter__(self) -> Iterator[Any]:
            if self.iterable is None:
                return
            try:
                for item in self.iterable:
                    yield item
                    self.update()
            except GeneratorExit:
                self.close()
                raise
            except BaseException:
                self.close(failed=True)
                raise

        def update(self, n: int | float | None = 1) -> None:
            now = time.monotonic()
            increment = 1 if n is None else n
            self.n += int(increment)
            self.current_file_downloaded = max(0, self.current_file_downloaded + int(increment))

            if not self.enabled:
                return

            if now - self.last_update_time < 0.2 and self.n != self.total:
                return

            self.last_update_time = now
            if self.live is not None:
                self.live.update(self._render())

        def refresh(self) -> None:
            if self.live is not None and not self.closed:
                self.live.update(self._render(), refresh=True)

        def close(self, *, completed: bool = False, failed: bool = False) -> None:
            if self.closed:
                return

            self.closed = True
            self.refresh_stop_event.set()
            if (
                self.refresh_thread is not None
                and self.refresh_thread.is_alive()
                and threading.current_thread() is not self.refresh_thread
            ):
                self.refresh_thread.join(timeout=1)
            if self.live is not None and self.started:
                if failed and self.enabled:
                    self._stop_live_with_failed_panel()
                else:
                    self.live.stop()
            if completed and self.enabled:
                self._print_summary(completed=True)

        def set_location(self, location: str) -> None:
            self.location = location

        def set_total_files(self, total_files: int | None) -> None:
            self.total_files = total_files
            self.refresh()

        def set_current_file_index(self, index: int | None) -> None:
            self.current_file_index = index
            self.refresh()

        def set_current_file(self, desc: str | None, total: int | None = None, initial: int = 0) -> None:
            self.current_file = str(desc or "").strip() or None
            self.current_file_total = total
            self.current_file_downloaded = max(0, initial)
            self.current_file_started_at = time.monotonic()
            self.refresh()

        def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
            if desc is not None and not self.has_explicit_name and "complete" not in desc.lower():
                self.display_name = desc
            if refresh:
                self.refresh()
            if desc is not None and "complete" in desc.lower():
                self.close(completed=True)

        def __rich__(self) -> Panel:
            return self._render()

        def _average_speed_bytes(self, now: float) -> float:
            elapsed = max(now - self.start_time, 1e-9)
            return self.n / elapsed

        def _predicted_target(self, now: float) -> float:
            average_speed_bytes = self._average_speed_bytes(now)
            if average_speed_bytes <= 0:
                return float(self.n)

            time_since_update = max(now - self.last_update_time, 0.0)
            prediction_window = min(time_since_update, 2.0)
            predicted = self.n + average_speed_bytes * prediction_window
            if self.total is not None:
                predicted = min(predicted, float(self.total))
            return max(predicted, float(self.n))

        def _advance_display(self, now: float) -> None:
            elapsed = max(now - self.last_render_time, 1e-9)
            previous_display_n = self.display_n
            target_n = self._predicted_target(now)
            distance = target_n - self.display_n
            average_speed_bytes = self._average_speed_bytes(now)

            if distance > 0:
                # Move at a mostly linear pace. If the display is behind confirmed bytes,
                # catch up over a short fixed horizon instead of jumping by callback batches.
                catchup_speed = max(0.0, float(self.n) - self.display_n) / 1.5
                display_speed = max(average_speed_bytes, catchup_speed)
                self.display_n += max(1.0, display_speed * elapsed)
                self.display_n = min(self.display_n, target_n)
            elif distance < 0:
                # If the prediction went too far, wait for confirmed progress instead of moving backward.
                self.display_n = min(self.display_n, target_n + average_speed_bytes * 2.0)

            displayed_delta = max(self.display_n - previous_display_n, 0.0)
            if self.n >= self._minimum_speed_sample_bytes:
                self.speed_samples_mb.append(displayed_delta / elapsed / (1024 * 1024))
                if len(self.speed_samples_mb) > 500:
                    self.speed_samples_mb = self.speed_samples_mb[-500:]
            self.last_render_time = now

        def _format_duration(self, seconds: float | None) -> str:
            if seconds is None:
                return "unknown"
            if seconds < 60:
                return f"{seconds:.0f}s"

            minutes, remaining_seconds = divmod(round(seconds), 60)
            if minutes < 60:
                return f"{minutes}m {remaining_seconds}s"

            hours, remaining_minutes = divmod(minutes, 60)
            return f"{hours}h {remaining_minutes}m"

        def _format_size_mb(self, size: float) -> str:
            return f"{size / (1024 * 1024):.1f} MB"

        def _summary_table(self, *, completed: bool) -> Table:
            elapsed = max(time.monotonic() - self.start_time, 1e-9)
            average_speed_mb = self.n / elapsed / (1024 * 1024)

            table = Table.grid(padding=(0, 2))
            table.add_column(style="bold")
            table.add_column()
            table.add_row("Target", self.display_name or "-")
            if not completed:
                table.add_row("Status", "failed")
                if self.total:
                    table.add_row(
                        "Downloaded",
                        f"{self._format_size_mb(float(self.n))} / {self._format_size_mb(float(self.total))}",
                    )
                else:
                    table.add_row("Downloaded", self._format_size_mb(float(self.n)))
            table.add_row("Average speed", f"{average_speed_mb:.2f} MB/s")
            table.add_row("Time", self._format_duration(elapsed))
            if completed:
                table.add_row("Size", self._format_size_mb(float(self.total or self.n)))
            if not completed and self.location:
                table.add_row("Location", self.location)
            return table

        def _print_summary(self, *, completed: bool) -> None:
            table = self._summary_table(completed=completed)
            title = "Download complete" if completed else "Download failed"
            border_style = "green" if completed else "red"
            summary_console = Console(stderr=True, width=self.console.width)
            summary_console.print(Panel(table, title=title, border_style=border_style))

        def _stop_live_with_failed_panel(self) -> None:
            assert self.live is not None

            if not all(hasattr(self.live, attr) for attr in ("_lock", "_started", "console", "_live_render")):
                self.final_status = "failed"
                self.live.transient = False
                self.live.update(self._render(), refresh=True)
                self.live.stop()
                return

            with self.live._lock:
                if not self.live._started:
                    return

                self.final_status = "failed"
                self.live._started = False
                self.live.console.clear_live()

                if self.live._nested:
                    self.live.console.print(self._render())
                    return

                if self.live.auto_refresh and self.live._refresh_thread is not None:
                    self.live._refresh_thread.stop()
                    self.live._refresh_thread = None

                self.live.vertical_overflow = "visible"
                with self.live.console:
                    self.live._disable_redirect_io()
                    self.live.console.pop_render_hook()
                    if not self.live._alt_screen and not self.live.console.is_jupyter:
                        self.live.console.control(self.live._live_render.restore_cursor())
                    self.live.console.show_cursor(True)
                    if self.live._alt_screen:
                        self.live.console.set_alt_screen(False)
                    self.live.console.print(self._render())

        def _is_waiting_on_current_file(self, now: float) -> bool:
            return (
                self.current_file is not None
                and self.current_file_total is not None
                and self.current_file_downloaded == 0
                and now - self.current_file_started_at >= 0.2
            )

        def _progress_line(self, percent: float | None, width: int, *, pending: bool, now: float) -> Text:
            if percent is None:
                return Text("unknown", style="dim")

            filled = round(width * percent)
            empty = width - filled
            line = Text()

            if filled:
                if percent < 0.33:
                    style = "red"
                elif percent < 0.66:
                    style = "yellow"
                else:
                    style = "green"
                line.append("━" * filled, style=style)

            if empty:
                if pending:
                    pulse = ["─"] * empty
                    pulse_width = min(6, max(2, empty // 8 or 2))
                    pulse_start = int(now * 12) % max(empty, 1)
                    for index in range(pulse_width):
                        pulse[(pulse_start + index) % empty] = "━"

                    for char in pulse:
                        line.append(char, style="cyan" if char == "━" else "dim")
                else:
                    line.append("─" * empty, style="dim")

            return line

        def _speed_graph(self, width: int, height: int = 10) -> Text:
            points = self.speed_samples_mb[-width:]
            if not points:
                return Text("No speed samples yet", style="dim")

            max_speed = max(max(points), 0.1)
            labels = [f"{(row / height) * max_speed:.2f} MB/s" for row in range(height, 0, -1)]
            label_width = max(len(label) for label in labels)
            axis_padding = label_width + 1
            normalized = [round((speed / max_speed) * height) for speed in points]
            padded = [0] * (width - len(normalized)) + normalized
            graph = Text()

            for row, label in zip(range(height, 0, -1), labels):
                graph.append(f"{label:>{label_width}} ", style="dim")
                graph.append("│", style="dim")
                for value in padded:
                    graph.append("█" if value >= row else " ", style="cyan")
                graph.append("\n")

            graph.append(" " * axis_padding, style="dim")
            graph.append("└", style="dim")
            graph.append("─" * width, style="dim")
            graph.append("\n")
            graph.append(" " * (axis_padding + 1), style="dim")
            graph.append("recent samples →", style="dim")
            return graph

        def _download_row(self, percent: float | None, progress_width: int, pending: bool, now: float) -> Text:
            row = Text("Downloaded ", style="bold")
            row.append(self._progress_line(percent, width=progress_width, pending=pending, now=now))

            if self.total is not None and percent is not None:
                row.append(
                    f" {percent * 100:.1f}% ({self.display_n / (1024 * 1024):.1f} / {self.total / (1024 * 1024):.1f} MB)"
                )
            else:
                row.append(f" {self.display_n / (1024 * 1024):.1f} MB")
            return row

        def _file_context_row(self) -> Text:
            if self.current_file is None and self.current_file_index is None and self.total_files is None:
                return Text("Preparing files...", style="dim")

            row = Text("File ", style="bold")
            if self.current_file_index is not None and self.total_files is not None:
                row.append(f"{self.current_file_index} / {self.total_files}")
            elif self.current_file_index is not None:
                row.append(str(self.current_file_index))
            else:
                row.append("?")

            if self.current_file:
                row.append(f"  {self.current_file}", style="cyan")

            return row

        def _render(self) -> Panel:
            if self.final_status == "failed":
                return Panel(self._summary_table(completed=False), title="Download failed", border_style="red")

            now = time.monotonic()
            self._advance_display(now)
            elapsed = max(now - self.start_time, 1e-9)
            waiting_on_current_file = self._is_waiting_on_current_file(now)
            has_meaningful_progress = self.n >= self._minimum_speed_sample_bytes
            average_speed_mb = self._average_speed_bytes(now) / (1024 * 1024) if has_meaningful_progress else 0.0
            instant_speed_mb = self.speed_samples_mb[-1] if self.speed_samples_mb else 0.0

            if self.total:
                percent = min(1.0, max(0.0, self.display_n / self.total))
                average_speed_bytes = average_speed_mb * 1024 * 1024
                remaining_seconds = (
                    max(self.total - self.display_n, 0) / average_speed_bytes if average_speed_bytes > 0 else None
                )
                suffix = f" {percent * 100:.1f}% ({self.display_n / (1024 * 1024):.1f} / {self.total / (1024 * 1024):.1f} MB)"
            else:
                percent = None
                remaining_seconds = None
                suffix = f" {self.display_n / (1024 * 1024):.1f} MB"

            content_width = max(40, self.console.width - 4)
            progress_width = max(10, content_width - len("Downloaded ") - len(suffix))
            graph_axis_width = len(f"{max(max(self.speed_samples_mb, default=0.1), 0.1):.2f} MB/s") + 2
            graph_width = max(20, content_width - graph_axis_width)

            table = Table.grid(expand=True)
            table.add_column(ratio=1)
            table.add_column(justify="right")
            table.add_row("Current speed", f"{instant_speed_mb:.2f} MB/s")
            table.add_row("Average speed", f"{average_speed_mb:.2f} MB/s")
            table.add_row("Elapsed", f"{elapsed:.1f}s")
            table.add_row("ETA", self._format_duration(remaining_seconds))

            title = "Live download"
            if self.display_name:
                title = f"{title} - {self.display_name}"

            return Panel(
                Group(
                    self._file_context_row(),
                    self._download_row(percent, progress_width, waiting_on_current_file, now),
                    table,
                    Text(""),
                    Text("Speed over time", style="bold"),
                    self._speed_graph(width=graph_width),
                ),
                title=title,
                border_style="blue",
            )

    return cast(type[base_tqdm], RichDownloadProgress)
