from typing import List
from rich.progress import Task, Text, BarColumn, TextColumn, ProgressColumn, Progress, TaskID, DownloadColumn

try:
    from hf_xet import PyItemProgressUpdate, PyTotalProgressUpdate
except ImportError as e:
    raise ImportError(f"The current version of hf_xet does not yet support detailed upload progress; please upgrade ({e})")


# Pretty print bytes out.
def format_bytes(v: float) -> str:
    """Human-readable formatting of speed"""
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if v < 1024:
            return f"{v:>6.2f} {unit}"
        v /= 1024
    return f"{v:>6.2f} TB"

class PlainBarColumn(BarColumn):
    def __init__(self, bar_width: int = 40, fill_char: str = "=", empty_char: str = " "):
        super().__init__(bar_width=None)  # We'll use our own width
        self.fixed_width = bar_width
        self.fill_char = fill_char
        self.empty_char = empty_char

    def render(self, task: Task) -> Text:
        total = task.total
        completed = task.completed

        width = self.fixed_width or 40
        if total is None: 
            return Text(f"[{self.empty_char * width}]")
        if total == 0:
            return Text(f"[{self.fill_char * width}]")

        ratio = min(max(completed / total, 0.0), 1.0)
        filled_width = int(ratio * width)
        empty_width = width - filled_width

        bar_str = f"[{self.fill_char * filled_width}{self.empty_char * empty_width}]"
        return Text(bar_str)

class PlainTransferSpeedColumn(ProgressColumn):
    """Column showing total average speed: total bytes / total time"""
    def render(self, task: Task) -> Text:
        if task.total is None or task.start_time is None:
            return Text("---")
        elapsed = task.finished_time or task.elapsed
        if not elapsed or elapsed <= 0.0:
            return Text("0.0 B/s")

        speed = task.completed / elapsed
        return Text(f"{format_bytes(speed)}/s")
    

class PlainPercentageColumn(TextColumn): 
    """
    Column showing total completion percentage.
    """

    def render(self, task: Task) -> Text:
        if task.total is None or task.start_time is None:
            return Text("---")
        elif task.total == 0:
            return Text("100%")
        else:
            return Text(f"{ (100. * task.completed) / task.total:>3.1f}%")


class PlainDownloadColumn(DownloadColumn):
    """Black-and-white version of DownloadColumn that shows completed / total."""
    def render(self, task: Task) -> Text:
        completed = task.completed
        total = task.total
        if total is None:
            return Text(format_bytes(completed))
        
        return Text(f"{format_bytes(completed)} / {format_bytes(total)}")
    

class XetProgressTracker:

    def __init__(self, target_num_rows: int = 10, max_rows : int = 20):
        """
        Progress tracker to take themed updates. 

        A lot of the fields have been customized to more closely match the tqdm style updates; more possible.
        """
        self.progress = Progress(
            TextColumn("{task.description}"),
            PlainBarColumn(bar_width = 40),
            TextColumn("{task.percentage:>3.0f}%"),
            "•",
            PlainDownloadColumn(),
            "•",
            PlainTransferSpeedColumn(),
        )

        self.target_num_rows = target_num_rows
        self.max_rows = max_rows
        self.num_active_tasks = 0
        self.active = True

        # mapping file_name → Task
        self.tasks: dict[str, TaskID] = {}

        # running totals
        self.total_bytes = 0
        self.total_transfer_bytes = 0
        self.bytes_processed_tid = self.progress.add_task("TOTAL", total=None)
        self.upload_progress_tid = self.progress.add_task("  <NEW DATA>", total=None)

        # Track completed tasks for eviction as needed.
        self.completed_tasks = set()

        self.progress.start()

    def update_progress(self, total_update : PyTotalProgressUpdate, item_updates: List[PyItemProgressUpdate]):
        if not self.active:
            return

        # Update the two totals, as we track them here   
        # bump totals so the bars’ "total=" is correct
        self.total_bytes += total_update.total_bytes_increment
        self.total_transfer_bytes += total_update.total_transfer_bytes_increment

        self.progress.update(
            self.bytes_processed_tid,
            total=self.total_bytes,
            advance=total_update.total_bytes_completion_increment
        )
        self.progress.update(
            self.upload_progress_tid,
            total=self.total_transfer_bytes,
            advance=total_update.total_transfer_bytes_completion_increment
        )
        
        # Evict the oldest *finished* bars down to max_rows.
        if self.num_active_tasks > self.target_num_rows:
            for tid in self.progress.task_ids: 
                if tid in self.completed_tasks:  # Skip the two global progresses
                    self.progress.remove_task(tid)
                    self.num_active_tasks -= 1

                    if self.num_active_tasks <= self.target_num_rows:
                        break

        # Update per-file progress
        for it in item_updates:
            name = it.item_name

            # only start tracking once there's some progress and if we have space to show it; otherwise, wait 
            # for some of the rows to complete and clear out first.  Because the information is tracked in the bytes_completed, 
            # we won't lose anything here -- when the task comes in, the progress will still be correct.
            if name not in self.tasks and it.bytes_completed != 0 and self.num_active_tasks < self.max_rows:
                tid = self.tasks[name] = self.progress.add_task("  " + name, completed=it.bytes_completed, total=it.total_bytes)
                self.num_active_tasks += 1
            else:
                tid = self.tasks.get(name)
                if tid:
                    self.progress.advance(tid, it.bytes_completion_increment)

            if it.bytes_completed == it.total_bytes: 
                if tid:
                    self.completed_tasks.add(tid)



    def close(self, success: bool):
        """ 
        Finish the bar down; no further updates will be processed. 

        On success, marks everything as completed.
        """
        if success:
            # Set these so the total is always a number instead of None; when completed, it means that it will show it at 100% instead of 0% 
            self.progress.update(self.bytes_processed_tid, total=self.total_bytes,completed=self.total_bytes, refresh=True)
            self.progress.update(self.upload_progress_tid, total=self.total_transfer_bytes, completed=self.total_transfer_bytes, refresh=True)

            # force all remaining file bars to complete
            tasks = self.progress.tasks
            for task in tasks: 
                if not task.finished:
                    self.progress.update(task.id, completed=task.total)

        # stop rendering (bars remain on screen)
        self.progress.stop()
        self.active = False


__all__ = [
    "XetProgressTracker",
]