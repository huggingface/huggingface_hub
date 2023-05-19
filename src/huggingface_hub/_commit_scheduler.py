import abc
import atexit
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar
from uuid import uuid4

from .hf_api import HfApi


ItemT = TypeVar("ItemT")
ReturnT = TypeVar("ReturnT")

# TODO: add logging everywhere
# TODO: Start scheduler => sleep X seconds => flush
# TODO: cancel + restart scheduler on each new item
# TODO: store futures in class? How to deal with exceptions?


class CommitScheduler(abc.ABC, Generic[ItemT, ReturnT]):
    def __init__(
        self,
        repo_id: str,
        *,
        repo_type: Optional[str],
        revision: Optional[str] = None,
        token: Optional[str] = None,
        commit_after_n_items: int = 100,
        max_items_per_commit: int = 500,
        commit_after_n_seconds: int = 30 * 60,
        max_seconds_in_queue: int = 120 * 60,
        api: Optional["HfApi"] = None,
    ) -> None:
        """
        Scheduler class to commit items to the Hub in batches.

        Args:
            repo_id (`str`):
                The id of the repo to commit to.
            repo_type (`str`, `optional`):
                The type of the repo to commit to. Defaults to `model`.
            revision (`str`, `optional`):
                The revision of the repo to commit to. Defaults to `main`.
            token (`str`, `optional`):
                The token to use to commit to the repo. Defaults to the token saved on the machine.
            commit_after_n_items (`int`, `optional`):
                The number of items to trigger a commit. If the number of items in the queue is over this threshold,
                the scheduler triggers a commit no matter the timing.  Can be set to -1 if you don't want to schedule
                commits based on the number of items. Defaults to 100.
            max_items_per_commit (`int`, `optional`):
                The maximum number of items in a single commit. If the number of items in the queue is over this
                threshold, commits will be chunked. Can be set to -1 if you don't want to limit the number of items in
                a single commit. Defaults to 500.
            commit_after_n_seconds (`int`, `optional`):
                The number of seconds to wait after the last item has been queued before triggering a commit. Can be
                set to -1 if you don't want to schedule commits based on the timing. Defaults to 30 minutes.
            max_seconds_in_queue (`int`, `optional`):
                The maximum number of seconds an item can stay in the queue before being committed. Past this deadline,
                a commit is triggered, no matter the number of items. Can be set to -1 if you don't want to limit the
                time an item can stay in the queue. Defaults to 2 hours.
            api (`HfApi`, `optional`):
                The API client to use to commit to the Hub. Can be set with custom settings (user agent, token,...).
        """
        # Scheduler params
        _check_positive_or_minus_one("commit_after_n_items", commit_after_n_items)
        _check_positive_or_minus_one("max_items_per_commit", max_items_per_commit)
        _check_positive_or_minus_one("commit_after_n_seconds", commit_after_n_seconds)
        _check_positive_or_minus_one("max_seconds_in_queue", max_seconds_in_queue)

        self.commit_after_n_items = commit_after_n_items  # TODO
        self.max_items_per_commit = max_items_per_commit  # TODO
        self.commit_after_n_seconds = commit_after_n_seconds  # TODO
        self.max_seconds_in_queue = max_seconds_in_queue  # TODO

        # Commit-related
        self.api = api or HfApi()
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revision = revision
        self.token = token

        # Internals
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._callbacks: List[Callable[[List[ItemT], ReturnT, Optional[Exception]], None]] = []
        self._items: List[ItemT] = []

        # On last resort, flush at the end of the script
        atexit.register(self.flush)

    def add_item(self, item: ItemT) -> None:
        self.add_items([item])

    def add_items(self, items: List[ItemT]) -> None:
        with self._lock:
            self._items.extend(items)
            # TODO: handle flush nicely (schedule + nb items per commit)
            if self.max_items_per_commit > 0 and len(self._items) >= self.max_items_per_commit:
                self.flush()

    def register_callback(self, callback: Callable[[List[ItemT], ReturnT, Optional[Exception]], None]) -> None:
        self._callbacks.append(callback)

    def flush(self) -> None:
        with self._lock:
            self._pool.submit(self._flush, self._items)
            self._items = []

    def _flush(self, items: List[ItemT]) -> ReturnT:
        exception = None
        try:
            output = self._push_to_hub(items)
        except Exception as e:
            exception = e

        for callback in self._callbacks:
            try:
                callback(items, output, exception)
            except Exception:
                pass

        if exception is not None:
            raise exception
        return output

    @abc.abstractmethod
    def _push_to_hub(self, items: List[ItemT]) -> ReturnT:
        ...


class JsonlCommitScheduler(CommitScheduler[Dict, str]):
    def _push_to_hub(self, items: List[Dict]) -> Any:
        buffer = StringIO()
        for item in items:
            buffer.write(json.dumps(item))
            buffer.write("\n")

        filename = f"data-{uuid4()}.jsonl"
        self.api.upload_file(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            path_in_repo=filename,
            revision=self.revision,
            token=self.token,
            path_or_fileobj=buffer.getvalue().encode(),
            commit_message="Upload JSONL file using JsonCommitScheduler",
        )
        return filename


scheduler = JsonlCommitScheduler("huggingface/feedback-data", repo_type="dataset")
scheduler.register_callback(lambda items, output, exception: print(f"Uploaded {len(items)} items to {output}"))
scheduler.add_item({"a": 1})


def _check_positive_or_minus_one(name: str, value: int) -> None:
    if not (value > 0 or value == -1):
        raise ValueError(f"'{name}' must be a positive integer or -1, not {value}.")
