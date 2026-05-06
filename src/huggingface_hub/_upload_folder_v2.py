# Copyright 2025-present, the HuggingFace Inc. team.
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
"""Robust, resumable, multi-worker folder upload.

Architecture
~~~~~~~~~~~~
Four dedicated stage threads connected by queues::

    hash_stage → mode_stage → preupload_stage → commit_stage
       (pool)      (batch)        (pool)          (batch)

*  `hash_stage` — hashes files in a :class:`ThreadPoolExecutor`.
*  `mode_stage` — batch-fetches upload mode (xet vs regular) from the Hub,
   routes items to either `preupload_stage` or straight to `commit_stage`.
*  `preupload_stage` — uploads xet blobs via a shared :class:`_XetUploader`.
*  `commit_stage` — collects ready items and commits in adaptive batches.

Each stage receives items from its upstream queue and pushes results to its downstream queue.  A `_SENTINEL` marker
signals "no more items from this producer", giving clean, race-free termination.

Resumability is achieved by persisting per-file metadata after every step (`<folder>/.cache/huggingface/upload/`).
On restart, items skip already-completed stages.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

from ._commit_api import (
    CommitOperationAdd,
    CommitOperationDelete,
    UploadInfo,
    UploadMode,
    _fetch_upload_modes,
    _validate_path_in_repo,
)
from ._local_folder import LocalUploadFileMetadata, LocalUploadFilePaths, get_local_upload_paths, read_upload_metadata
from .constants import DEFAULT_REVISION, REPO_TYPE_MODEL, REPO_TYPES
from .utils import DEFAULT_IGNORE_PATTERNS, filter_repo_objects, tqdm
from .utils._runtime import is_xet_available
from .utils._xet import XetTokenType, fetch_xet_connection_info_from_repo_info
from .utils.sha import sha_fileobj


if TYPE_CHECKING:
    from .hf_api import HfApi

logger = logging.getLogger(__name__)

# Queue sentinel — signals "this producer is done".
_SENTINEL = object()

# Batch sizes
_PREUPLOAD_BATCH = 256
_FETCH_MODE_BATCH = 100

# Commit pacing
_COMMIT_INTERVAL = 5 * 60  # force a commit after this many seconds of silence
_COMMIT_RAMP = [20, 50, 75, 100, 125, 200, 250, 400, 600, 1000]

# Display
_BAR_WIDTH = 20
_STATUS_INTERVAL = 30  # non-TTY fallback


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def upload_folder_v2(
    api: "HfApi",
    *,
    repo_id: str,
    folder_path: str | Path,
    path_in_repo: str | None = None,
    commit_message: str | None = None,
    commit_description: str | None = None,
    token: str | bool | None = None,
    repo_type: str | None = None,
    revision: str | None = None,
    create_pr: bool | None = None,
    parent_commit: str | None = None,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    delete_patterns: list[str] | str | None = None,
    num_workers: int | None = None,
):
    """Upload a local folder to a Hub repo — robust, resumable, multi-worker.

    Same API as ``HfApi.upload_folder`` plus ``num_workers``.
    Requires Xet storage (``hf_xet`` package).
    """
    # -- Xet is mandatory -------------------------------------------------------
    if not is_xet_available():
        raise EnvironmentError(
            "upload_folder_v2 requires Xet storage. Install `hf_xet`: `pip install huggingface_hub[hf_xet]`"
        )

    # -- Validate & defaults ---------------------------------------------------
    if repo_type is None:
        repo_type = REPO_TYPE_MODEL
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
    if revision is None:
        revision = DEFAULT_REVISION
    create_pr = create_pr or False

    folder_path = Path(folder_path).expanduser().resolve()
    if not folder_path.is_dir():
        raise ValueError(f"Provided path: '{folder_path}' is not a directory")

    if path_in_repo is None:
        path_in_repo = ""
    prefix = f"{path_in_repo.strip('/')}/" if path_in_repo else ""

    if ignore_patterns is None:
        ignore_patterns = []
    elif isinstance(ignore_patterns, str):
        ignore_patterns = [ignore_patterns]
    ignore_patterns = ignore_patterns + DEFAULT_IGNORE_PATTERNS

    if num_workers is None:
        nb_cores = os.cpu_count() or 1
        num_workers = max(nb_cores // 2, 1)

    commit_message = commit_message or "Upload folder using huggingface_hub"

    # -- Create PR upfront if requested ----------------------------------------
    pr_url: str | None = None
    target_revision = revision
    if create_pr:
        pr = api.create_pull_request(
            repo_id=repo_id,
            title=commit_message,
            description=commit_description,
            repo_type=repo_type,
            token=token,
        )
        pr_url = pr.url
        target_revision = pr.git_reference
        logger.info(f"Created PR: {pr_url} (revision: {target_revision})")

    # -- Prepare delete operations ---------------------------------------------
    delete_operations = api._prepare_folder_deletions(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=DEFAULT_REVISION if create_pr else revision,
        token=token,
        path_in_repo=path_in_repo,
        delete_patterns=delete_patterns,
    )

    # -- List & read cached metadata -------------------------------------------
    filtered_relpaths = list(
        filter_repo_objects(
            (p.relative_to(folder_path).as_posix() for p in sorted(folder_path.glob("**/*")) if p.is_file()),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    )
    logger.info(f"Found {len(filtered_relpaths)} files to upload")
    if not filtered_relpaths and not delete_operations:
        logger.warning("Nothing to upload or delete.")
        return None

    # Delete-only shortcut: no need for the full pipeline.
    if not filtered_relpaths:
        return api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=target_revision,
            operations=delete_operations,
            commit_message=commit_message,
            commit_description=commit_description,
            parent_commit=parent_commit,
        )

    items: list[_UploadItem] = []
    for relpath in tqdm(filtered_relpaths, desc="Reading cached metadata"):
        repo_path = prefix + relpath
        paths = get_local_upload_paths(folder_path, repo_path)
        metadata = read_upload_metadata(folder_path, repo_path)
        items.append(_UploadItem(paths=paths, metadata=metadata))

    # -- Categorize by current progress (for resumability) ---------------------
    need_hash: list[_UploadItem] = []
    need_mode: list[_UploadItem] = []
    need_preupload: list[_UploadItem] = []
    need_commit: list[_UploadItem] = []
    for item in items:
        m = item.metadata
        if m.sha256 is None:
            need_hash.append(item)
        elif m.upload_mode is None:
            need_mode.append(item)
        elif m.upload_mode == "lfs" and not m.is_uploaded:
            need_preupload.append(item)
        elif not m.is_committed:
            need_commit.append(item)

    already_done = len(items) - len(need_hash) - len(need_mode) - len(need_preupload) - len(need_commit)
    logger.info(
        f"Pipeline: {len(need_hash)} to hash, {len(need_mode)} to check mode, "
        f"{len(need_preupload)} to preupload, {len(need_commit)} to commit, "
        f"{already_done} already done"
    )

    # All files already committed — only deletes remain (if any).
    if all(it.metadata.is_committed or it.metadata.should_ignore for it in items):
        if delete_operations:
            return api.create_commit(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=target_revision,
                operations=delete_operations,
                commit_message=commit_message,
                commit_description=commit_description,
                parent_commit=parent_commit,
            )
        logger.info("All files already uploaded and committed.")
        return None

    # -- Setup inter-stage queues (pre-populate from resumed state) ------------
    q_mode: queue.Queue[_UploadItem | object] = queue.Queue()
    q_preupload: queue.Queue[_UploadItem | object] = queue.Queue()
    q_commit: queue.Queue[_UploadItem | object] = queue.Queue()

    for it in need_mode:
        q_mode.put(it)
    for it in need_preupload:
        q_preupload.put(it)
    for it in need_commit:
        q_commit.put(it)

    # -- Shared abort signal & error collection --------------------------------
    abort = threading.Event()
    errors: list[BaseException] = []
    commit_result: dict[str, Any] = {"last_commit_info": None}

    headers = api._build_hf_headers(token=token)

    # -- Live display (3 lines: preparing / uploading / committing) ------------
    display = _LiveDisplay(items)

    # -- Shared xet uploader ---------------------------------------------------
    xet_uploader = _XetUploader(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=target_revision,
        headers=headers,
        endpoint=api.endpoint,
        display=display,
    )

    # -- Launch the four stage threads -----------------------------------------
    stages = [
        threading.Thread(
            target=_hash_stage,
            daemon=True,
            kwargs=dict(items=need_hash, q_out=q_mode, num_workers=num_workers, abort=abort, errors=errors),
        ),
        threading.Thread(
            target=_mode_stage,
            daemon=True,
            kwargs=dict(
                q_in=q_mode,
                q_lfs=q_preupload,
                q_ready=q_commit,
                api=api,
                repo_id=repo_id,
                repo_type=repo_type,
                revision=target_revision,
                headers=headers,
                abort=abort,
                errors=errors,
            ),
        ),
        threading.Thread(
            target=_preupload_stage,
            daemon=True,
            kwargs=dict(
                q_in=q_preupload,
                q_out=q_commit,
                xet_uploader=xet_uploader,
                num_workers=num_workers,
                abort=abort,
                errors=errors,
            ),
        ),
        threading.Thread(
            target=_commit_stage,
            daemon=True,
            kwargs=dict(
                q_in=q_commit,
                api=api,
                repo_id=repo_id,
                repo_type=repo_type,
                revision=target_revision,
                commit_message=commit_message,
                commit_description=commit_description,
                parent_commit=parent_commit,
                delete_operations=delete_operations,
                result=commit_result,
                display=display,
                abort=abort,
                errors=errors,
            ),
        ),
    ]
    for t in stages:
        t.start()

    # -- Main thread: refresh display ------------------------------------------
    while any(t.is_alive() for t in stages):
        time.sleep(0.5)
        if abort.is_set():
            break
        display.refresh()

    for t in stages:
        t.join(timeout=10)

    display.close()

    if errors:
        raise errors[0]

    logger.info("Upload complete. %s", _summary(items))

    # -- Patch PR info onto the last commit ------------------------------------
    last_commit = commit_result["last_commit_info"]
    if last_commit is not None and pr_url is not None:
        last_commit.pr_url = pr_url
        last_commit.pr_revision = target_revision
        last_commit.pr_num = int(target_revision.split("/")[-1]) if target_revision else None
    return last_commit


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class _UploadItem:
    paths: LocalUploadFilePaths
    metadata: LocalUploadFileMetadata


class _CommitPacer:
    """Grows/shrinks commit batch size based on success and duration."""

    def __init__(self) -> None:
        self._idx = 1

    @property
    def target(self) -> int:
        return _COMMIT_RAMP[self._idx]

    def report(self, success: bool, n_items: int, duration: float) -> None:
        if not success:
            self._idx = max(0, self._idx - 1)
        elif n_items >= _COMMIT_RAMP[self._idx] and duration < 40:
            self._idx = min(len(_COMMIT_RAMP) - 1, self._idx + 1)


# ---------------------------------------------------------------------------
# Three-line live display
# ---------------------------------------------------------------------------


class _LiveDisplay:
    """Three-line live progress display on stderr::

        Preparing   ████████████████████  11,100 / 11,100 ✓
        Uploading   ██████████████░░░░░░  3.8GB / 8.2GB  19.7MB/s · 4.7 files/s
        Committing  ██████████████████░░  10,800 / 11,100  14 commits

    On a TTY the three lines are rewritten in-place every ~0.5 s (driven by
    the main thread).  The xet callback thread only writes to shared counters;
    the main thread does all rendering.

    When stderr is not a TTY, falls back to periodic ``logger.info`` calls.
    """

    _N_LINES = 3

    def __init__(self, items: list[_UploadItem]) -> None:
        self._tty = sys.stderr.isatty()
        self._items = items
        self._lock = threading.Lock()
        self._drawn = False
        self._last_fallback = 0.0
        self._commit_count = 0

        # Xet progress — written by callback, read by renderer
        self._xet_bytes = 0
        self._xet_transfer_total = 0
        self._xet_offset = 0
        self._xet_speed_ema: float = 0
        self._xet_start: float | None = None
        self._xet_completed: set[str] = set()

    # -- called from xet callback thread ---------------------------------------

    def update_xet(self, total_update: Any, item_updates: Any) -> None:
        with self._lock:
            if self._xet_start is None:
                self._xet_start = time.time()
            self._xet_bytes += total_update.total_transfer_bytes_completion_increment
            self._xet_transfer_total = self._xet_offset + total_update.total_transfer_bytes
            if total_update.total_transfer_bytes_completion_rate is not None:
                rate = total_update.total_transfer_bytes_completion_rate
                if self._xet_speed_ema == 0:
                    self._xet_speed_ema = rate
                else:
                    self._xet_speed_ema = 0.1 * rate + 0.9 * self._xet_speed_ema
            for item in item_updates:
                if item.bytes_completed == item.total_bytes and item.total_bytes > 0:
                    self._xet_completed.add(item.item_name)

    def notify_batch_done(self) -> None:
        with self._lock:
            self._xet_offset = self._xet_transfer_total

    # -- called from commit thread ---------------------------------------------

    def notify_commit(self) -> None:
        with self._lock:
            self._commit_count += 1

    # -- called from main thread -----------------------------------------------

    def refresh(self) -> None:
        with self._lock:
            if self._tty:
                self._redraw()
            else:
                now = time.time()
                if now - self._last_fallback >= _STATUS_INTERVAL:
                    logger.info(_summary(self._items))
                    self._last_fallback = now

    def close(self) -> None:
        with self._lock:
            if self._tty and self._drawn:
                self._redraw()
            self._tty = False

    # -- internals -------------------------------------------------------------

    def _redraw(self) -> None:
        if self._drawn:
            sys.stderr.write(f"\033[{self._N_LINES}A")
        self._redraw_lines()

    def _redraw_lines(self) -> None:
        lines = [self._line_preparing(), self._line_uploading(), self._line_committing()]
        width = shutil.get_terminal_size().columns
        for line in lines:
            truncated = line[: width - 4] + "..." if len(line) > width - 1 else line
            sys.stderr.write(f"\r\033[K{truncated}\n")
        sys.stderr.flush()
        self._drawn = True

    def _line_preparing(self) -> str:
        total = len(self._items)
        prepared = sum(1 for it in self._items if it.metadata.upload_mode is not None or it.metadata.should_ignore)
        bar = _bar(prepared, total)
        done = " ✓" if prepared >= total else ""
        return f"  Preparing   {bar}  {prepared:,} / {total:,}{done}"

    def _line_uploading(self) -> str:
        n_xet = sum(1 for it in self._items if it.metadata.upload_mode == "lfs" and not it.metadata.should_ignore)
        all_classified = all(it.metadata.upload_mode is not None or it.metadata.should_ignore for it in self._items)

        if n_xet == 0:
            if all_classified:
                return f"  Uploading   {_bar(1, 1)}  - ✓"
            return f"  Uploading   {_bar(0, 1)}  -"

        xet_uploaded = sum(
            1
            for it in self._items
            if it.metadata.upload_mode == "lfs" and not it.metadata.should_ignore and it.metadata.is_uploaded
        )
        bar = _bar(xet_uploaded, n_xet)

        parts: list[str] = []
        if self._xet_bytes > 0:
            parts.append(_format_bytes(self._xet_bytes))
        if self._xet_speed_ema > 0:
            parts.append(f"{_format_bytes(self._xet_speed_ema)}/s")
        extra = f"  {' · '.join(parts)}" if parts else ""

        done = " ✓" if xet_uploaded >= n_xet else ""

        return f"  Uploading   {bar}  {xet_uploaded:,} / {n_xet:,} files{extra}{done}"

    def _line_committing(self) -> str:
        total = len(self._items)
        ignored = sum(1 for it in self._items if it.metadata.should_ignore)
        effective = total - ignored
        committed = sum(1 for it in self._items if it.metadata.is_committed)
        bar = _bar(committed, effective) if effective > 0 else _bar(0, 1)
        commits_str = f"  {self._commit_count} commits" if self._commit_count > 1 else ""
        done = " ✓" if committed >= effective > 0 else ""
        return f"  Committing  {bar}  {committed:,} / {effective:,}{commits_str}{done}"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _bar(current: float, total: float, width: int = _BAR_WIDTH) -> str:
    if total <= 0:
        return "░" * width
    ratio = min(current / total, 1.0)
    filled = int(ratio * width)
    return "█" * filled + "░" * (width - filled)


def _format_bytes(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(n) < 1000:
            if n < 10:
                return f"{n:.2f}{unit}"
            elif n < 100:
                return f"{n:.1f}{unit}"
            return f"{n:.0f}{unit}"
        n /= 1000
    return f"{n:.1f}PB"


def _summary(items: list[_UploadItem]) -> str:
    total = len(items)
    committed = sum(1 for it in items if it.metadata.is_committed)
    ignored = sum(1 for it in items if it.metadata.should_ignore)
    n_regular = sum(1 for it in items if it.metadata.upload_mode == "regular")
    n_xet = sum(1 for it in items if it.metadata.upload_mode == "lfs" and not it.metadata.should_ignore)
    return f"{committed}/{total} files committed ({n_regular} regular, {n_xet} xet, {ignored} ignored)"


# ---------------------------------------------------------------------------
# Xet uploader
# ---------------------------------------------------------------------------


class _XetUploader:
    """Wraps ``hf_xet.upload_files`` with progress fed into a :class:`_LiveDisplay`."""

    def __init__(
        self,
        repo_id: str,
        repo_type: str,
        revision: str | None,
        headers: dict[str, str],
        endpoint: str | None,
        display: _LiveDisplay,
    ) -> None:
        self._repo_id = repo_id
        self._repo_type = repo_type
        self._revision = revision
        self._headers = headers
        self._endpoint = endpoint
        self._display = display

        self._lock = threading.Lock()
        self._connection_info: Any | None = None

    def _ensure_connection(self) -> None:
        with self._lock:
            if self._connection_info is not None:
                return
        from .errors import HfHubHTTPError, XetAuthorizationError

        try:
            info = fetch_xet_connection_info_from_repo_info(
                token_type=XetTokenType.WRITE,
                repo_id=self._repo_id,
                repo_type=self._repo_type,
                revision=self._revision,
                headers=self._headers,
                endpoint=self._endpoint,
            )
        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                raise XetAuthorizationError(
                    f"Unauthorized to upload to xet storage for {self._repo_type}/{self._repo_id}. "
                    "Check that your access token has write access."
                ) from e
            raise
        with self._lock:
            self._connection_info = info

    def _token_refresher(self) -> tuple[str, int]:
        from .errors import XetRefreshTokenError

        new_info = fetch_xet_connection_info_from_repo_info(
            token_type=XetTokenType.WRITE,
            repo_id=self._repo_id,
            repo_type=self._repo_type,
            revision=self._revision,
            headers=self._headers,
            endpoint=self._endpoint,
        )
        if new_info is None:
            raise XetRefreshTokenError("Failed to refresh xet token")
        return new_info.access_token, new_info.expiration_unix_epoch

    def upload_batch(self, items: list[_UploadItem]) -> None:
        from hf_xet import upload_files

        self._ensure_connection()
        with self._lock:
            conn = self._connection_info
        assert conn is not None

        additions = [_to_hacky_add(item) for item in items]
        paths = [str(op.path_or_fileobj) for op in additions if isinstance(op.path_or_fileobj, (str, Path))]
        sha256s = [op.upload_info.sha256.hex() for op in additions if isinstance(op.path_or_fileobj, (str, Path))]

        if not paths:
            return

        xet_headers = self._headers.copy()
        xet_headers.pop("authorization", None)

        def callback(total_update, item_updates):
            self._display.update_xet(total_update, item_updates)

        upload_files(
            paths,
            conn.endpoint,
            (conn.access_token, conn.expiration_unix_epoch),
            lambda: self._token_refresher(),
            callback,
            self._repo_type,
            request_headers=xet_headers,
            sha256s=sha256s,
        )

        self._display.notify_batch_done()

        for item in items:
            item.metadata.is_uploaded = True
            item.metadata.save(item.paths)


# ---------------------------------------------------------------------------
# Stage threads
# ---------------------------------------------------------------------------


def _hash_stage(
    items: list[_UploadItem],
    q_out: queue.Queue[_UploadItem | object],
    num_workers: int,
    abort: threading.Event,
    errors: list[BaseException],
) -> None:
    """Hash files in parallel, push each into *q_out* as soon as it's ready."""
    try:
        if not items:
            return

        def hash_one(item: _UploadItem) -> None:
            if abort.is_set():
                return
            _do_sha256(item)
            q_out.put(item)

        with ThreadPoolExecutor(num_workers) as pool:
            futs = [pool.submit(hash_one, it) for it in items]
            for fut in as_completed(futs):
                if abort.is_set():
                    break
                fut.result()
    except Exception as e:
        logger.error(f"Hash stage failed: {e}")
        errors.append(e)
        abort.set()
    finally:
        q_out.put(_SENTINEL)


def _mode_stage(
    q_in: queue.Queue[_UploadItem | object],
    q_lfs: queue.Queue[_UploadItem | object],
    q_ready: queue.Queue[_UploadItem | object],
    api: "HfApi",
    repo_id: str,
    repo_type: str,
    revision: str,
    headers: dict[str, str],
    abort: threading.Event,
    errors: list[BaseException],
) -> None:
    """Batch-fetch upload mode, route items to xet preupload or straight to commit."""
    try:
        while not abort.is_set():
            batch = _collect_batch(q_in, _FETCH_MODE_BATCH)
            if batch is None:
                break
            if not batch:
                continue
            _do_fetch_upload_mode(
                batch, api=api, repo_id=repo_id, repo_type=repo_type, revision=revision, headers=headers
            )
            for item in batch:
                if item.metadata.should_ignore:
                    continue
                if item.metadata.upload_mode == "lfs":
                    q_lfs.put(item)
                else:
                    q_ready.put(item)
    except Exception as e:
        logger.error(f"Mode stage failed: {e}")
        errors.append(e)
        abort.set()
    finally:
        q_lfs.put(_SENTINEL)
        q_ready.put(_SENTINEL)


def _preupload_stage(
    q_in: queue.Queue[_UploadItem | object],
    q_out: queue.Queue[_UploadItem | object],
    xet_uploader: _XetUploader,
    num_workers: int,
    abort: threading.Event,
    errors: list[BaseException],
) -> None:
    """Upload xet blobs via the shared uploader, push items into *q_out*."""
    try:
        with ThreadPoolExecutor(num_workers) as pool:
            pending: set[Any] = set()

            while not abort.is_set():
                batch = _collect_batch(q_in, _PREUPLOAD_BATCH, timeout=0.5)
                if batch is None:
                    break
                if batch:
                    fut = pool.submit(_preupload_and_enqueue, batch, q_out, xet_uploader)
                    pending.add(fut)

                done = {f for f in pending if f.done()}
                for f in done:
                    pending.discard(f)
                    f.result()

            for fut in as_completed(pending):
                if abort.is_set():
                    break
                fut.result()
    except Exception as e:
        logger.error(f"Preupload stage failed: {e}")
        errors.append(e)
        abort.set()
    finally:
        q_out.put(_SENTINEL)


def _commit_stage(
    q_in: queue.Queue[_UploadItem | object],
    api: "HfApi",
    repo_id: str,
    repo_type: str,
    revision: str,
    commit_message: str,
    commit_description: str | None,
    parent_commit: str | None,
    delete_operations: list[CommitOperationDelete],
    result: dict[str, Any],
    display: _LiveDisplay,
    abort: threading.Event,
    errors: list[BaseException],
) -> None:
    """Collect ready items and commit in adaptive batches.

    Receives items from two producers (mode_stage for regular files,
    preupload_stage for xet files) and waits for two ``_SENTINEL`` markers
    before finishing.
    """
    n_producers = 2
    try:
        pacer = _CommitPacer()
        batch: list[_UploadItem] = []
        sentinels = 0
        commit_count = 0
        last_commit_ts = time.time()
        pending_deletes = list(delete_operations)

        while sentinels < n_producers and not abort.is_set():
            want = max(1, pacer.target - len(batch))
            new_items = _collect_batch(q_in, want)
            if new_items is None:
                sentinels += 1
                continue
            batch.extend(new_items)

            now = time.time()
            should_commit = len(batch) >= pacer.target or (now - last_commit_ts > _COMMIT_INTERVAL and batch)
            if should_commit:
                chunk = batch[: pacer.target]
                batch = batch[pacer.target :]
                _do_commit(
                    chunk,
                    api=api,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                    commit_message=commit_message,
                    commit_description=commit_description,
                    parent_commit=parent_commit,
                    pending_deletes=pending_deletes,
                    commit_count=commit_count,
                    pacer=pacer,
                    result=result,
                    display=display,
                )
                commit_count += 1
                last_commit_ts = time.time()

        # Flush remaining items.
        while batch and not abort.is_set():
            chunk = batch[: pacer.target]
            batch = batch[pacer.target :]
            _do_commit(
                chunk,
                api=api,
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                commit_message=commit_message,
                commit_description=commit_description,
                parent_commit=parent_commit,
                pending_deletes=pending_deletes,
                commit_count=commit_count,
                pacer=pacer,
                result=result,
                display=display,
            )
            commit_count += 1
    except Exception as e:
        logger.error(f"Commit stage failed: {e}")
        errors.append(e)
        abort.set()


# ---------------------------------------------------------------------------
# Atomic operations
# ---------------------------------------------------------------------------


def _do_sha256(item: _UploadItem) -> None:
    if item.metadata.sha256 is not None:
        return
    with item.paths.file_path.open("rb") as f:
        item.metadata.sha256 = sha_fileobj(f).hex()
    item.metadata.save(item.paths)


def _do_fetch_upload_mode(
    items: list[_UploadItem],
    api: "HfApi",
    repo_id: str,
    repo_type: str,
    revision: str,
    headers: dict[str, str],
) -> None:
    additions = [_to_hacky_add(item) for item in items]
    _fetch_upload_modes(
        additions=additions,
        repo_type=repo_type,
        repo_id=repo_id,
        headers=headers,
        revision=quote(revision, safe=""),
        endpoint=api.endpoint,
    )
    for item, add_op in zip(items, additions):
        item.metadata.upload_mode = add_op._upload_mode
        item.metadata.should_ignore = add_op._should_ignore
        item.metadata.remote_oid = add_op._remote_oid
        item.metadata.save(item.paths)


def _preupload_and_enqueue(
    items: list[_UploadItem],
    q_out: queue.Queue[_UploadItem | object],
    xet_uploader: _XetUploader,
) -> None:
    """Upload a batch via xet and forward items to the commit queue."""
    xet_uploader.upload_batch(items)
    for item in items:
        q_out.put(item)


def _do_commit(
    items: list[_UploadItem],
    api: "HfApi",
    repo_id: str,
    repo_type: str,
    revision: str,
    commit_message: str,
    commit_description: str | None,
    parent_commit: str | None,
    pending_deletes: list[CommitOperationDelete],
    commit_count: int,
    pacer: _CommitPacer,
    result: dict[str, Any],
    display: _LiveDisplay,
) -> None:
    is_first = commit_count == 0
    msg = commit_message if is_first else f"{commit_message} ({commit_count + 1})"
    p_commit = parent_commit if is_first else None

    delete_ops: list[CommitOperationDelete] = []
    if is_first and pending_deletes:
        delete_ops = list(pending_deletes)

    additions = [_to_hacky_add(item) for item in items]
    operations: list[Any] = delete_ops + additions

    start = time.time()
    try:
        commit_info = api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            operations=operations,
            commit_message=msg,
            commit_description=commit_description,
            parent_commit=p_commit,
        )
        for item in items:
            item.metadata.is_committed = True
            item.metadata.save(item.paths)
        if is_first:
            pending_deletes.clear()
        result["last_commit_info"] = commit_info
        pacer.report(True, len(items), time.time() - start)
        display.notify_commit()
    except Exception:
        pacer.report(False, len(items), time.time() - start)
        raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _HackyCommitOperationAdd(CommitOperationAdd):
    """Bypasses the expensive ``__post_init__`` validation/hashing."""

    def __post_init__(self) -> None:
        self.path_in_repo = _validate_path_in_repo(self.path_in_repo)
        if isinstance(self.path_or_fileobj, Path):
            self.path_or_fileobj = str(self.path_or_fileobj)


def _to_hacky_add(item: _UploadItem) -> _HackyCommitOperationAdd:
    paths, meta = item.paths, item.metadata
    op = _HackyCommitOperationAdd(path_in_repo=paths.path_in_repo, path_or_fileobj=paths.file_path)

    with paths.file_path.open("rb") as f:
        sample = f.peek(512)[:512]

    if meta.sha256 is None:
        raise ValueError(f"sha256 not computed for {paths.path_in_repo}")

    op.upload_info = UploadInfo(sha256=bytes.fromhex(meta.sha256), size=meta.size, sample=sample)
    op._upload_mode = cast("UploadMode | None", meta.upload_mode)
    op._should_ignore = meta.should_ignore
    op._remote_oid = meta.remote_oid
    op._is_uploaded = meta.is_uploaded

    if meta.is_uploaded and meta.upload_mode == "lfs":
        op.path_or_fileobj = b""

    return op


def _collect_batch(
    q: queue.Queue[_UploadItem | object],
    max_size: int,
    timeout: float = 1.0,
) -> list[_UploadItem] | None:
    """Collect up to *max_size* items from *q*.

    Returns ``None`` when the sentinel is received (producer done).
    Returns ``[]`` on timeout (no items available yet).
    """
    try:
        first = q.get(timeout=timeout)
    except queue.Empty:
        return []

    if first is _SENTINEL:
        return None

    result: list[_UploadItem] = [cast(_UploadItem, first)]
    while len(result) < max_size:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        if item is _SENTINEL:
            q.put(_SENTINEL)  # put back so the next call sees it
            break
        result.append(cast(_UploadItem, item))

    return result
