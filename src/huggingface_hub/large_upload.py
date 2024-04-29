"""
EXPERIMENTAL
"""

import enum
import logging
import os
import queue
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple, Union

from ._commit_api import CommitOperationAdd, UploadInfo, _fetch_upload_modes
from ._local_folder import LocalUploadFileMetadata, LocalUploadFilePaths, get_local_upload_paths, read_upload_metadata
from .constants import DEFAULT_REVISION, REPO_TYPE_MODEL, REPO_TYPES
from .hf_api import HfApi
from .utils import DEFAULT_IGNORE_PATTERNS, filter_repo_objects
from .utils.sha import sha_fileobj


logger = logging.getLogger(__name__)

REPORT_STATUS_EVERY = 60  # seconds


def large_upload(
    repo_id: str,
    folder_path: Union[str, Path],
    api: Optional[HfApi] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    private: bool = False,
    allow_patterns: Optional[Union[List[str], str]] = None,
    ignore_patterns: Optional[Union[List[str], str]] = None,
    num_workers: Optional[int] = None,
):
    """Used to upload a large folder in the most resilient way possible.

    Steps:
    0. Check args and setup
    1. Create repo is missing
    2. List files to upload.
    3. Start workers:
        - Compute sha256
        - Get upload modes
        - Pre-upload LFS files
        - Make commits
    """
    # 0. Check args and setup
    if repo_type is None:
        repo_type = REPO_TYPE_MODEL
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
    if revision is None:
        revision = DEFAULT_REVISION

    folder_path = Path(folder_path).expanduser().resolve()
    if not folder_path.is_dir():
        raise ValueError(f"Provided path: '{folder_path}' is not a directory")

    if ignore_patterns is None:
        ignore_patterns = []
    elif isinstance(ignore_patterns, str):
        ignore_patterns = [ignore_patterns]
    ignore_patterns += DEFAULT_IGNORE_PATTERNS

    if num_workers is None:
        nb_cores = os.cpu_count() or 1
        num_workers = max(nb_cores - 2, 2)  # Use all but 2 cores, or at least 2 cores

    if api is None:
        api = HfApi()

    # 1. Create repo if missing
    repo_url = api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    logger.info(f"Repo created: {repo_url}")
    repo_id = repo_url.repo_id

    # 2. List files to upload
    #    Taken from '_prepare_upload_folder_additions(...)'
    paths_list = [
        get_local_upload_paths(folder_path, relpath)
        for relpath in filter_repo_objects(
            (path.relative_to(folder_path).as_posix() for path in folder_path.glob("**/*") if path.is_file()),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    ]
    logger.info(f"Found {len(paths_list)} candidate files to upload")

    # Read metadata for each file
    items = [(paths, read_upload_metadata(folder_path, paths.path_in_repo)) for paths in paths_list]

    # Start workers
    status = LargeUploadStatus(items)
    threads = [
        threading.Thread(
            target=_worker_job,
            kwargs={
                "status": status,
                "api": api,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "revision": revision,
            },
        )
        for _ in range(num_workers)
    ]

    for thread in threads:
        thread.start()

    while True:
        logger.info(status.current_report())
        time.sleep(REPORT_STATUS_EVERY)
        if status.is_done():
            logging.info("Is done: exiting main loop")
            break

    for thread in threads:
        thread.join()

    logger.info(status.current_report())
    logging.info("Upload is complete!")


####################
# Logic to manage workers and synchronize tasks
####################


class WorkerJob(enum.Enum):
    SHA256 = enum.auto()
    GET_UPLOAD_MODE = enum.auto()
    PREUPLOAD_LFS = enum.auto()
    COMMIT = enum.auto()


JOB_ITEM_T = Tuple[LocalUploadFilePaths, LocalUploadFileMetadata]


class LargeUploadStatus:
    """Contains information, queues and tasks for a large upload process."""

    def __init__(self, items: List[JOB_ITEM_T]):
        self.items = items
        self.queue_sha256: queue.Queue[JOB_ITEM_T] = queue.Queue()
        self.queue_get_upload_mode: queue.Queue[JOB_ITEM_T] = queue.Queue()
        self.queue_preupload_lfs: queue.Queue[JOB_ITEM_T] = queue.Queue()
        self.queue_commit: queue.Queue[JOB_ITEM_T] = queue.Queue()
        self.lock = Lock()

        self.nb_workers_sha256: int = 0
        self.nb_workers_get_upload_mode: int = 0
        self.nb_workers_preupload_lfs: int = 0
        self.nb_workers_commit: int = 0
        self.last_commit_attempt: Optional[float] = None

        self._started_at = datetime.now()

        # Setup queues
        for item in self.items:
            paths, metadata = item
            if metadata.sha256 is None:
                self.queue_sha256.put(item)
            elif metadata.upload_mode is None:
                self.queue_get_upload_mode.put(item)
            elif metadata.upload_mode == "lfs" and not metadata.is_uploaded:
                self.queue_preupload_lfs.put(item)
            elif not metadata.is_committed:
                self.queue_commit.put(item)
            else:
                logger.debug(f"Skipping file {paths.path_in_repo} (already uploaded and committed)")

    def current_report(self) -> str:
        """Generate a report of the current status of the large upload."""
        nb_hashed = 0
        size_hashed = 0
        nb_preuploaded = 0
        nb_lfs = 0
        nb_lfs_unsure = 0
        size_preuploaded = 0
        nb_committed = 0
        size_committed = 0
        total_size = 0
        ignored_files = 0
        total_files = 0

        with self.lock:
            for _, metadata in self.items:
                if metadata.should_ignore:
                    ignored_files += 1
                    continue
                total_size += metadata.size
                total_files += 1
                if metadata.sha256 is not None:
                    nb_hashed += 1
                    size_hashed += metadata.size
                if metadata.upload_mode == "lfs":
                    nb_lfs += 1
                if metadata.upload_mode is None:
                    nb_lfs_unsure += 1
                if metadata.is_uploaded:
                    nb_preuploaded += 1
                    size_preuploaded += metadata.size
                if metadata.is_committed:
                    nb_committed += 1
                    size_committed += metadata.size
            total_size_str = _format_size(total_size)

            now = datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            elapsed = now - self._started_at
            elapsed_str = str(elapsed).split(".")[0]  # remove milliseconds

            message = "\n\n" + "#" * 10 + "\n"
            message += "Large upload status:\n"
            message += "  Progress:\n"
            message += f"    {nb_hashed}/{total_files} hashed files ({_format_size(size_hashed)}/{total_size_str})\n"
            message += f"    {nb_preuploaded}/{nb_lfs} preuploaded LFS files ({_format_size(size_preuploaded)}/{total_size_str})"
            if nb_lfs_unsure > 0:
                message += f" (+{nb_lfs_unsure} files with unknown upload mode yet)"
            message += "\n"
            message += (
                f"    {nb_committed}/{total_files} committed files ({_format_size(size_committed)}/{total_size_str})\n"
            )
            message += f"    ({ignored_files} gitignored files)\n"
            message += "  Jobs:\n"
            message += f"    sha256: {self.nb_workers_sha256} workers ({self.queue_sha256.qsize()} items in queue)\n"
            message += f"    get_upload_mode: {self.nb_workers_get_upload_mode} workers ({self.queue_get_upload_mode.qsize()} items in queue)\n"
            message += f"    preupload_lfs: {self.nb_workers_preupload_lfs} workers ({self.queue_preupload_lfs.qsize()} items in queue)\n"
            message += f"    commit: {self.nb_workers_commit} workers ({self.queue_commit.qsize()} items in queue)\n"
            message += f"  Elapsed time: {elapsed_str}\n"
            message += f"  Current time: {now_str}\n"
            message += "#" * 10 + "\n\n"
            return message

    def is_done(self) -> bool:
        with self.lock:
            return all(metadata.is_committed or metadata.should_ignore for _, metadata in self.items)


def _worker_job(
    status: LargeUploadStatus,
    api: HfApi,
    repo_id: str,
    repo_type: str,
    revision: str,
):
    """
    Tasks:
        1. Compute sha256 (single file)
        2. Get upload mode (multiple files, max 50)
        3. Preupload LFS (single file)
        4. Commit (multiple files, max 50)

    Order of priority:
        1. Commit if more than 5 minutes since last commit attempt (and at least 1 file).
        2. Commit if at least 25 files are ready to commit.
        3. Get upload mode if at least 10 files.
        4. Preupload LFS file if at least 1 file and no worker is preuploading LFS.
        5. Compute sha256 if at least 1 file and no worker is computing sha256.
        6. Get upload mode if at least 1 file and no worker is getting upload mode.
        7. Compute LFS file if at least 1 file.
        8. Compute sha256 if at least 1 file.
        9. Get upload mode if at least 1 file.
        10. Commit if at least 1 file.

    Special rules:
        - TODO: If `hf_transfer` => only 1 LFS uploader at a time.
        - Always: only one worker can commit at a time.
    """
    while True:
        next_job: Optional[Tuple[WorkerJob, List[JOB_ITEM_T]]] = None

        # Determine next task
        with status.lock:
            # 1. Commit if more than 5 minutes since last commit attempt (and at least 1 file)
            if (
                status.nb_workers_commit == 0
                and status.queue_commit.qsize() > 0
                and (status.last_commit_attempt is None or time.time() - status.last_commit_attempt > 5 * 60)
            ):
                status.nb_workers_commit += 1
                next_job = (WorkerJob.COMMIT, _get_n(status.queue_commit, 25))
                logger.debug("Job: commit (more than 5 minutes since last commit attempt)")

            # 2. Commit if at least 25 files are ready to commit
            elif status.nb_workers_commit == 0 and status.queue_commit.qsize() >= 25:
                status.nb_workers_commit += 1
                next_job = (WorkerJob.COMMIT, _get_n(status.queue_commit, 25))
                logger.debug("Job: commit (>25 files ready)")

            # 3. Get upload mode if at least 10 files
            elif status.queue_get_upload_mode.qsize() >= 10:
                status.nb_workers_get_upload_mode += 1
                next_job = (WorkerJob.GET_UPLOAD_MODE, _get_n(status.queue_get_upload_mode, 50))
                logger.debug("Job: get upload mode (>10 files ready)")

            # 4. Preupload LFS file if at least 1 file and no worker is preuploading LFS
            elif status.queue_preupload_lfs.qsize() > 0 and status.nb_workers_preupload_lfs == 0:
                status.nb_workers_preupload_lfs += 1
                next_job = (WorkerJob.PREUPLOAD_LFS, _get_one(status.queue_preupload_lfs))
                logger.debug("Job: preupload LFS (no other worker preuploading LFS)")

            # 5. Compute sha256 if at least 1 file and no worker is computing sha256
            elif status.queue_sha256.qsize() > 0 and status.nb_workers_sha256 == 0:
                status.nb_workers_sha256 += 1
                next_job = (WorkerJob.SHA256, _get_one(status.queue_sha256))
                logger.debug("Job: sha256 (no other worker computing sha256)")

            # 6. Get upload mode if at least 1 file and no worker is getting upload mode
            elif status.queue_get_upload_mode.qsize() > 0 and status.nb_workers_get_upload_mode == 0:
                status.nb_workers_get_upload_mode += 1
                next_job = (WorkerJob.GET_UPLOAD_MODE, _get_n(status.queue_get_upload_mode, 50))
                logger.debug("Job: get upload mode (no other worker getting upload mode)")

            # 7. Compute LFS file if at least 1 file
            elif status.queue_preupload_lfs.qsize() > 0:
                status.nb_workers_preupload_lfs += 1
                next_job = (WorkerJob.PREUPLOAD_LFS, _get_one(status.queue_preupload_lfs))
                logger.debug("Job: preupload LFS")

            # 8. Compute sha256 if at least 1 file
            elif status.queue_sha256.qsize() > 0:
                status.nb_workers_sha256 += 1
                next_job = (WorkerJob.SHA256, _get_one(status.queue_sha256))
                logger.debug("Job: sha256")

            # 9. Get upload mode if at least 1 file
            elif status.queue_get_upload_mode.qsize() > 0:
                status.nb_workers_get_upload_mode += 1
                next_job = (WorkerJob.GET_UPLOAD_MODE, _get_n(status.queue_get_upload_mode, 50))
                logger.debug("Job: get upload mode")

            # 10. Commit if at least 1 file
            elif status.nb_workers_commit == 0 and status.queue_commit.qsize() > 0:
                status.nb_workers_commit += 1
                next_job = (WorkerJob.COMMIT, _get_n(status.queue_commit, 25))
                logger.debug("Job: commit")

            # End of job
            else:
                logger.debug("No more tasks to perform!")
                return

        # Perform task
        job, items = next_job

        if job == WorkerJob.SHA256:
            item = items[0]  # single item
            try:
                _compute_sha256(item)
                status.queue_get_upload_mode.put(item)
            except Exception as e:
                logger.error(f"Failed to compute sha256: {e}")
                traceback.format_exc()
                status.queue_sha256.put(item)

            with status.lock:
                status.nb_workers_sha256 -= 1

        elif job == WorkerJob.GET_UPLOAD_MODE:
            try:
                _get_upload_mode(items, api=api, repo_id=repo_id, repo_type=repo_type, revision=revision)
            except Exception as e:
                logger.error(f"Failed to get upload mode: {e}")
                traceback.format_exc()

            # Items are either:
            # - dropped (if should_ignore)
            # - put in LFS queue (if LFS)
            # - put in commit queue (if regular)
            # - or put back (if error occurred).
            for item in items:
                _, metadata = item
                if metadata.should_ignore:
                    continue
                if metadata.upload_mode == "lfs":
                    status.queue_preupload_lfs.put(item)
                elif metadata.upload_mode == "regular":
                    status.queue_commit.put(item)
                else:
                    status.queue_get_upload_mode.put(item)

            with status.lock:
                status.nb_workers_get_upload_mode -= 1

        elif job == WorkerJob.PREUPLOAD_LFS:
            item = items[0]  # single item
            try:
                _preupload_lfs(item, api=api, repo_id=repo_id, repo_type=repo_type, revision=revision)
                status.queue_commit.put(item)
            except Exception as e:
                logger.error(f"Failed to preupload LFS: {e}")
                traceback.format_exc()
                status.queue_preupload_lfs.put(item)

            with status.lock:
                status.nb_workers_preupload_lfs -= 1

        elif job == WorkerJob.COMMIT:
            try:
                _commit(items, api=api, repo_id=repo_id, repo_type=repo_type, revision=revision)
            except Exception as e:
                logger.error(f"Failed to commit: {e}")
                traceback.format_exc()
                for item in items:
                    status.queue_commit.put(item)
            with status.lock:
                status.last_commit_attempt = time.time()
                status.nb_workers_commit -= 1


####################
# Atomic jobs (sha256, get_upload_mode, preupload_lfs, commit)
####################


def _compute_sha256(item: JOB_ITEM_T) -> None:
    """Compute sha256 of a file and save it in metadata."""
    paths, metadata = item
    if metadata.sha256 is None:
        with paths.file_path.open("rb") as f:
            metadata.sha256 = sha_fileobj(f).hex()
    metadata.save(paths)


def _get_upload_mode(items: List[JOB_ITEM_T], api: HfApi, repo_id: str, repo_type: str, revision: str) -> None:
    """Get upload mode for each file and update metadata.

    Also receive info if the file should be ignored.
    """
    additions = [_build_hacky_operation(item) for item in items]
    _fetch_upload_modes(
        additions=additions,
        repo_type=repo_type,
        repo_id=repo_id,
        headers=api._build_hf_headers(),
        revision=revision,
    )
    for item, addition in zip(items, additions):
        paths, metadata = item
        metadata.upload_mode = addition._upload_mode
        metadata.should_ignore = addition._should_ignore
        metadata.save(paths)


def _preupload_lfs(item: JOB_ITEM_T, api: HfApi, repo_id: str, repo_type: str, revision: str) -> None:
    """Preupload LFS file and update metadata."""
    paths, metadata = item
    addition = _build_hacky_operation(item)
    api.preupload_lfs_files(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        additions=[addition],
    )

    metadata.is_uploaded = True
    metadata.save(paths)


def _commit(items: List[JOB_ITEM_T], api: HfApi, repo_id: str, repo_type: str, revision: str) -> None:
    """Commit files to the repo."""
    additions = [_build_hacky_operation(item) for item in items]
    api.create_commit(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        operations=additions,
        commit_message="Add files using large-upload tool",
    )
    for paths, metadata in items:
        metadata.is_committed = True
        metadata.save(paths)


####################
# Hacks with CommitOperationAdd to bypass checks/sha256 calculation
####################


class HackyCommitOperationAdd(CommitOperationAdd):
    def __post_init__(self) -> None:
        if isinstance(self.path_or_fileobj, Path):
            self.path_or_fileobj = str(self.path_or_fileobj)


def _build_hacky_operation(item: JOB_ITEM_T) -> HackyCommitOperationAdd:
    paths, metadata = item
    operation = HackyCommitOperationAdd(path_in_repo=paths.path_in_repo, path_or_fileobj=paths.file_path)
    with paths.file_path.open("rb") as file:
        sample = file.peek(512)[:512]
    if metadata.sha256 is None:
        raise ValueError("sha256 must have been computed by now!")
    operation.upload_info = UploadInfo(sha256=bytes.fromhex(metadata.sha256), size=metadata.size, sample=sample)
    return operation


####################
# Misc helpers
####################


def _get_one(queue: queue.Queue[JOB_ITEM_T]) -> List[JOB_ITEM_T]:
    return [queue.get()]


def _get_n(queue: queue.Queue[JOB_ITEM_T], n: int) -> List[JOB_ITEM_T]:
    return [queue.get() for _ in range(min(queue.qsize(), n))]


def _format_size(num: int) -> str:
    """Format size in bytes into a human-readable string.

    Taken from https://stackoverflow.com/a/1094933
    TODO: deduplicate this from `_cache_manager.py`
    """
    num_f = float(num)
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num_f) < 1000.0:
            return f"{num_f:3.1f}{unit}"
        num_f /= 1000.0
    return f"{num_f:.1f}Y"
