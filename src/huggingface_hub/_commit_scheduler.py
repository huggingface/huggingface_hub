import atexit
import logging
import time
from pathlib import Path
from threading import Lock, Thread
from typing import List, Optional, Union, Dict
from dataclasses import dataclass
from .hf_api import IGNORE_GIT_FOLDER_PATTERNS, CommitInfo, HfApi, _prepare_upload_folder_additions, CommitOperationAdd
from .utils import filter_repo_objects

logger = logging.getLogger(__name__)


# TODO: partial files in CommitOperationAdd !!!


@dataclass(frozen=True)
class _FileToUpload:
    local_path: Path
    path_in_repo: str
    size_limit: int
    last_modified: float


class CommitScheduler:
    def __init__(
        self,
        *,
        repo_id: str,
        folder_path: Union[str, Path],
        every: Union[int, float] = 5,
        path_in_repo: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        private: bool = False,
        token: Optional[str] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        hf_api: Optional["HfApi"] = None,
    ) -> None:
        """
        Scheduler to upload a local folder to the Hub at regular intervals (e.g. push to hub every 5 minutes).

        Args:
            repo_id (`str`):
                The id of the repo to commit to.
            folder_path (`str` or `Path`):
                Path to the local folder to upload regularly.
            every (`int` or `float`, *optional*):
                The number of minutes between each commit. Defaults to 5 minutes.
            path_in_repo (`str`, *optional*):
                Relative path of the directory in the repo, for example: `"checkpoints/"`. Defaults to the root folder
                of the repository.
            repo_type (`str`, *optional*):
                The type of the repo to commit to. Defaults to `model`.
            revision (`str`, *optional*):
                The revision of the repo to commit to. Defaults to `main`.
            private (`bool`, *optional*):
                Whether to make the repo private. Defaults to `False`. This value is ignored if the repo already exist.
            token (`str`, *optional*):
                The token to use to commit to the repo. Defaults to the token saved on the machine.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are uploaded.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not uploaded.
            hf_api (`HfApi`, *optional*):
                The [`HfApi`] client to use to commit to the Hub. Can be set with custom settings (user agent, token,...).
        """
        self.api = hf_api or HfApi()

        # Repository
        repo_url = self.api.create_repo(
            repo_id=repo_id, token=token, private=private, repo_type=repo_type, exist_ok=True
        )
        self.repo_id = repo_url.repo_id
        self.repo_type = repo_type
        self.revision = revision
        self.token = token

        # Folder
        self.folder_path = Path(folder_path).expanduser().resolve()
        self.path_in_repo = path_in_repo or ""
        self.allow_patterns = allow_patterns

        if ignore_patterns is None:
            ignore_patterns = []
        elif isinstance(ignore_patterns, str):
            ignore_patterns = [ignore_patterns]
        self.ignore_patterns = ignore_patterns + IGNORE_GIT_FOLDER_PATTERNS

        if self.folder_path.is_file():
            raise ValueError(f"'folder_path' must be a directory, not a file: '{self.folder_path}'.")
        self.folder_path.mkdir(parents=True, exist_ok=True)

        # Keep track of already uploaded files
        self.last_future: Optional[CommitInfo] = None
        self.last_uploaded: Dict[Path:float] = {}  # key is local path, value is timestamp

        # Scheduler
        if not every > 0:
            raise ValueError(f"'every' must be a positive integer, not '{every}'.")
        self.lock = Lock()
        self.every = every

        logger.info(f"Scheduled job to push '{self.folder_path}' to '{self.repo_id}' every {self.every} minutes.")
        self._scheduler_thread = Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        atexit.register(self._push_to_hub)

    def _run_scheduler(self) -> None:
        """Dumb thread waiting between each scheduled push to Hub."""
        while True:
            self.last_future = self.api.run_as_future(self._push_to_hub)
            time.sleep(self.every * 60)

    def _push_to_hub(self) -> Optional[CommitInfo]:
        logger.info("Scheduled commit triggered.")

        # Check files to upload (with lock)
        with self.lock:
            logger.debug("Listing files to upload for scheduled commit.")

            # List files from folder (taken from `_prepare_upload_folder_additions`)
            relpath_to_abspath = {
                path.relative_to(self.folder_path).as_posix(): path
                for path in sorted(self.folder_path.glob("**/*"))  # sorted to be deterministic
                if path.is_file()
            }
            prefix = f"{self.path_in_repo.strip('/')}/" if self.path_in_repo else ""

            # Filter with pattern + filter out unchanged files + retrieve current file size
            files_to_upload: List[_FileToUpload] = []
            for relpath in filter_repo_objects(
                relpath_to_abspath.keys(), allow_patterns=self.allow_patterns, ignore_patterns=self.ignore_patterns
            ):
                local_path = relpath_to_abspath[relpath]
                stat = local_path.stat()
                if self.last_uploaded.get(local_path) is None or self.last_uploaded[local_path] != stat.st_mtime:
                    files_to_upload.append(
                        _FileToUpload(
                            local_path=local_path,
                            path_in_repo=prefix + relpath,
                            size_limit=stat.st_size,
                            last_modified=stat.st_mtime,
                        )
                    )

        # Return if nothing to upload
        if len(files_to_upload) == 0:
            logger.debug("Dropping schedule commit: no changed file to upload.")
            return None

        # Convert `_FileToUpload` as `CommitOperationAdd` (=> compute file shas + limit to file size)
        logger.debug("Removing unchanged files since previous scheduled commit.")
        add_operations = [
            # TODO: partial files!!!
            CommitOperationAdd(
                path_or_fileobj=file_to_upload.local_path,  # absolute path on disk
                path_in_repo=file_to_upload.path_in_repo,  # "absolute" path in repo
            )
            for file_to_upload in files_to_upload
        ]

        # Upload files (append mode expected - no need for lock)
        logger.debug("Uploading files for scheduled commit.")
        commit_info = self.api.create_commit(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            operations=add_operations,
            commit_message="Scheduled Commit",
            revision=self.revision,
        )

        # Successful commit: keep track of the latest "last_modified" for each file
        for file in files_to_upload:
            self.last_uploaded[file.local_path] = file.last_modified
        return commit_info
