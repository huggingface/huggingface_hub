"""
Type definitions and utilities for the `create_commit` API
"""
import base64
import io
import os
import warnings
from collections import defaultdict
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Union

import requests

from .constants import ENDPOINT
from .lfs import UploadInfo, _validate_batch_actions, lfs_upload, post_lfs_batch_info
from .utils import (
    build_hf_headers,
    chunk_iterable,
    hf_raise_for_status,
    logging,
    validate_hf_hub_args,
)
from .utils._deprecation import _deprecate_method
from .utils._typing import Literal


logger = logging.get_logger(__name__)


UploadMode = Literal["lfs", "regular"]


@dataclass
class CommitOperationDelete:
    """
    Data structure holding necessary info to delete a file or a folder from a repository
    on the Hub.

    Args:
        path_in_repo (`str`):
            Relative filepath in the repo, for example: `"checkpoints/1fec34a/weights.bin"`
            for a file or `"checkpoints/1fec34a/"` for a folder.
        is_folder (`bool` or `Literal["auto"]`, *optional*)
            Whether the Delete Operation applies to a folder or not. If "auto", the path
            type (file or folder) is guessed automatically by looking if path ends with
            a "/" (folder) or not (file). To explicitly set the path type, you can set
            `is_folder=True` or `is_folder=False`.
    """

    path_in_repo: str
    is_folder: Union[bool, Literal["auto"]] = "auto"

    def __post_init__(self):
        if self.is_folder == "auto":
            self.is_folder = self.path_in_repo.endswith("/")
        if not isinstance(self.is_folder, bool):
            raise ValueError(
                "Wrong value for `is_folder`. Must be one of [`True`, `False`,"
                f" `'auto'`]. Got '{self.is_folder}'."
            )


@dataclass
class CommitOperationAdd:
    """
    Data structure holding necessary info to upload a file to a repository on the Hub.

    Args:
        path_in_repo (`str`):
            Relative filepath in the repo, for example: `"checkpoints/1fec34a/weights.bin"`
        path_or_fileobj (`str`, `bytes`, or `BinaryIO`):
            Either:
            - a path to a local file (as str) to upload
            - a buffer of bytes (`bytes`) holding the content of the file to upload
            - a "file object" (subclass of `io.BufferedIOBase`), typically obtained
                with `open(path, "rb")`. It must support `seek()` and `tell()` methods.

    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If `path_or_fileobj` is not one of `str`, `bytes` or `io.BufferedIOBase`.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If `path_or_fileobj` is a `str` but not a path to an existing file.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If `path_or_fileobj` is a `io.BufferedIOBase` but it doesn't support both
            `seek()` and `tell()`.
    """

    path_in_repo: str
    path_or_fileobj: Union[str, bytes, BinaryIO]
    upload_info: UploadInfo = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validates `path_or_fileobj` and compute `upload_info`."""
        # Validate `path_or_fileobj` value
        if isinstance(self.path_or_fileobj, str):
            path_or_fileobj = os.path.normpath(os.path.expanduser(self.path_or_fileobj))
            if not os.path.isfile(path_or_fileobj):
                raise ValueError(
                    f"Provided path: '{path_or_fileobj}' is not a file on the local"
                    " file system"
                )
        elif not isinstance(self.path_or_fileobj, (io.BufferedIOBase, bytes)):
            # ^^ Inspired from: https://stackoverflow.com/questions/44584829/how-to-determine-if-file-is-opened-in-binary-or-text-mode
            raise ValueError(
                "path_or_fileobj must be either an instance of str, bytes or"
                " io.BufferedIOBase. If you passed a file-like object, make sure it is"
                " in binary mode."
            )
        if isinstance(self.path_or_fileobj, io.BufferedIOBase):
            try:
                self.path_or_fileobj.tell()
                self.path_or_fileobj.seek(0, os.SEEK_CUR)
            except (OSError, AttributeError) as exc:
                raise ValueError(
                    "path_or_fileobj is a file-like object but does not implement"
                    " seek() and tell()"
                ) from exc

        # Compute "upload_info" attribute
        if isinstance(self.path_or_fileobj, str):
            self.upload_info = UploadInfo.from_path(self.path_or_fileobj)
        elif isinstance(self.path_or_fileobj, bytes):
            self.upload_info = UploadInfo.from_bytes(self.path_or_fileobj)
        else:
            self.upload_info = UploadInfo.from_fileobj(self.path_or_fileobj)

    @_deprecate_method(
        version="0.14", message="Operation is validated at initialization."
    )
    def validate(self) -> None:
        pass

    @_deprecate_method(version="0.14", message="Use `upload_info` property instead.")
    def _upload_info(self) -> UploadInfo:
        return self.upload_info

    @contextmanager
    def as_file(self):
        """
        A context manager that yields a file-like object allowing to read the underlying
        data behind `path_or_fileobj`.

        Example:

        ```python
        >>> operation = CommitOperationAdd(
        ...        path_in_repo="remote/dir/weights.h5",
        ...        path_or_fileobj="./local/weights.h5",
        ... )
        CommitOperationAdd(path_in_repo='remote/dir/weights.h5', path_or_fileobj='./local/weights.h5')

        >>> with operation.as_file() as file:
        ...     content = file.read()
        ```

        """
        if isinstance(self.path_or_fileobj, str):
            with open(self.path_or_fileobj, "rb") as file:
                yield file
        elif isinstance(self.path_or_fileobj, bytes):
            yield io.BytesIO(self.path_or_fileobj)
        elif isinstance(self.path_or_fileobj, io.BufferedIOBase):
            prev_pos = self.path_or_fileobj.tell()
            yield self.path_or_fileobj
            self.path_or_fileobj.seek(prev_pos, io.SEEK_SET)

    def b64content(self) -> bytes:
        """
        The base64-encoded content of `path_or_fileobj`

        Returns: `bytes`
        """
        with self.as_file() as file:
            return base64.b64encode(file.read())


CommitOperation = Union[CommitOperationAdd, CommitOperationDelete]


def warn_on_overwriting_operations(operations: List[CommitOperation]) -> None:
    """
    Warn user when a list of operations is expected to overwrite itself in a single
    commit.

    Rules:
    - If a filepath is updated by multiple `CommitOperationAdd` operations, a warning
      message is triggered.
    - If a filepath is updated at least once by a `CommitOperationAdd` and then deleted
      by a `CommitOperationDelete`, a warning is triggered.
    - If a `CommitOperationDelete` deletes a filepath that is then updated by a
      `CommitOperationAdd`, no warning is triggered. This is usually useless (no need to
      delete before upload) but can happen if a user deletes an entire folder and then
      add new files to it.
    """
    nb_additions_per_path: Dict[str, int] = defaultdict(int)
    for operation in operations:
        path_in_repo = operation.path_in_repo
        if isinstance(operation, CommitOperationAdd):
            if nb_additions_per_path[path_in_repo] > 0:
                warnings.warn(
                    "About to update multiple times the same file in the same commit:"
                    f" '{path_in_repo}'. This can cause undesired inconsistencies in"
                    " your repo."
                )
            nb_additions_per_path[path_in_repo] += 1
            for parent in PurePosixPath(path_in_repo).parents:
                # Also keep track of number of updated files per folder
                # => warns if deleting a folder overwrite some contained files
                nb_additions_per_path[str(parent)] += 1
        if isinstance(operation, CommitOperationDelete):
            if nb_additions_per_path[str(PurePosixPath(path_in_repo))] > 0:
                if operation.is_folder:
                    warnings.warn(
                        "About to delete a folder containing files that have just been"
                        f" updated within the same commit: '{path_in_repo}'. This can"
                        " cause undesired inconsistencies in your repo."
                    )
                else:
                    warnings.warn(
                        "About to delete a file that have just been updated within the"
                        f" same commit: '{path_in_repo}'. This can cause undesired"
                        " inconsistencies in your repo."
                    )


@validate_hf_hub_args
def upload_lfs_files(
    *,
    additions: List[CommitOperationAdd],
    repo_type: str,
    repo_id: str,
    token: Optional[str],
    endpoint: Optional[str] = None,
    num_threads: int = 5,
):
    """
    Uploads the content of `additions` to the Hub using the large file storage protocol.

    Relevant external documentation:
        - LFS Batch API: https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md

    Args:
        additions (`List` of `CommitOperationAdd`):
            The files to be uploaded
        repo_type (`str`):
            Type of the repo to upload to: `"model"`, `"dataset"` or `"space"`.
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        token (`str`, *optional*):
            An authentication token ( See https://huggingface.co/settings/tokens )
        num_threads (`int`, *optional*):
            The number of concurrent threads to use when uploading. Defaults to 5.


    Raises: `RuntimeError` if an upload failed for any reason

    Raises: `ValueError` if the server returns malformed responses

    Raises: `requests.HTTPError` if the LFS batch endpoint returned an HTTP
        error

    """
    # Step 1: retrieve upload instructions from the LFS batch endpoint.
    #         Upload instructions are retrieved by chunk of 256 files to avoid reaching
    #         the payload limit.
    batch_actions: List[Dict] = []
    for chunk in chunk_iterable(additions, chunk_size=256):
        batch_actions_chunk, batch_errors_chunk = post_lfs_batch_info(
            upload_infos=[op.upload_info for op in chunk],
            token=token,
            repo_id=repo_id,
            repo_type=repo_type,
            endpoint=endpoint,
        )

        # If at least 1 error, we do not retrieve information for other chunks
        if batch_errors_chunk:
            message = "\n".join(
                [
                    f'Encountered error for file with OID {err.get("oid")}:'
                    f' `{err.get("error", {}).get("message")}'
                    for err in batch_errors_chunk
                ]
            )
            raise ValueError(f"LFS batch endpoint returned errors:\n{message}")

        batch_actions += batch_actions_chunk

    # Step 2: upload files concurrently according to these instructions
    oid2addop = {add_op.upload_info.sha256.hex(): add_op for add_op in additions}
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        logger.debug(
            f"Uploading {len(batch_actions)} LFS files to the Hub using up to"
            f" {num_threads} threads concurrently"
        )
        # Upload the files concurrently, stopping on the first exception
        futures2operation: Dict[futures.Future, CommitOperationAdd] = {
            pool.submit(
                _upload_lfs_object,
                operation=oid2addop[batch_action["oid"]],
                lfs_batch_action=batch_action,
                token=token,
            ): oid2addop[batch_action["oid"]]
            for batch_action in batch_actions
        }
        completed, pending = futures.wait(
            futures2operation,
            return_when=futures.FIRST_EXCEPTION,
        )

        for pending_future in pending:
            # Cancel pending futures
            # Uploads already in progress can't be stopped unfortunately,
            # as `requests` does not support cancelling / aborting a request
            pending_future.cancel()

        for future in completed:
            operation = futures2operation[future]
            try:
                future.result()
            except Exception as exc:
                # Raise the first exception encountered
                raise RuntimeError(
                    f"Error while uploading {operation.path_in_repo} to the Hub"
                ) from exc


def _upload_lfs_object(
    operation: CommitOperationAdd, lfs_batch_action: dict, token: Optional[str]
):
    """
    Handles uploading a given object to the Hub with the LFS protocol.

    Defers to [`~utils.lfs.lfs_upload`] for the actual upload logic.

    Can be a No-op if the content of the file is already present on the hub
    large file storage.

    Args:
        operation (`CommitOperationAdd`):
            The add operation triggering this upload
        lfs_batch_action (`dict`):
            Upload instructions from the LFS batch endpoint for this object.
            See [`~utils.lfs.post_lfs_batch_info`] for more details.
        token (`str`, *optional*):
            A [user access token](https://hf.co/settings/tokens) to authenticate requests against the Hub

    Raises: `ValueError` if `lfs_batch_action` is improperly formatted
    """
    _validate_batch_actions(lfs_batch_action)
    upload_info = operation.upload_info
    actions = lfs_batch_action.get("actions")
    if actions is None:
        # The file was already uploaded
        logger.debug(
            f"Content of file {operation.path_in_repo} is already present upstream"
            " - skipping upload"
        )
        return
    upload_action = lfs_batch_action["actions"].get("upload")
    verify_action = lfs_batch_action["actions"].get("verify")

    with operation.as_file() as fileobj:
        logger.debug(f"Uploading {operation.path_in_repo} as LFS file...")
        lfs_upload(
            fileobj=fileobj,
            upload_action=upload_action,
            verify_action=verify_action,
            upload_info=upload_info,
            token=token,
        )
        logger.debug(f"{operation.path_in_repo}: Upload successful")


def _validate_preupload_info(preupload_info: dict):
    files = preupload_info.get("files")
    if not isinstance(files, list):
        raise ValueError("preupload_info is improperly formatted")
    for file_info in files:
        if not (
            isinstance(file_info, dict)
            and isinstance(file_info.get("path"), str)
            and isinstance(file_info.get("uploadMode"), str)
            and (file_info["uploadMode"] in ("lfs", "regular"))
        ):
            raise ValueError("preupload_info is improperly formatted:")
    return preupload_info


@validate_hf_hub_args
def fetch_upload_modes(
    additions: Iterable[CommitOperationAdd],
    repo_type: str,
    repo_id: str,
    token: Optional[str],
    revision: str,
    endpoint: Optional[str] = None,
    create_pr: bool = False,
) -> Dict[str, UploadMode]:
    """
    Requests the Hub "preupload" endpoint to determine wether each input file
    should be uploaded as a regular git blob or as git LFS blob.

    Args:
        additions (`Iterable` of :class:`CommitOperationAdd`):
            Iterable of :class:`CommitOperationAdd` describing the files to
            upload to the Hub.
        repo_type (`str`):
            Type of the repo to upload to: `"model"`, `"dataset"` or `"space"`.
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        token (`str`, *optional*):
            An authentication token ( See https://huggingface.co/settings/tokens )
        revision (`str`):
            The git revision to upload the files to. Can be any valid git revision.

    Returns: `Dict[str, UploadMode]`
        Key is the file path, value is the upload mode ("regular" or "lfs").

    Raises:
        [`~utils.HfHubHTTPError`]
            If the Hub API returned an error.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If the Hub API response is improperly formatted.
    """
    endpoint = endpoint if endpoint is not None else ENDPOINT
    headers = build_hf_headers(token=token)

    # Fetch upload mode (LFS or regular) chunk by chunk.
    upload_modes: Dict[str, UploadMode] = {}
    for chunk in chunk_iterable(additions, 256):
        payload = {
            "files": [
                {
                    "path": op.path_in_repo,
                    "sample": base64.b64encode(op.upload_info.sample).decode("ascii"),
                    "size": op.upload_info.size,
                    "sha": op.upload_info.sha256.hex(),
                }
                for op in chunk
            ]
        }

        resp = requests.post(
            f"{endpoint}/api/{repo_type}s/{repo_id}/preupload/{revision}",
            json=payload,
            headers=headers,
            params={"create_pr": "1"} if create_pr else None,
        )
        hf_raise_for_status(resp)
        preupload_info = _validate_preupload_info(resp.json())
        upload_modes.update(
            **{file["path"]: file["uploadMode"] for file in preupload_info["files"]}
        )

    # If a file is empty, it is most likely a mistake.
    # => a warning message is triggered to warn the user.
    #    => except if `.gitkeep` as it is a legit use case for an empty file.
    #
    # Empty files cannot be uploaded as LFS (S3 would fail with a 501 Not Implemented)
    # => empty files are uploaded as "regular" to still allow users to commit them.
    for addition in additions:
        if addition.upload_info.size == 0:
            path = addition.path_in_repo
            if not path.endswith(".gitkeep"):
                warnings.warn(
                    f"About to commit an empty file: '{path}'. Are you sure this is"
                    " intended?"
                )
            upload_modes[path] = "regular"

    return upload_modes


def prepare_commit_payload(
    operations: Iterable[CommitOperation],
    upload_modes: Dict[str, UploadMode],
    commit_message: str,
    commit_description: Optional[str] = None,
    parent_commit: Optional[str] = None,
) -> Iterable[Dict[str, Any]]:
    """
    Builds the payload to POST to the `/commit` API of the Hub.

    Payload is returned as an iterator so that it can be streamed as a ndjson in the
    POST request.

    For more information, see:
        - https://github.com/huggingface/huggingface_hub/issues/1085#issuecomment-1265208073
        - http://ndjson.org/
    """
    commit_description = commit_description if commit_description is not None else ""

    # 1. Send a header item with the commit metadata
    header_value = {"summary": commit_message, "description": commit_description}
    if parent_commit is not None:
        header_value["parentCommit"] = parent_commit
    yield {"key": "header", "value": header_value}

    # 2. Send operations, one per line
    for operation in operations:
        # 2.a. Case adding a regular file
        if (
            isinstance(operation, CommitOperationAdd)
            and upload_modes.get(operation.path_in_repo) == "regular"
        ):
            yield {
                "key": "file",
                "value": {
                    "content": operation.b64content().decode(),
                    "path": operation.path_in_repo,
                    "encoding": "base64",
                },
            }
        # 2.b. Case adding an LFS file
        elif (
            isinstance(operation, CommitOperationAdd)
            and upload_modes.get(operation.path_in_repo) == "lfs"
        ):
            yield {
                "key": "lfsFile",
                "value": {
                    "path": operation.path_in_repo,
                    "algo": "sha256",
                    "oid": operation.upload_info.sha256.hex(),
                    "size": operation.upload_info.size,
                },
            }
        # 2.c. Case deleting a file or folder
        elif isinstance(operation, CommitOperationDelete):
            yield {
                "key": "deletedFolder" if operation.is_folder else "deletedFile",
                "value": {"path": operation.path_in_repo},
            }
        # 2.d. Never expected to happen
        else:
            raise ValueError(
                f"Unknown operation to commit. Operation: {operation}. Upload mode:"
                f" {upload_modes.get(operation.path_in_repo)}"
            )
