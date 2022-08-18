"""
Type definitions and utilities for the `create_commit` API
"""

import base64
import io
import os
import sys
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import requests

from .constants import ENDPOINT
from .lfs import UploadInfo, _validate_batch_actions, lfs_upload, post_lfs_batch_info
from .utils import logging
from .utils._errors import _raise_convert_bad_request


logger = logging.get_logger(__name__)


UploadMode = Literal["lfs", "regular"]


@dataclass
class CommitOperationDelete:
    """
    Data structure holding necessary info to delete
    a file from a repository on the Hub

    Args:
        path_in_repo (`str`):
            Relative filepath in the repo, for example:
            `"checkpoints/1fec34a/weights.bin"`
    """

    path_in_repo: str


@dataclass
class CommitOperationAdd:
    """
    Data structure holding necessary info to upload a file
    to a repository on the Hub

    Args:
        path_in_repo (`str`):
            Relative filepath in the repo, for example:
            `"checkpoints/1fec34a/weights.bin"`
        path_or_fileobj (`str`, `bytes`, or `BinaryIO`):
            Either:
            - a path to a local file (as str) to upload
            - a buffer of bytes (`bytes`) holding the content of the file to upload
            - a "file object" (subclass of `io.BufferedIOBase`), typically obtained
                with `open(path, "rb")`. It must support `seek()` and `tell()` methods.
    """

    path_in_repo: str
    path_or_fileobj: Union[str, bytes, BinaryIO]

    __upload_info: Optional[UploadInfo] = field(default=None, init=False)

    def validate(self):
        """
        Ensures `path_or_fileobj` is valid:
        - Ensures it is either a `str`, `bytes` or an instance of `io.BufferedIOBase`
        - If it is a `str`, ensure that it is a file path and that the file exists
        - If it is an instance of `io.BufferedIOBase`, ensures it supports `seek()` and `tell()`

        Raises: `ValueError` if `path_or_fileobj` is not valid
        """
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

    def _upload_info(self) -> UploadInfo:
        """
        Computes and caches UploadInfo for the underlying data behind `path_or_fileobj`
        Triggers `self.validate`.

        Raises: `ValueError` if self.validate fails
        """
        self.validate()
        if self.__upload_info is None:
            if isinstance(self.path_or_fileobj, str):
                self.__upload_info = UploadInfo.from_path(self.path_or_fileobj)
            elif isinstance(self.path_or_fileobj, bytes):
                self.__upload_info = UploadInfo.from_bytes(self.path_or_fileobj)
            else:
                self.__upload_info = UploadInfo.from_fileobj(self.path_or_fileobj)
        return self.__upload_info

    @contextmanager
    def as_file(self):
        """
        A context manager that yields a file-like object allowing to read the underlying
        data behind `path_or_fileobj`.

        Triggers `self.validate`.

        Raises: `ValueError` if self.validate fails

        Example:

        ```python
        >>> operation = CommitOperationAdd(
        ...        path_in_repo="remote/dir/weights.h5",
        ...        path_or_fileobj="./local/weights.h5",
        ... )
        CommitOperationAdd(path_in_repo='remote/dir/weights.h5', path_or_fileobj='./local/weights.h5', _upload_info=None)

        >>> with operation.as_file() as file:
        ...     content = file.read()
        ```

        """
        self.validate()
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


def upload_lfs_files(
    *,
    additions: Iterable[CommitOperationAdd],
    repo_type: str,
    repo_id: str,
    token: str,
    endpoint: Optional[str] = None,
    num_threads: int = 5,
):
    """
    Uploads the content of `additions` to the Hub using the large file storage protocol.

    Relevant external documentation:
        - LFS Batch API: https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md

    Args:
        additions (`Iterable` of `CommitOperationAdd`):
            The files to be uploaded
        repo_type (`str`):
            Type of the repo to upload to: `"model"`, `"dataset"` or `"space"`.
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        token (`str`):
            An authentication token ( See https://huggingface.co/settings/tokens )
        num_threads (`int`, *optional*):
            The number of concurrent threads to use when uploading. Defaults to 5.


    Raises: `RuntimeError` if an upload failed for any reason

    Raises: `ValueError` if the server returns malformed responses

    Raises: `requests.HTTPError` if the LFS batch endpoint returned an HTTP
        error

    """
    # Step 1: retrieve upload instructions from the LFS batch endpoint
    batch_actions, batch_errors = post_lfs_batch_info(
        upload_infos=[op._upload_info() for op in additions],
        token=token,
        repo_id=repo_id,
        repo_type=repo_type,
        endpoint=endpoint,
    )
    if batch_errors:
        message = "\n".join(
            [
                f'Encountered error for file with OID {err.get("oid")}:'
                f' `{err.get("error", {}).get("message")}'
                for err in batch_errors
            ]
        )
        raise ValueError(f"LFS batch endpoint returned errors:\n{message}")

    # Step 2: upload files concurrently according to these instructions
    oid2addop = {add_op._upload_info().sha256.hex(): add_op for add_op in additions}
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
    operation: CommitOperationAdd, lfs_batch_action: dict, token: str
):
    """
    Handles uploading a given object to the Hub with the LFS protocol.

    Defers to [`~huggingface_hub.utils.lfs.lfs_upload`] for the actual upload logic.

    Can be a No-op if the content of the file is already present on the hub
    large file storage.

    Args:
        operation (`CommitOprationAdd`):
            The add operation triggering this upload
        lfs_batch_action (`dict`):
            Upload instructions from the LFS batch endpoint for this object.
            See [`~huggingface_hub.utils.lfs..post_lfs_batch_info`] for more details.
        token (`str`):
            A [user access token](https://hf.co/settings/tokens) to authenticate requests against the Hub

    Raises: `ValueError` if `lfs_batch_action` is improperly formatted
    """
    _validate_batch_actions(lfs_batch_action)
    upload_info = operation._upload_info()
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
        lfs_upload(
            fileobj=fileobj,
            upload_action=upload_action,
            verify_action=verify_action,
            upload_info=upload_info,
            token=token,
        )
        logger.debug(f"{operation.path_in_repo}: Upload successful")


def validate_preupload_info(preupload_info: dict):
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


def fetch_upload_modes(
    additions: Iterable[CommitOperationAdd],
    repo_type: str,
    repo_id: str,
    token: str,
    revision: str,
    endpoint: Optional[str] = None,
    create_pr: Optional[bool] = None,
) -> List[Tuple[CommitOperationAdd, UploadMode]]:
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
        token (`str`):
            An authentication token ( See https://huggingface.co/settings/tokens )
        revision (`str`):
            The git revision to upload the files to. Can be any valid git revision.

    Returns:
        list of 2-tuples, the first element being the add operation and
        the second element the associated upload mode

    Raises:
        :class:`requests.HTTPError`:
            If the Hub API returned an error
        :class:`ValueError`:
            If the Hub API returned an HTTP 400 error (bad request)
    """
    endpoint = endpoint if endpoint is not None else ENDPOINT
    headers = {"authorization": f"Bearer {token}"} if token is not None else None
    payload = {
        "files": [
            {
                "path": op.path_in_repo,
                "sample": base64.b64encode(op._upload_info().sample).decode("ascii"),
                "size": op._upload_info().size,
                "sha": op._upload_info().sha256.hex(),
            }
            for op in additions
        ]
    }

    resp = requests.post(
        f"{endpoint}/api/{repo_type}s/{repo_id}/preupload/{revision}",
        json=payload,
        headers=headers,
        params={"create_pr": "1"} if create_pr else None,
    )
    _raise_convert_bad_request(resp, endpoint_name="preupload")

    preupload_info = validate_preupload_info(resp.json())

    path2mode: Dict[str, UploadMode] = {
        file["path"]: file["uploadMode"] for file in preupload_info["files"]
    }

    return [(op, path2mode[op.path_in_repo]) for op in additions]


def prepare_commit_payload(
    additions: Iterable[Tuple[CommitOperationAdd, UploadMode]],
    deletions: Iterable[CommitOperationDelete],
    commit_message: str,
    commit_description: Optional[str] = None,
    parent_commit: Optional[str] = None,
):
    """
    Builds the payload to POST to the `/commit` API of the Hub
    """
    commit_description = commit_description if commit_description is not None else ""

    payload = {
        **({"parentCommit": parent_commit} if parent_commit is not None else {}),
        "summary": commit_message,
        "description": commit_description,
        "files": [
            {
                "path": add_op.path_in_repo,
                "encoding": "base64",
                "content": add_op.b64content().decode(),
            }
            for (add_op, upload_mode) in additions
            if upload_mode == "regular"
        ],
        "lfsFiles": [
            {
                "path": add_op.path_in_repo,
                "algo": "sha256",
                "oid": add_op._upload_info().sha256.hex(),
            }
            for (add_op, upload_mode) in additions
            if upload_mode == "lfs"
        ],
        "deletedFiles": [{"path": del_op.path_in_repo} for del_op in deletions],
    }
    return payload
