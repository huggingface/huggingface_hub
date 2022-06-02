"""
Type definitions and utilities for the `create_commit` API
"""

import base64
import io
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union


if sys.version_info >= (3, 8):
    from typing import Literal, TypedDict
else:
    from typing_extensions import Literal, TypedDict

import requests
from requests.auth import HTTPBasicAuth

from .constants import ENDPOINT, REPO_TYPES_URL_PREFIXES
from .lfs import (
    LfsBatchObject,
    LfsBatchObjectError,
    LfsBatchResponse,
    UploadInfo,
    upload_lfs_file,
)
from .utils import logging


logger = logging.get_logger(__name__)


UploadMode = Literal["lfs", "regular"]


@dataclass
class CommitOperationDelete:
    """
    Data strcture holding necessary info to delete
    a file from a repository on the HF Hub
    """

    path_in_repo: str
    """
    Path of the file to delete in the repo
    """


@dataclass
class CommitOperationAdd:
    """
    Data structire holding necessary info to upload a file
    to a repository on the HF Hub
    """

    path_in_repo: str
    """
    Path in the repository where the uploaded file will be saved
    """
    path_or_fileobj: Union[str, bytes, BinaryIO]
    """
    Either:
        - a path to a local file (as str) to upload
        - a buffer of bytes (``bytes``) holding the content of the file to upload
        - a "file object" (subclass of ``io.BufferedIOBase``), typically obtained
            with `open(path, "rb")`
    """

    _upload_info: Optional[UploadInfo] = field(default=None, init=False)

    def validate(self):
        """
        Ensures ``path_or_fileobj`` is valid

        Raises:
            ``ValueError``
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

    def upload_info(self) -> UploadInfo:
        """
        Computes and caches UploadInfo for the underlying data behind ``path_or_fileobj``
        Triggers ``self.validate``.

        Raises:
            ValueError: if self.validate fails
        """
        self.validate()
        if self._upload_info is None:
            if isinstance(self.path_or_fileobj, str):
                self._upload_info = UploadInfo.from_path(self.path_or_fileobj)
            elif isinstance(self.path_or_fileobj, bytes):
                self._upload_info = UploadInfo.from_bytes(self.path_or_fileobj)
            else:
                self._upload_info = UploadInfo.from_readable(self.path_or_fileobj)
        return self._upload_info

    @contextmanager
    def fileobj(self):
        self.validate()
        if isinstance(self.path_or_fileobj, str):
            with open(self.path_in_repo, "rb") as file:
                yield file
        elif isinstance(self.path_or_fileobj, bytes):
            yield io.BytesIO(self.path_or_fileobj)
        elif isinstance(self.path_or_fileobj, io.BufferedIOBase):
            yield self.path_or_fileobj
            self.path_or_fileobj.seek(0, io.SEEK_SET)

    def b64content(self) -> bytes:
        """
        The base64-encoded content of ``path_or_fileobj``

        Return:
            ``bytes``
        """
        if isinstance(self.path_or_fileobj, str):
            with open(self.path_or_fileobj, "rb") as reader:
                return base64.b64encode(reader.read())
        elif isinstance(self.path_or_fileobj, bytes):
            return base64.b64encode(self.path_or_fileobj)
        else:
            return base64.b64encode(self.path_or_fileobj.read())


CommitOperation = Union[CommitOperationAdd, CommitOperationDelete]


def base_url(repo_type: str, repo_id: str, endpoint: Optional[str] = None) -> str:
    endpoint = endpoint if endpoint is not None else ENDPOINT
    prefix = ""
    if repo_type in REPO_TYPES_URL_PREFIXES:
        prefix = REPO_TYPES_URL_PREFIXES[repo_type]
    return f"{endpoint}/{prefix}{repo_id}"


def upload_lfs_files(
    additions: Iterable[CommitOperationAdd],
    repo_type: str,
    repo_id: str,
    token: str,
    revision: str,
    endpoint: Optional[str] = None,
):
    """
    Uploads the content of ``additions`` to the HF Hub using the large file storage protocol.

    Args:
        additions (``Iterable`` of ``CommitOperationAdd``):
            The files to be uploaded
        repo_type (``str``):
            Type of the repo to upload to: `"model"`, `"dataset"` or `"space"`.
        repo_id (``str``):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        token (``str``):
            An authentication token ( See https://huggingface.co/settings/tokens )
        revision (``str``):
            The git revision to upload the files to. Can be any valid git revision.

    """
    endpoint = endpoint if endpoint is not None else ENDPOINT

    common_headers = {
        "Accept": "application/vnd.git-lfs+json",
        "Content-Type": "application/vnd.git-lfs+json",
    }

    oid2addop = {add_op.upload_info().sha256.hex(): add_op for add_op in additions}
    batch_url = (
        f"{base_url(repo_type, repo_id, endpoint=endpoint)}.git/info/lfs/objects/batch"
    )
    batch_res = requests.post(
        batch_url,
        headers=common_headers,
        json={
            "operation": "upload",
            "transfers": ["basic", "multipart"],
            "objects": [
                {
                    "oid": add_op.upload_info().sha256.hex(),
                    "size": add_op.upload_info().size,
                }
                for add_op in additions
            ],
            "ref": {
                "name": revision,
            },
            "hash_algo": "sha256",
        },
        auth=HTTPBasicAuth("access_token", token),
    )
    batch_res.raise_for_status()

    try:
        payload: LfsBatchResponse = batch_res.json()
        objects = payload.get("objects", [])
        errors: List[LfsBatchObjectError] = [
            obj for obj in objects if obj.get("error") is not None
        ]
        if errors:
            message = "\n".join(
                [
                    f'Encountered error for file with OID {obj["oid"]}:'
                    f' `{obj["error"]["message"]}'
                    for obj in errors
                ]
            )
            raise ValueError(f"LFS batch endpoint returned errors:\n{message}")

        obj: LfsBatchObject
        for obj in objects:
            add_op = oid2addop.get(obj["oid"])
            if add_op is None:
                raise ValueError(f"Unknown OID: {obj['oid']}")
            upload_info = add_op.upload_info()
            upload_action = obj.get("actions", {}).get("upload", None)
            if upload_action is None:
                # The file was already uploaded
                logger.debug(
                    f"Content of file {add_op.path_in_repo} is already present upstream"
                    " - skipping upload"
                )
                continue

            with add_op.fileobj() as fileobj:
                upload_lfs_file(
                    fileobj=fileobj,
                    upload_action=upload_action,
                    upload_info=upload_info,
                )
            logger.debug(f"{add_op.path_in_repo}: Upload successful")

    except KeyError as err:
        raise ValueError("Malformed response from LFS batch endpoint") from err


def _preupload_payload(add_operation: CommitOperationAdd):
    upload_info = add_operation.upload_info()
    return {
        "path": add_operation.path_in_repo,
        "sample": base64.b64encode(upload_info.sample).decode("ascii"),
        "size": upload_info.size,
        "sha": upload_info.sha256.hex(),
    }


class PreUploadFileResponse(TypedDict):
    path: str
    uploadMode: UploadMode


def fetch_upload_modes(
    additions: Iterable[CommitOperationAdd],
    repo_type: str,
    repo_id: str,
    token: str,
    revision: str,
    endpoint: Optional[str] = None,
) -> List[Tuple[CommitOperationAdd, UploadMode]]:
    """
    Requests the HF Hub to determine wether each input file should be
    uploaded as a regular git blob or as git LFS blob.

    Args:
        additions (``Iterable`` of :class:`CommitOperationAdd`):
            Iterable of :class:`CommitOperationAdd` describing the files to
            upload to the HF hub.
        repo_type (``str``):
            Type of the repo to upload to: `"model"`, `"dataset"` or `"space"`.
        repo_id (``str``):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        token (``str``):
            An authentication token ( See https://huggingface.co/settings/tokens )
        revision (``str``):
            The git revision to upload the files to. Can be any valid git revision.

    Returns:
        list of 2-tuples, the first element being the add operation and
        the second element the associated upload mode

    Raises:
        :class:`requests.HTTPError`:
            If the HF Hub API returned an error
    """
    endpoint = endpoint if endpoint is not None else ENDPOINT
    headers = {"authorization": f"Bearer {token}"} if token is not None else None
    payload = {"files": [_preupload_payload(op) for op in additions]}

    resp = requests.post(
        f"{endpoint}/api/{repo_type}s/{repo_id}/preupload/{revision}",
        json=payload,
        headers=headers,
    )
    resp.raise_for_status()

    preupload_info: List[PreUploadFileResponse] = resp.json().get("files", [])
    path2mode: Dict[str, UploadMode] = {
        file["path"]: file["uploadMode"] for file in preupload_info
    }

    return [(op, path2mode[op.path_in_repo]) for op in additions]


class CommitFilesPayloadFile(TypedDict):
    path: str
    encoding: Literal["base64"]
    content: str


class CommitFilesPayloadLfsFile(TypedDict):
    path: str
    algo: Literal["sha256"]
    oid: str


class CommitFilesPayloadDeletedFile(TypedDict):
    path: str


class CommitFilesPayload(TypedDict):
    summary: str
    description: str
    files: List[CommitFilesPayloadFile]
    lfsFiles: List[CommitFilesPayloadLfsFile]
    deletedFiles: List[CommitFilesPayloadDeletedFile]


def prepare_commit_payload(
    additions: Iterable[Tuple[CommitOperationAdd, UploadMode]],
    deletions: Iterable[CommitOperationDelete],
    commit_summary: str,
    commit_description: Optional[str] = None,
) -> CommitFilesPayload:
    """
    Builds the payload to pass the the `commit` API of the HF Hub
    """
    commit_description = commit_description if commit_description is not None else ""

    return {
        "summary": commit_summary,
        "description": commit_description,
        "files": [
            {
                "path": add_op.path_in_repo,
                "encoding": "base64",
                "content": add_op.b64content.decode(),
            }
            for (add_op, upload_mode) in additions
            if upload_mode == "regular"
        ],
        "lfsFiles": [
            {
                "path": add_op.path_in_repo,
                "algo": "sha256",
                "oid": add_op.upload_info().sha256.hex(),
            }
            for (add_op, upload_mode) in additions
            if upload_mode == "lfs"
        ],
        "deletedFiles": [{"path": del_op.path_in_repo} for del_op in deletions],
    }
