import base64
import io
import os
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from math import ceil
from os.path import getsize
from typing import Dict, Generator, Iterable, List, Literal, Optional, TypedDict, Union

import requests
from requests.auth import HTTPBasicAuth

from .constants import ENDPOINT, REPO_TYPES_URL_PREFIXES
from .utils import logging


logger = logging.get_logger(__name__)


UploadMode = Literal["lfs", "regular"]


@dataclass
class CommitOperationDelete:
    path_in_repo: str


@dataclass
class CommitOperationAdd:
    path_in_repo: str
    path_or_fileobj: Union[str, bytes, io.BufferedIOBase]

    def __post_init__(self):
        if isinstance(self.path_or_fileobj, str):
            path_or_fileobj = os.path.normpath(os.path.expanduser(self.path_or_fileobj))
            if not os.path.isfile(path_or_fileobj):
                raise ValueError(f"Provided path: '{path_or_fileobj}' is not a file")
        elif not isinstance(self.path_or_fileobj, (io.BufferedIOBase, bytes)):
            # ^^ Test from: https://stackoverflow.com/questions/44584829/how-to-determine-if-file-is-opened-in-binary-or-text-mode
            raise ValueError(
                "path_or_fileobj must be either an instance of str, bytes or BinaryIO."
                " If you passed a file-like object, make sure it is in binary mode."
            )

    def iter_content(
        self, chunk_size: Optional[int] = None
    ) -> Generator[bytes, None, None]:
        if isinstance(self.path_or_fileobj, bytes):
            if chunk_size is None:
                yield self.path_or_fileobj
            else:
                idx = 0
                while chunk := self.path_or_fileobj[idx : idx + chunk_size]:
                    yield chunk
        elif isinstance(self.path_or_fileobj, str):
            with open(self.path_or_fileobj, "rb") as reader:
                while chunk := reader.read(chunk_size):
                    yield chunk
        else:
            while chunk := self.path_or_fileobj.read(chunk_size or -1):
                yield chunk


CommitOperation = Union[CommitOperationAdd, CommitOperationDelete]


@dataclass
class CommitOperationAddAnnotated(CommitOperationAdd):
    size: int
    sha: str
    upload_mode: UploadMode

    @property
    def b64content(self) -> bytes:
        if isinstance(self.path_or_fileobj, str):
            with open(self.path_or_fileobj, "rb") as reader:
                return base64.b64encode(reader.read())
        elif isinstance(self.path_or_fileobj, bytes):
            return base64.b64encode(self.path_or_fileobj)
        else:
            return base64.b64encode(self.path_or_fileobj.read())


class LfsAction(TypedDict, total=False):
    href: str
    header: Dict[str, str]
    expires_in: int
    expires_at: str


class LfsResponseActions(TypedDict, total=False):
    download: LfsAction
    upload: LfsAction
    verify: LfsAction


class LfsResponseError(TypedDict):
    code: int
    message: str


class LfsResponseObjectBase(TypedDict):
    oid: str
    size: int


class LfsResponseObjectSuccess(LfsResponseObjectBase):
    actions: LfsResponseActions


class LfsResponseObjectError(LfsResponseObjectBase):
    error: LfsResponseError


HASH_CHUNK_SIZE = 512


def sha_iter(iterable: Iterable[bytes]):
    sha = sha256()
    for chunk in iterable:
        sha.update(chunk)
    return sha.digest()


def sha_fileobj(fileobj: io.BufferedIOBase, chunk_size: Optional[int] = None) -> bytes:
    """
    Computes the sha256 hash of the given file object, by chunks of size `chunk_size`.

    Args:
        fileobj (`BinaryIO`):
            The File object to compute sha256 for, typically obtained with `open`
        chunk_size (`int`, *optional*):
            The number of bytes to read from `fileobj` at once, defaults to 512

    Returns:
        `bytes`: `fileobj`'s sha256 hash as bytes
    """
    chunk_size = chunk_size if chunk_size is not None else HASH_CHUNK_SIZE
    return sha_iter(iter(partial(fileobj.read, chunk_size), b""))


def base_url(repo_type: str, repo_id: str, endpoint: Optional[str] = None) -> str:
    endpoint = endpoint if endpoint is not None else ENDPOINT
    prefix = ""
    if repo_type in REPO_TYPES_URL_PREFIXES:
        prefix = REPO_TYPES_URL_PREFIXES[repo_type]
    return f"{endpoint}/{prefix}{repo_id}"


def upload_lfs_files(
    files: Iterable[CommitOperationAddAnnotated],
    repo_type: str,
    repo_id: str,
    token: str,
    revision: str,
    endpoint: Optional[str] = None,
):
    """
    Handles the optional upload of `files` to HF Hub large file storage.
    Ignores files with `"regular"` `upload_mode`.

    Args:
        files (``Iterable`` of ``FileUpload``):
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

    oid2file = {file.sha: file for file in files}
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
                    "oid": file.sha,
                    "size": file.size,
                }
                for file in oid2file.values()
                if file.upload_mode == "lfs"
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
        objects = batch_res.json()["objects"]
        errors: List[LfsResponseObjectError] = [
            obj for obj in objects if "error" in obj
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

        obj: LfsResponseObjectSuccess
        for obj in objects:
            file = oid2file.get(obj["oid"])
            if file is None:
                raise ValueError(f"Unknown OID: {obj['oid']}")
            upload_action = obj.get("actions", {}).get("upload", None)
            if upload_action is None:
                # The file was already uploaded
                logger.debug(
                    f"Content of file {file.path_in_repo} is already present upstream -"
                    " skipping upload"
                )
                continue

            chunk_size = upload_action.get("header", {}).get("chunk_size", None)
            if chunk_size is not None:
                logger.debug(f"Starting multi-part upload for file {file.path_in_repo}")
                if isinstance(chunk_size, str):
                    chunk_size = int(chunk_size, 10)
                else:
                    raise ValueError(
                        "Malformed response from LFS batch endpoint: `chunk_size`"
                        " should be a string"
                    )
                parts = {
                    key: value
                    for key, value in upload_action["header"].items()
                    if key.isdigit() and len(key) > 0
                }
                if len(parts) != ceil(file.size / chunk_size):
                    raise ValueError("Invalid server response to upload large LFS file")
                completion_url = upload_action["href"]
                completion_body = {
                    "oid": file.sha,
                    "parts": [
                        {
                            "partNumber": int(part_num),
                            "etag": "",
                        }
                        for part_num in parts
                    ],
                }
                content_iter = file.iter_content(chunk_size)
                for idx, part_url in enumerate(parts.values()):
                    logger.debug(
                        f"{file.path_in_repo}: Uploadig part {idx} of {len(parts)}"
                    )

                    part_res = requests.put(part_url, data=next(content_iter))
                    part_res.raise_for_status()
                    completion_body["parts"][idx]["etag"] = part_res.headers.get(
                        "ETag", ""
                    )
                completion_res = requests.post(
                    completion_url, json=completion_body, headers=common_headers
                )
                completion_res.raise_for_status()
            else:
                logger.debug(
                    f"Starting single-part upload for file {file.path_in_repo}"
                )
                data = next(file.iter_content(None))
                upload_res = requests.put(upload_action["href"], data=data)
                upload_res.raise_for_status()
            logger.debug(f"{file.path_in_repo}: Upload successful")

    except KeyError as err:
        raise ValueError("Malformed response from LFS batch endpoint") from err


class PreUploadFileResponse(TypedDict):
    path: str
    uploadMode: UploadMode


class PreUploadPayload(TypedDict):
    sample: str
    size: int
    path: str
    sha: str


def preupload_payload(file: CommitOperationAdd) -> PreUploadPayload:
    if isinstance(file.path_or_fileobj, str):
        size = getsize(file.path_or_fileobj)
        with io.open(file.path_or_fileobj, "rb") as reader:
            sample = reader.peek(512)
            sha = sha_fileobj(reader)
    elif isinstance(file.path_or_fileobj, bytes):
        size = len(file.path_or_fileobj)
        sample = file.path_or_fileobj[:512]
        sha = sha256(file.path_or_fileobj).digest()
    else:
        sample = file.path_or_fileobj.read(512)
        file.path_or_fileobj.seek(0, io.SEEK_SET)
        sha = sha_fileobj(file.path_or_fileobj)
        size = file.path_or_fileobj.tell()
        file.path_or_fileobj.seek(0, io.SEEK_SET)

    sample = base64.b64encode(sample).decode("ascii")
    return {
        "sample": sample,
        "size": size,
        "path": file.path_in_repo,
        "sha": sha.hex(),
    }


def prepare_file_upload(
    files: Iterable[CommitOperationAdd],
    repo_type: str,
    repo_id: str,
    token: str,
    revision: str,
    endpoint: Optional[str] = None,
) -> List[CommitOperationAddAnnotated]:
    """
    Requests the HF Hub to determine wether each input file should be
    uploaded as a regular git blob or as git LFS blob.

    Args:
        files (``Iterable`` of :class:`FileUploadInput`):
            Iterable of :class:`FileUploadInput` describing the files to
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

    Raises:
        :class:`requests.HTTPError`:
            If the HF Hub API returned an error
    """
    endpoint = endpoint if endpoint is not None else ENDPOINT
    headers = {"authorization": f"Bearer {token}"} if token is not None else None

    payload = [preupload_payload(file) for file in files]
    path2payload = {item["path"]: item for item in payload}

    resp = requests.post(
        f"{endpoint}/api/{repo_type}s/{repo_id}/preupload/{revision}",
        json={"files": payload},
        headers=headers,
    )
    resp.raise_for_status()

    preupload_info: List[PreUploadFileResponse] = resp.json().get("files", [])
    path2mode: Dict[str, UploadMode] = {
        file["path"]: file["uploadMode"] for file in preupload_info
    }

    return [
        CommitOperationAddAnnotated(
            path_or_fileobj=file.path_or_fileobj,
            path_in_repo=file.path_in_repo,
            size=path2payload[file.path_in_repo]["size"],
            sha=path2payload[file.path_in_repo]["sha"],
            upload_mode=path2mode[file.path_in_repo],
        )
        for file in files
    ]


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
    additions: Iterable[CommitOperationAddAnnotated],
    deletions: Iterable[CommitOperationDelete],
    commit_summary: str,
    commit_description: Optional[str] = None,
) -> CommitFilesPayload:
    commit_description = commit_description if commit_description is not None else ""

    return {
        "summary": commit_summary,
        "description": commit_description,
        "files": [
            {
                "path": file.path_in_repo,
                "encoding": "base64",
                "content": file.b64content.decode(),
            }
            for file in additions
            if file.upload_mode == "regular"
        ],
        "lfsFiles": [
            {"path": file.path_in_repo, "algo": "sha256", "oid": file.sha}
            for file in additions
            if file.upload_mode == "lfs"
        ],
        "deletedFiles": [{"path": deletion.path_in_repo} for deletion in deletions],
    }
