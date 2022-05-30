import base64
import io
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from huggingface_hub.utils import logging
from math import ceil
from os.path import getsize
from typing import BinaryIO, Dict, Iterable, List, Literal, Optional, TypedDict

import requests
from requests.auth import HTTPBasicAuth
from huggingface_hub.constants import ENDPOINT, REPO_TYPES_URL_PREFIXES

logger = logging.get_logger(__name__)


UploadMode = Literal["lfs", "regular"]


@dataclass
class FileUploadInput:
    local_path: str
    remote_path: str


@dataclass
class FileUpload(FileUploadInput):
    size: int
    sha: str
    upload_mode: UploadMode

    @property
    def b64content(self) -> bytes:
        with open(self.local_path, "rb") as reader:
            return base64.b64encode(reader.read())


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


def sha_fileobj(fileobj: BinaryIO, chunk_size: Optional[int] = None) -> bytes:
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
    sha = sha256()
    for chunk in iter(partial(fileobj.read, chunk_size), b""):
        sha.update(chunk)
    return sha.digest()


def base_url(repo_type: str, repo_id: str, endpoint: Optional[str] = None) -> str:
    endpoint = endpoint if endpoint is not None else ENDPOINT
    prefix = ""
    if repo_type in REPO_TYPES_URL_PREFIXES:
        prefix = REPO_TYPES_URL_PREFIXES[repo_type]
    return f"{endpoint}/{prefix}{repo_id}"


def upload_lfs_files(
    files: Iterable[FileUpload],
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
                    f"Content of file {file.remote_path} is already present upstream - skipping"
                    " upload"
                )
                continue

            chunk_size = upload_action.get("header", {}).get("chunk_size", None)
            if chunk_size is not None:
                logger.debug(f"Starting multi-part upload for file {file.remote_path}")
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
                with open(file.local_path, "rb") as data:
                    for idx, part_url in enumerate(parts.values()):
                        logger.debug(
                            f"{file.remote_path}: Uploadig part {idx} of {len(parts)}"
                        )

                        part_res = requests.put(part_url, data=data.read(chunk_size))
                        part_res.raise_for_status()
                        completion_body["parts"][idx]["etag"] = part_res.headers.get(
                            "ETag", ""
                        )
                completion_res = requests.post(
                    completion_url, json=completion_body, headers=common_headers
                )
                completion_res.raise_for_status()
            else:
                logger.debug(f"Starting single-part upload for file {file.remote_path}")

                with open(file.local_path, "rb") as data:
                    upload_res = requests.put(upload_action["href"], data=data)
                upload_res.raise_for_status()
            logger.debug(f"{file.remote_path}: Upload successful")

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


def preupload_payload(file: FileUploadInput) -> PreUploadPayload:
    size = getsize(file.local_path)
    with io.open(file.local_path, "rb") as reader:
        sample = reader.peek(512)
        sha = sha_fileobj(reader)
    sample = base64.b64encode(sample).decode("ascii")
    return {
        "sample": sample,
        "size": size,
        "path": file.remote_path,
        "sha": sha.hex(),
    }


def prepare_file_upload(
    files: Iterable[FileUploadInput],
    repo_type: str,
    repo_id: str,
    token: str,
    revision: str,
    endpoint: Optional[str] = None,
) -> List[FileUpload]:
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
        FileUpload(
            local_path=file.local_path,
            remote_path=file.remote_path,
            size=path2payload[file.remote_path]["size"],
            sha=path2payload[file.remote_path]["sha"],
            upload_mode=path2mode[file.remote_path],
        )
        for file in files
    ]


class CommitFilesPayloadFileEntryRegular(TypedDict):
    path: str
    encoding: Literal["base64"]
    content: str


class CommitFilesPayloadFileEntryLFS(TypedDict):
    path: str
    algo: Literal["sha256"]
    oid: str


class CommitFilesPayload(TypedDict):
    summary: str
    description: str
    files: List[CommitFilesPayloadFileEntryRegular]
    lfsFiles: List[CommitFilesPayloadFileEntryLFS]


def prepare_commit_payload(
    files: Iterable[FileUpload],
    commit_summary: str,
    commit_description: Optional[str] = None,
) -> CommitFilesPayload:
    """TODO: find a better name"""
    commit_description = commit_description if commit_description is not None else ""

    return {
        "summary": commit_summary,
        "description": commit_description,
        "files": [
            {
                "path": file.remote_path,
                "encoding": "base64",
                "content": file.b64content.decode(),
            }
            for file in files
            if file.upload_mode == "regular"
        ],
        "lfsFiles": [
            {"path": file.remote_path, "algo": "sha256", "oid": file.sha}
            for file in files
            if file.upload_mode == "lfs"
        ],
    }
