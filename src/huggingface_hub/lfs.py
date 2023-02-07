# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
"""Git LFS related type definitions and utilities"""
import io
import os
import re
from contextlib import AbstractContextManager
from dataclasses import dataclass
from math import ceil
from os.path import getsize
from typing import BinaryIO, Iterable, List, Optional, Tuple

import requests
from requests.auth import HTTPBasicAuth

from huggingface_hub.constants import ENDPOINT, REPO_TYPES_URL_PREFIXES

from .utils import (
    get_token_to_send,
    hf_raise_for_status,
    http_backoff,
    validate_hf_hub_args,
)
from .utils._typing import TypedDict
from .utils.sha import sha256, sha_fileobj


OID_REGEX = re.compile(r"^[0-9a-f]{40}$")

LFS_MULTIPART_UPLOAD_COMMAND = "lfs-multipart-upload"

LFS_HEADERS = {
    "Accept": "application/vnd.git-lfs+json",
    "Content-Type": "application/vnd.git-lfs+json",
}


@dataclass
class UploadInfo:
    """
    Dataclass holding required information to determine whether a blob
    should be uploaded to the hub using the LFS protocol or the regular protocol

    Args:
        sha256 (`bytes`):
            SHA256 hash of the blob
        size (`int`):
            Size in bytes of the blob
        sample (`bytes`):
            First 512 bytes of the blob
    """

    sha256: bytes
    size: int
    sample: bytes

    @classmethod
    def from_path(cls, path: str):
        size = getsize(path)
        with io.open(path, "rb") as file:
            sample = file.peek(512)[:512]
            sha = sha_fileobj(file)
        return cls(size=size, sha256=sha, sample=sample)

    @classmethod
    def from_bytes(cls, data: bytes):
        sha = sha256(data).digest()
        return cls(size=len(data), sample=data[:512], sha256=sha)

    @classmethod
    def from_fileobj(cls, fileobj: BinaryIO):
        sample = fileobj.read(512)
        fileobj.seek(0, io.SEEK_SET)
        sha = sha_fileobj(fileobj)
        size = fileobj.tell()
        fileobj.seek(0, io.SEEK_SET)
        return cls(size=size, sha256=sha, sample=sample)


def _validate_lfs_action(lfs_action: dict):
    """validates response from the LFS batch endpoint"""
    if not (
        isinstance(lfs_action.get("href"), str)
        and (lfs_action.get("header") is None or isinstance(lfs_action.get("header"), dict))
    ):
        raise ValueError("lfs_action is improperly formatted")
    return lfs_action


def _validate_batch_actions(lfs_batch_actions: dict):
    """validates response from the LFS batch endpoint"""
    if not (isinstance(lfs_batch_actions.get("oid"), str) and isinstance(lfs_batch_actions.get("size"), int)):
        raise ValueError("lfs_batch_actions is improperly formatted")

    upload_action = lfs_batch_actions.get("actions", {}).get("upload")
    verify_action = lfs_batch_actions.get("actions", {}).get("verify")
    if upload_action is not None:
        _validate_lfs_action(upload_action)
    if verify_action is not None:
        _validate_lfs_action(verify_action)
    return lfs_batch_actions


def _validate_batch_error(lfs_batch_error: dict):
    """validates response from the LFS batch endpoint"""
    if not (isinstance(lfs_batch_error.get("oid"), str) and isinstance(lfs_batch_error.get("size"), int)):
        raise ValueError("lfs_batch_error is improperly formatted")
    error_info = lfs_batch_error.get("error")
    if not (
        isinstance(error_info, dict)
        and isinstance(error_info.get("message"), str)
        and isinstance(error_info.get("code"), int)
    ):
        raise ValueError("lfs_batch_error is improperly formatted")
    return lfs_batch_error


@validate_hf_hub_args
def post_lfs_batch_info(
    upload_infos: Iterable[UploadInfo],
    token: Optional[str],
    repo_type: str,
    repo_id: str,
    endpoint: Optional[str] = None,
) -> Tuple[List[dict], List[dict]]:
    """
    Requests the LFS batch endpoint to retrieve upload instructions

    Learn more: https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md

    Args:
        upload_infos (`Iterable` of `UploadInfo`):
            `UploadInfo` for the files that are being uploaded, typically obtained
            from `CommitOperationAdd.upload_info`
        repo_type (`str`):
            Type of the repo to upload to: `"model"`, `"dataset"` or `"space"`.
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        token (`str`, *optional*):
            An authentication token ( See https://huggingface.co/settings/tokens )

    Returns:
        `LfsBatchInfo`: 2-tuple:
            - First element is the list of upload instructions from the server
            - Second element is an list of errors, if any

    Raises:
        `ValueError`: If an argument is invalid or the server response is malformed

        `HTTPError`: If the server returned an error
    """
    endpoint = endpoint if endpoint is not None else ENDPOINT
    url_prefix = ""
    if repo_type in REPO_TYPES_URL_PREFIXES:
        url_prefix = REPO_TYPES_URL_PREFIXES[repo_type]
    batch_url = f"{endpoint}/{url_prefix}{repo_id}.git/info/lfs/objects/batch"
    resp = requests.post(
        batch_url,
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        },
        json={
            "operation": "upload",
            "transfers": ["basic", "multipart"],
            "objects": [
                {
                    "oid": upload.sha256.hex(),
                    "size": upload.size,
                }
                for upload in upload_infos
            ],
            "hash_algo": "sha256",
        },
        auth=HTTPBasicAuth(
            "access_token",
            get_token_to_send(token or True),  # type: ignore  # Token must be provided or retrieved
        ),
    )
    hf_raise_for_status(resp)
    batch_info = resp.json()

    objects = batch_info.get("objects", None)
    if not isinstance(objects, list):
        raise ValueError("Malformed response from server")

    return (
        [_validate_batch_actions(obj) for obj in objects if "error" not in obj],
        [_validate_batch_error(obj) for obj in objects if "error" in obj],
    )


def lfs_upload(
    fileobj: BinaryIO,
    upload_info: UploadInfo,
    upload_action: dict,
    verify_action: Optional[dict],
    token: Optional[str],
):
    """
    Uploads a file using the git lfs protocol and determines automatically whether or not
    to use the multipart transfer protocol

    Args:
        fileobj (file-like object):
            The content of the file to upload
        upload_info (`UploadInfo`):
            Upload info for `fileobj`
        upload_action (`dict`):
            The `upload` action from the LFS Batch endpoint. Must contain
            a `href` field, and optionally a `header` field.
        verify_action (`dict`):
            The `verify` action from the LFS Batch endpoint. Must contain
            a `href` field, and optionally a `header` field. The `href` URL will
            be called after a successful upload.
        token (`str`, *optional*):
            A [user access token](https://hf.co/settings/tokens) to authenticate requests
            against the Hub.

    Returns:
        `requests.Response`:
            the response from the completion request in case of a multi-part upload, and the
            response from the single PUT request in case of a single-part upload

    Raises:
        `ValueError`: if some objects / responses are malformed

        `requests.HTTPError`
    """
    _validate_lfs_action(upload_action)
    if verify_action is not None:
        _validate_lfs_action(verify_action)

    header = upload_action.get("header", {})
    chunk_size = header.get("chunk_size", None)
    if chunk_size is not None:
        if isinstance(chunk_size, str):
            chunk_size = int(chunk_size, 10)
        else:
            raise ValueError("Malformed response from LFS batch endpoint: `chunk_size` should be a string")
        _upload_multi_part(
            completion_url=upload_action["href"],
            fileobj=fileobj,
            chunk_size=chunk_size,
            header=header,
            upload_info=upload_info,
        )
    else:
        _upload_single_part(
            upload_url=upload_action["href"],
            fileobj=fileobj,
        )
    if verify_action is not None:
        verify_resp = requests.post(
            verify_action["href"],
            auth=HTTPBasicAuth(
                username="USER",
                # Token must be provided or retrieved
                password=get_token_to_send(token or True),  # type: ignore
            ),
            json={"oid": upload_info.sha256.hex(), "size": upload_info.size},
        )
        hf_raise_for_status(verify_resp)


def _upload_single_part(upload_url: str, fileobj: BinaryIO):
    """
    Uploads `fileobj` as a single PUT HTTP request (basic LFS transfer protocol)

    Args:
        upload_url (`str`):
            The URL to PUT the file to.
        fileobj:
            The file-like object holding the data to upload.

    Returns: `requests.Response`

    Raises: `requests.HTTPError` if the upload resulted in an error
    """
    upload_res = http_backoff("PUT", upload_url, data=fileobj)
    hf_raise_for_status(upload_res)
    return upload_res


class PayloadPartT(TypedDict):
    partNumber: int
    etag: str


class CompletionPayloadT(TypedDict):
    """Payload that will be sent to the Hub when uploading multi-part."""

    oid: str
    parts: List[PayloadPartT]


def _upload_multi_part(
    completion_url: str,
    fileobj: BinaryIO,
    header: dict,
    chunk_size: int,
    upload_info: UploadInfo,
):
    """
    Uploads `fileobj` using HF multipart LFS transfer protocol.

    Args:
        completion_url (`str`):
            The URL to GET after completing all parts uploads.
        fileobj:
            The file-like object holding the data to upload.
        header (`dict`):
            The `header` field from the `upload` action from the LFS
            Batch endpoint response
        chunk_size (`int`):
            The size in bytes of the parts. `fileobj` will be uploaded in parts
            of `chunk_size` bytes (except for the last part who can be smaller)
        upload_info (`UploadInfo`):
            `UploadInfo` for `fileobj`.

    Returns: `requests.Response`: The response from requesting `completion_url`.

    Raises: `requests.HTTPError` if uploading any of the parts resulted in an error.

    Raises: `requests.HTTPError` if requesting `completion_url` resulted in an error.

    """
    sorted_part_upload_urls = [
        upload_url
        for _, upload_url in sorted(
            [
                (int(part_num, 10), upload_url)
                for part_num, upload_url in header.items()
                if part_num.isdigit() and len(part_num) > 0
            ],
            key=lambda t: t[0],
        )
    ]
    num_parts = len(sorted_part_upload_urls)
    if num_parts != ceil(upload_info.size / chunk_size):
        raise ValueError("Invalid server response to upload large LFS file")

    completion_payload: CompletionPayloadT = {
        "oid": upload_info.sha256.hex(),
        "parts": [
            {
                "partNumber": idx + 1,
                "etag": "",
            }
            for idx in range(num_parts)
        ],
    }

    for part_idx, part_upload_url in enumerate(sorted_part_upload_urls):
        with SliceFileObj(
            fileobj,
            seek_from=chunk_size * part_idx,
            read_limit=chunk_size,
        ) as fileobj_slice:
            part_upload_res = http_backoff("PUT", part_upload_url, data=fileobj_slice)
            hf_raise_for_status(part_upload_res)
            etag = part_upload_res.headers.get("etag")
            if etag is None or etag == "":
                raise ValueError(f"Invalid etag (`{etag}`) returned for part {part_idx +1} of {num_parts}")
            completion_payload["parts"][part_idx]["etag"] = etag

    completion_res = requests.post(
        completion_url,
        json=completion_payload,
        headers=LFS_HEADERS,
    )
    hf_raise_for_status(completion_res)
    return completion_res


class SliceFileObj(AbstractContextManager):
    """
    Utility context manager to read a *slice* of a seekable file-like object as a seekable, file-like object.

    This is NOT thread safe

    Inspired by stackoverflow.com/a/29838711/593036

    Credits to @julien-c

    Args:
        fileobj (`BinaryIO`):
            A file-like object to slice. MUST implement `tell()` and `seek()` (and `read()` of course).
            `fileobj` will be reset to its original position when exiting the context manager.
        seek_from (`int`):
            The start of the slice (offset from position 0 in bytes).
        read_limit (`int`):
            The maximum number of bytes to read from the slice.

    Attributes:
        previous_position (`int`):
            The previous position

    Examples:

    Reading 200 bytes with an offset of 128 bytes from a file (ie bytes 128 to 327):
    ```python
    >>> with open("path/to/file", "rb") as file:
    ...     with SliceFileObj(file, seek_from=128, read_limit=200) as fslice:
    ...         fslice.read(...)
    ```

    Reading a file in chunks of 512 bytes
    ```python
    >>> import os
    >>> chunk_size = 512
    >>> file_size = os.getsize("path/to/file")
    >>> with open("path/to/file", "rb") as file:
    ...     for chunk_idx in range(ceil(file_size / chunk_size)):
    ...         with SliceFileObj(file, seek_from=chunk_idx * chunk_size, read_limit=chunk_size) as fslice:
    ...             chunk = fslice.read(...)

    ```
    """

    def __init__(self, fileobj: BinaryIO, seek_from: int, read_limit: int):
        self.fileobj = fileobj
        self.seek_from = seek_from
        self.read_limit = read_limit

    def __enter__(self):
        self._previous_position = self.fileobj.tell()
        end_of_stream = self.fileobj.seek(0, os.SEEK_END)
        self._len = min(self.read_limit, end_of_stream - self.seek_from)
        # ^^ The actual number of bytes that can be read from the slice
        self.fileobj.seek(self.seek_from, io.SEEK_SET)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fileobj.seek(self._previous_position, io.SEEK_SET)

    def read(self, n: int = -1):
        pos = self.tell()
        if pos >= self._len:
            return b""
        remaining_amount = self._len - pos
        data = self.fileobj.read(remaining_amount if n < 0 else min(n, remaining_amount))
        return data

    def tell(self) -> int:
        return self.fileobj.tell() - self.seek_from

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        start = self.seek_from
        end = start + self._len
        if whence in (os.SEEK_SET, os.SEEK_END):
            offset = start + offset if whence == os.SEEK_SET else end + offset
            offset = max(start, min(offset, end))
            whence = os.SEEK_SET
        elif whence == os.SEEK_CUR:
            cur_pos = self.fileobj.tell()
            offset = max(start - cur_pos, min(offset, end - cur_pos))
        else:
            raise ValueError(f"whence value {whence} is not supported")
        return self.fileobj.seek(offset, whence) - self.seek_from

    def __iter__(self):
        yield self.read(n=4 * 1024 * 1024)
