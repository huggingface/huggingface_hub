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
import subprocess
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass
from math import ceil
from os.path import getsize
from typing import BinaryIO, Iterable, List, Optional, Tuple

import requests
from huggingface_hub.constants import ENDPOINT, REPO_TYPES_URL_PREFIXES
from requests.auth import HTTPBasicAuth

from .utils.sha import sha256, sha_fileobj


OID_REGEX = re.compile(r"^[0-9a-f]{40}$")

LFS_MULTIPART_UPLOAD_COMMAND = "lfs-multipart-upload"


def install_lfs_in_userspace():
    """
    If in Linux, installs git-lfs in userspace (sometimes useful if you can't
    `sudo apt install` or equivalent).
    """
    if sys.platform != "linux":
        raise ValueError("Only implemented for Linux right now")
    GIT_LFS_TARBALL = "https://github.com/git-lfs/git-lfs/releases/download/v2.13.1/git-lfs-linux-amd64-v2.13.1.tar.gz"
    CWD = os.path.join(os.getcwd(), "install_lfs")
    os.makedirs(CWD, exist_ok=True)
    subprocess.run(
        ["wget", "-O", "tarball.tar.gz", GIT_LFS_TARBALL], check=True, cwd=CWD
    )
    subprocess.run(["tar", "-xvzf", "tarball.tar.gz"], check=True, cwd=CWD)
    subprocess.run(["bash", "install.sh"], check=True, cwd=CWD)


LFS_HEADERS = {
    "Accept": "application/vnd.git-lfs+json",
    "Content-Type": "application/vnd.git-lfs+json",
}


@dataclass
class UploadInfo:
    """
    Dataclass holding required information to determine wether a blob
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


def validate_batch_actions(lfs_batch_actions: dict):
    if not (
        isinstance(lfs_batch_actions.get("oid"), str)
        and isinstance(lfs_batch_actions.get("size"), int)
    ):
        raise ValueError("lfs_batch_actions is improperly formatted")

    upload_action = lfs_batch_actions.get("actions", {}).get("upload")
    if upload_action is not None:
        validate_lfs_upload_action(upload_action)
    return lfs_batch_actions


def validate_batch_error(lfs_batch_error: dict):
    if not (
        isinstance(lfs_batch_error.get("oid"), str)
        and isinstance(lfs_batch_error.get("size"), int)
    ):
        raise ValueError("lfs_batch_error is improperly formatted")
    error_info = lfs_batch_error.get("error")
    if not (
        isinstance(error_info, dict)
        and isinstance(error_info.get("message"), str)
        and isinstance(error_info.get("code"), int)
    ):
        raise ValueError("lfs_batch_error is improperly formatted")
    return lfs_batch_error


def validate_lfs_upload_action(lfs_upload_action: dict):
    if not (
        isinstance(lfs_upload_action.get("href"), str)
        and (
            lfs_upload_action.get("header") is None
            or isinstance(lfs_upload_action.get("header"), dict)
        )
    ):
        raise ValueError("lfs_upload_action is improperly formatted")
    return lfs_upload_action


def post_lfs_batch_info(
    upload_infos: Iterable[UploadInfo],
    token: str,
    repo_type: str,
    repo_id: str,
    revision: str,
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
        token (`str`):
            An authentication token ( See https://huggingface.co/settings/tokens )
        revision (`str`):
            The git revision to upload the files to. Can be any valid git revision.

    Returns:
        `LfsBatchInfo`: 2-tuple:
            - First element is the list of upload instructions from the server
            - Second element is an list of errors, if any

    Raises:
        `ValueError`: If an argument is invalid or the server response is malformed

        `HTTPError`: If the server returned an error
    """
    endpoint = endpoint if endpoint is not None else ENDPOINT
    if repo_type not in REPO_TYPES_URL_PREFIXES:
        raise ValueError(
            "Invalid value for `repo_type`, must be one of"
            f" {tuple(REPO_TYPES_URL_PREFIXES.keys())}"
        )
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
            "ref": {
                "name": revision,
            },
            "hash_algo": "sha256",
        },
        auth=HTTPBasicAuth("access_token", token),
    )
    resp.raise_for_status()
    batch_info = resp.json()

    objects = batch_info.get("objects", None)
    if not isinstance(objects, list):
        raise ValueError("Malformed response from server")

    return (
        [validate_batch_actions(obj) for obj in objects if "error" not in obj],
        [validate_batch_error(obj) for obj in objects if "error" in obj],
    )


def lfs_upload(
    fileobj: BinaryIO,
    upload_action: dict,
    upload_info: UploadInfo,
):
    """
    Uploads a file using the git lfs protocol and determines automatically whether or not
    to use the multipart transfer protocol

    Args:
        fileobj (file-like object):
            The content of the file to upload
        upload_action (`dict`):
            The `upload` action from the LFS Batch endpoint. Must contain
            a `href` field, and optionally a `header` field
        uplod_info (`UploadInfo`):
            Upload info for `fileobj`

    Returns:
        `requests.Response`:
            the repsonse from the completion request in case of amulti-part upload, and the
            response from the single PUT request in case of a single-part upload

    Raises:
        `ValueError`: if some objects / repsonses are malformed

        `requests.HTTPError`
    """
    validate_lfs_upload_action(upload_action)
    header = upload_action.get("header", {})
    chunk_size = header.get("chunk_size", None)
    if chunk_size is not None:
        if isinstance(chunk_size, str):
            chunk_size = int(chunk_size, 10)
        else:
            raise ValueError(
                "Malformed response from LFS batch endpoint: `chunk_size`"
                " should be a string"
            )
        return _upload_multi_part(
            completion_url=upload_action["href"],
            fileobj=fileobj,
            chunk_size=chunk_size,
            header=header,
            upload_info=upload_info,
        )
    return _upload_single_part(
        upload_url=upload_action["href"],
        fileobj=fileobj,
    )


def _upload_single_part(upload_url: str, fileobj: BinaryIO):
    upload_res = requests.put(upload_url, data=fileobj)
    upload_res.raise_for_status()
    return upload_res


def _upload_multi_part(
    completion_url: str,
    fileobj: BinaryIO,
    header: dict,
    chunk_size: int,
    upload_info: UploadInfo,
):
    """
    TODO @SBrandeis
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

    completion_payload = {
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
            part_upload_res = requests.put(part_upload_url, data=fileobj_slice)
            part_upload_res.raise_for_status()
            etag = part_upload_res.headers.get("etag")
            if etag is None or etag == "":
                raise ValueError(
                    f"Invalid etag (`{etag}`) returned for part {part_idx +1} of"
                    f" {num_parts}"
                )
            completion_payload["parts"][part_idx]["etag"] = etag

    completion_res = requests.post(
        completion_url,
        json=completion_payload,
        headers=LFS_HEADERS,
    )
    completion_res.raise_for_status()
    return completion_res


class SliceFileObj(AbstractContextManager):
    """
    Utility context manager to read a slice of a file-like object as a file-like object

    This is NOT thread safe

    Inspired by stackoverflow.com/a/29838711/593036

    Credits to @julien-c
    """

    def __init__(self, fileobj: BinaryIO, seek_from: int, read_limit: int):
        self.fileobj = fileobj
        self.seek_from = seek_from
        self.read_limit = read_limit
        self.n_seen = 0

    def __enter__(self):
        self.previous_position = self.fileobj.tell()
        self.fileobj.seek(self.seek_from, io.SEEK_SET)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fileobj.seek(self.previous_position, io.SEEK_SET)

    def read(self, n: int = -1):
        if self.n_seen >= self.read_limit:
            return b""
        remaining_amount = self.read_limit - self.n_seen
        data = self.fileobj.read(
            remaining_amount if n < 0 else min(n, remaining_amount)
        )
        self.n_seen += len(data)
        return data

    def __iter__(self):
        yield self.read(n=4 * 1024 * 1024)
