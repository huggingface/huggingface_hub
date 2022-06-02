"""
Utilities & types to handle upload of large files to the HF Hub
via the git LFS protocol
"""

import io
from contextlib import AbstractContextManager
from dataclasses import dataclass
from math import ceil
from os.path import getsize
from typing import BinaryIO, Dict, List, TypedDict, Union

import requests
from _typeshed import SupportsRead

from .sha import sha256, sha_fileobj


LFS_HEADERS = {
    "Accept": "application/vnd.git-lfs+json",
    "Content-Type": "application/vnd.git-lfs+json",
}


@dataclass
class UploadInfo:
    """
    Dataclass holding required information to determine wether a blob
    should be uploaded to the hub using the LFS protocol or the regular protocol
    """

    sha256: bytes
    """SHA256 hash of the blob"""
    size: int
    """Size in bytes of the blob"""
    sample: bytes
    """First 512 bytes of the blob"""

    @classmethod
    def from_path(cls, path: str):
        size = getsize(path)
        with io.open(path, "rb") as file:
            sample = file.peek(512)
            sha = sha_fileobj(file)
        return cls(size=size, sha256=sha, sample=sample)

    @classmethod
    def from_bytes(cls, data: bytes):
        sha = sha256(data).digest()
        return cls(size=len(data), sample=data[:512], sha256=sha)

    @classmethod
    def from_readable(cls, readable: BinaryIO):
        sample = readable.read(512)
        readable.seek(0, io.SEEK_SET)
        sha = sha_fileobj(readable)
        size = readable.tell()
        readable.seek(0, io.SEEK_SET)
        return cls(size=size, sha256=sha, sample=sample)


class LfsBatchResponse(TypedDict):
    """
    Response from the LFS batch endpoint
    See https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md
    """

    objects: List[Union["LfsBatchObject", "LfsBatchObjectError"]]


class LfsBatchObjectBase(TypedDict):
    oid: str
    size: int


class LfsBatchObject(LfsBatchObjectBase):
    actions: "LfsObjectActions"


class LfsBatchObjectError(LfsBatchObjectBase):
    error: "LfsError"


class LfsObjectActions(TypedDict, total=False):
    download: "LfsAction"
    upload: "LfsAction"
    verify: "LfsAction"


class LfsAction(TypedDict):
    href: str
    header: Dict[str, str]


class LfsError(TypedDict):
    code: int
    message: str


def upload_lfs_file(
    fileobj: BinaryIO,
    upload_action: LfsAction,
    upload_info: UploadInfo,
):
    """
    Uploads a file using the git lfs protocol and determines automatically whether or not
    to use the multipart transfer protocol

    Args:
        fileobj (file-like object):
            The content of the file to upload
        upload_action (`LfsAction`):
            The `upload` action from the LFS Batch endpoint
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
    chunk_size = upload_action.get("header", {}).get("chunk_size", None)
    if chunk_size is not None:
        if isinstance(chunk_size, str):
            chunk_size = int(chunk_size, 10)
        else:
            raise ValueError(
                "Malformed response from LFS batch endpoint: `chunk_size`"
                " should be a string"
            )
        return _upload_multi_part(
            fileobj=fileobj,
            upload_action=upload_action,
            chunk_size=chunk_size,
            upload_info=upload_info,
        )
    return _upload_single_part(fileobj=fileobj, upload_action=upload_action)


def _upload_single_part(fileobj: BinaryIO, upload_action: LfsAction):
    upload_url = upload_action["href"]
    upload_res = requests.put(upload_url, data=fileobj)
    upload_res.raise_for_status()
    return upload_res


def _upload_multi_part(
    fileobj: BinaryIO,
    upload_action: LfsAction,
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
                for part_num, upload_url in upload_action["header"].items()
                if part_num.isdigit() and len(part_num) > 0
            ],
            key=lambda t: t[0],
        )
    ]
    num_parts = len(sorted_part_upload_urls)
    if num_parts != ceil(upload_info.size / chunk_size):
        raise ValueError("Invalid server response to upload large LFS file")

    completion_url = upload_action["href"]
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


class SliceFileObj(AbstractContextManager, SupportsRead[bytes]):
    """
    Utility context manager to read a slice of a file-like object as a file-like object

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

    def __exit__(self):
        self.fileobj.seek(self.previous_position, io.SEEK_SET)

    def read(self, n=-1):
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
