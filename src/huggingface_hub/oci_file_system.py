# Copyright 2024-present, the HuggingFace Inc. team.
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
"""Utilities for uploading and downloading Hugging Face Hub content to/from OCI Object Storage.

This module enables using Oracle Cloud Infrastructure (OCI) Object Storage as a destination
for ``hf_hub_download`` and ``snapshot_download`` by passing an ``oci://`` URI as ``local_dir``.

Usage::

    from huggingface_hub import snapshot_download

    # Download a full model snapshot into OCI Object Storage
    snapshot_download(
        "meta-llama/Llama-2-7b-hf",
        local_dir="oci://my-bucket@my-namespace/models/llama2-7b",
    )

    # Download a single file into OCI Object Storage
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        "bert-base-uncased",
        "config.json",
        local_dir="oci://my-bucket@my-namespace/bert/",
    )

Requirements:
    Install with: ``pip install huggingface_hub[oci]``

    OCI credentials must be configured via ``~/.oci/config`` or instance/resource principal.
    See https://oracle-cloud-infrastructure-python-sdk.readthedocs.io/en/latest/installation.html
"""

import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

from .utils import logging
from .utils._runtime import is_ocifs_available


logger = logging.get_logger(__name__)

OCI_URI_PREFIX = "oci://"


def is_oci_uri(path: Union[str, Path, None]) -> bool:
    """Return ``True`` if *path* is an OCI Object Storage URI (starts with ``oci://``).

    Args:
        path: The path string to check.

    Returns:
        ``True`` when *path* starts with ``"oci://"``, ``False`` otherwise.

    Example::

        >>> is_oci_uri("oci://my-bucket@namespace/models/bert")
        True
        >>> is_oci_uri("/local/path")
        False
    """
    return isinstance(path, str) and path.startswith(OCI_URI_PREFIX)


def _get_oci_fs(storage_options: Optional[dict] = None):
    """Return an ``ocifs.OCIFileSystem`` instance.

    Args:
        storage_options: Optional keyword arguments forwarded verbatim to
            ``ocifs.OCIFileSystem()``.  Use this to customise authentication
            (e.g. ``{"config": "/path/to/oci_config", "profile": "MY_PROFILE"}``).

    Raises:
        ImportError: If ``ocifs`` is not installed.
    """
    if not is_ocifs_available():
        raise ImportError(
            "OCI Object Storage support requires the `ocifs` package. "
            "Install it with: pip install huggingface_hub[oci]"
        )
    import ocifs

    return ocifs.OCIFileSystem(**(storage_options or {}))


def upload_folder_to_oci(
    local_folder: Union[str, Path],
    oci_uri: str,
    *,
    delete_local: bool = False,
    storage_options: Optional[dict] = None,
) -> str:
    """Upload a local folder tree to OCI Object Storage.

    All files under *local_folder* are uploaded, preserving their relative paths
    beneath *oci_uri*.

    Args:
        local_folder: Path to the local source folder.
        oci_uri: Destination OCI URI, e.g.
            ``"oci://my-bucket@namespace/models/llama2-7b"``.
        delete_local: If ``True``, delete *local_folder* from disk after a
            successful upload.
        storage_options: Optional kwargs forwarded to ``ocifs.OCIFileSystem``.

    Returns:
        The destination *oci_uri* that was passed in.

    Raises:
        ImportError: If ``ocifs`` is not installed.
        ValueError: If *oci_uri* does not start with ``"oci://"``.

    Example::

        upload_folder_to_oci(
            "/tmp/llama2-7b-snapshot",
            "oci://my-bucket@namespace/models/llama2-7b",
        )
    """
    if not is_oci_uri(oci_uri):
        raise ValueError(f"Expected an OCI URI (starting with 'oci://'), got: {oci_uri!r}")

    local_folder = Path(local_folder)
    fs = _get_oci_fs(storage_options)
    dest_root = oci_uri.rstrip("/")

    logger.info(f"Uploading {local_folder} to {oci_uri}")
    for local_file in sorted(local_folder.rglob("*")):
        if not local_file.is_file():
            continue
        relative = local_file.relative_to(local_folder)
        dest_path = f"{dest_root}/{relative.as_posix()}"
        logger.debug(f"  {relative} -> {dest_path}")
        fs.put_file(str(local_file), dest_path)

    if delete_local:
        shutil.rmtree(local_folder)
        logger.debug(f"Deleted local folder {local_folder}")

    return oci_uri


def upload_file_to_oci(
    local_path: Union[str, Path],
    oci_uri: str,
    *,
    storage_options: Optional[dict] = None,
) -> str:
    """Upload a single local file to OCI Object Storage.

    Args:
        local_path: Path to the local source file.
        oci_uri: Destination OCI URI.  If it ends with ``"/"``, the source
            filename is appended automatically.
        storage_options: Optional kwargs forwarded to ``ocifs.OCIFileSystem``.

    Returns:
        The resolved destination OCI URI (with filename appended when needed).

    Raises:
        ImportError: If ``ocifs`` is not installed.
        ValueError: If *oci_uri* does not start with ``"oci://"``.

    Example::

        upload_file_to_oci("/tmp/config.json", "oci://my-bucket@namespace/bert/")
        # uploads to oci://my-bucket@namespace/bert/config.json
    """
    if not is_oci_uri(oci_uri):
        raise ValueError(f"Expected an OCI URI (starting with 'oci://'), got: {oci_uri!r}")

    local_path = Path(local_path)
    fs = _get_oci_fs(storage_options)

    dest = f"{oci_uri}{local_path.name}" if oci_uri.endswith("/") else oci_uri
    logger.debug(f"Uploading {local_path} -> {dest}")
    fs.put_file(str(local_path), dest)
    return dest


def _snapshot_download_to_oci(
    repo_id: str,
    oci_uri: str,
    *,
    # Forward all snapshot_download kwargs
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    user_agent: Optional[Union[dict, str]] = None,
    etag_timeout: float = 10,
    force_download: bool = False,
    token: Optional[Union[bool, str]] = None,
    local_files_only: bool = False,
    allow_patterns: Optional[Union[list, str]] = None,
    ignore_patterns: Optional[Union[list, str]] = None,
    max_workers: int = 8,
    headers: Optional[dict] = None,
    endpoint: Optional[str] = None,
    storage_options: Optional[dict] = None,
) -> str:
    """Download a snapshot to a temporary directory and upload it to OCI Object Storage.

    This is the internal implementation called by :func:`snapshot_download` when
    *local_dir* is an ``oci://`` URI.  Prefer using ``snapshot_download`` directly.

    Returns:
        The destination *oci_uri*.
    """
    # Lazy import to avoid circular dependency
    from ._snapshot_download import snapshot_download

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info(f"Downloading {repo_id} snapshot to temporary directory before uploading to {oci_uri}")
        snapshot_download(
            repo_id,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=tmp_dir,
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
            etag_timeout=etag_timeout,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=max_workers,
            headers=headers,
            endpoint=endpoint,
        )
        upload_folder_to_oci(tmp_dir, oci_uri, storage_options=storage_options)

    return oci_uri


def _hf_hub_download_to_oci(
    repo_id: str,
    filename: str,
    oci_uri: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    user_agent: Optional[Union[dict, str]] = None,
    etag_timeout: float = 10,
    force_download: bool = False,
    token: Optional[Union[bool, str]] = None,
    local_files_only: bool = False,
    headers: Optional[dict] = None,
    endpoint: Optional[str] = None,
    storage_options: Optional[dict] = None,
) -> str:
    """Download a single file to a temporary location then upload it to OCI Object Storage.

    This is the internal implementation called by :func:`hf_hub_download` when
    *local_dir* is an ``oci://`` URI.  Prefer using ``hf_hub_download`` directly.

    Returns:
        The destination OCI URI (with the filename component appended when *oci_uri*
        ends with ``"/"``).
    """
    # Lazy import to avoid circular dependency
    from .file_download import hf_hub_download

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = hf_hub_download(
            repo_id,
            filename=filename,
            subfolder=subfolder,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=tmp_dir,
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
            etag_timeout=etag_timeout,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
            headers=headers,
            endpoint=endpoint,
        )
        dest = upload_file_to_oci(local_path, oci_uri, storage_options=storage_options)

    return dest
