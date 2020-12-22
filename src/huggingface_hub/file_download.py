import copy
import fnmatch
import io
import json
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple, Union

from tqdm.auto import tqdm

import requests
from filelock import FileLock

from . import __version__
from .hf_api import HfFolder


logger = logging.getLogger(__name__)


try:
    import torch

    _torch_available = True
except ImportError:
    _torch_available = False

try:
    import tensorflow as tf

    _tf_available = True
except ImportError:
    _tf_available = False


def is_torch_available():
    return _torch_available


def is_tf_available():
    return _tf_available


# Constants for file downloads

PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
CONFIG_NAME = "config.json"

HUGGINGFACE_CO_URL_TEMPLATE = (
    "https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
)


# default cache
hf_cache_home = os.path.expanduser(
    os.getenv(
        "HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface")
    )
)
default_cache_path = os.path.join(hf_cache_home, "hub")

HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", default_cache_path)


def hf_hub_url(
    model_id: str,
    filename: str,
    subfolder: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """
    Resolve a model identifier, a file name, and an optional revision id, to a huggingface.co-hosted url, redirecting
    to Cloudfront (a Content Delivery Network, or CDN) for large files (more than a few MBs).

    Cloudfront is replicated over the globe so downloads are way faster for the end user (and it also lowers our
    bandwidth costs).

    Cloudfront aggressively caches files by default (default TTL is 24 hours), however this is not an issue here
    because we implement a git-based versioning system on huggingface.co, which means that we store the files on S3/Cloudfront
    in a content-addressable way (i.e., the file name is its hash). Using content-addressable filenames means cache
    can't ever be stale.

    In terms of client-side caching from this library, we base our caching on the objects' ETag. An object's ETag is:
    its git-sha1 if stored in git, or its sha256 if stored in git-lfs.
    """
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    if revision is None:
        revision = "main"
    return HUGGINGFACE_CO_URL_TEMPLATE.format(
        model_id=model_id, revision=revision, filename=filename
    )


def url_to_filename(url: str, etag: Optional[str] = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way. If `etag` is specified, append its hash to the url's,
    delimited by a period. If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name so that TF 2.0 can
    identify it as a HDF5 file (see
    https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode("utf-8")
    filename = sha256(url_bytes).hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        filename += "." + sha256(etag_bytes).hexdigest()

    if url.endswith(".h5"):
        filename += ".h5"

    return filename


def filename_to_url(filename, cache_dir=None) -> Tuple[str, str]:
    """
    Return the url and etag (which may be ``None``) stored for `filename`. Raise ``EnvironmentError`` if `filename` or
    its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError("file {} not found".format(cache_path))

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise EnvironmentError("file {} not found".format(meta_path))

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag


def http_user_agent(
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    user_agent: Union[Dict, str, None] = None,
) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    if library_name is not None:
        ua = "{}/{}".format(library_name, library_version)
    else:
        ua = "unknown/None"
    ua += "; hf_hub/{}".format(__version__)
    ua += "; python/{}".format(sys.version.split()[0])
    if is_torch_available():
        ua += "; torch/{}".format(torch.__version__)
    if is_tf_available():
        ua += "; tensorflow/{}".format(tf.__version__)
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join("{}/{}".format(k, v) for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def http_get(
    url: str,
    temp_file: BinaryIO,
    proxies=None,
    resume_size=0,
    headers: Optional[Dict[str, str]] = None,
):
    """
    Donwload remote file. Do not gobble up errors.
    """
    headers = copy.deepcopy(headers)
    if resume_size > 0:
        headers["Range"] = "bytes=%d-" % (resume_size,)
    r = requests.get(url, stream=True, proxies=proxies, headers=headers)
    r.raise_for_status()
    content_length = r.headers.get("Content-Length")
    total = resume_size + int(content_length) if content_length is not None else None
    progress = tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        initial=resume_size,
        desc="Downloading",
        disable=bool(logger.getEffectiveLevel() == logging.NOTSET),
    )
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def cached_download(
    url: str,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    user_agent: Union[Dict, str, None] = None,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    use_auth_token: Union[str, None] = None,
    local_files_only=False,
) -> Optional[str]:  # pragma: no cover
    """
    Given a URL, look for the corresponding file in the local cache. If it's not there, download it. Then return the
    path to the cached file.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    headers = {
        "user-agent": http_user_agent(
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
        )
    }
    if isinstance(use_auth_token, str):
        headers["authorization"] = "Bearer {}".format(use_auth_token)
    elif use_auth_token:
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError(
                "You specified use_auth_token=True, but a huggingface token was not found."
            )
        headers["authorization"] = "Bearer {}".format(token)

    url_to_download = url
    etag = None
    if not local_files_only:
        try:
            r = requests.head(
                url,
                headers=headers,
                allow_redirects=False,
                proxies=proxies,
                timeout=etag_timeout,
            )
            r.raise_for_status()
            etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
            # We favor a custom header indicating the etag of the linked resource, and
            # we fallback to the regular etag header.
            # If we don't have any of those, raise an error.
            if etag is None:
                raise OSError(
                    "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
                )
            # In case of a redirect,
            # save an extra redirect on the request.get call,
            # and ensure we download the exact atomic version even if it changed
            # between the HEAD and the GET (unlikely, but hey).
            if 300 <= r.status_code <= 399:
                url_to_download = r.headers["Location"]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # etag is already None
            pass

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # etag is None == we don't have a connection or we passed local_files_only.
    # try to get the last downloaded one
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [
                file
                for file in fnmatch.filter(
                    os.listdir(cache_dir), filename.split(".")[0] + ".*"
                )
                if not file.endswith(".json") and not file.endswith(".lock")
            ]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            else:
                # If files cannot be found and local_files_only=True,
                # the models might've been found if local_files_only=False
                # Notify the user about that
                if local_files_only:
                    raise ValueError(
                        "Cannot find the requested files in the cached path and outgoing traffic has been"
                        " disabled. To enable model look-ups and downloads online, set 'local_files_only'"
                        " to False."
                    )
                else:
                    raise ValueError(
                        "Connection error, and we cannot find the requested files in the cached path."
                        " Please try again or make sure your Internet connection is on."
                    )

    # From now on, etag is not None.
    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):

        # If the download just completed while the lock was activated.
        if os.path.exists(cache_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return cache_path

        if resume_download:
            incomplete_path = cache_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager() -> "io.BufferedWriter":
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(
                tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False
            )
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info("downloading %s to %s", url, temp_file.name)

            http_get(
                url_to_download,
                temp_file,
                proxies=proxies,
                resume_size=resume_size,
                headers=headers,
            )

        logger.info("storing %s in cache at %s", url, cache_path)
        os.replace(temp_file.name, cache_path)

        logger.info("creating metadata file for %s", cache_path)
        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path
