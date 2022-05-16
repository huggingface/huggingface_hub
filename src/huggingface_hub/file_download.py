import copy
import fnmatch
import io
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple, Union

import packaging.version
from tqdm.auto import tqdm

import requests
from filelock import FileLock
from huggingface_hub import constants

from . import __version__
from .constants import (
    DEFAULT_REVISION,
    HUGGINGFACE_CO_URL_TEMPLATE,
    HUGGINGFACE_HUB_CACHE,
    REPO_TYPES,
    REPO_TYPES_URL_PREFIXES,
)
from .hf_api import HfFolder
from .utils import logging
from .utils._deprecation import _deprecate_positional_args


logger = logging.get_logger(__name__)

_PY_VERSION: str = sys.version.split()[0].rstrip("+")

if packaging.version.Version(_PY_VERSION) < packaging.version.Version("3.8.0"):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

_torch_version = "N/A"
_torch_available = False
try:
    _torch_version = importlib_metadata.version("torch")
    _torch_available = True
except importlib_metadata.PackageNotFoundError:
    pass

_pydot_available = False

try:
    _pydot_version = importlib_metadata.version("pydot")
    _pydot_available = True
except importlib_metadata.PackageNotFoundError:
    pass


def is_pydot_available():
    return _pydot_available


_graphviz_available = False

try:
    _graphviz_version = importlib_metadata.version("graphviz")
    _graphviz_available = True
except importlib_metadata.PackageNotFoundError:
    pass


def is_graphviz_available():
    return _graphviz_available


_tf_version = "N/A"
_tf_available = False
_tf_candidates = (
    "tensorflow",
    "tensorflow-cpu",
    "tensorflow-gpu",
    "tf-nightly",
    "tf-nightly-cpu",
    "tf-nightly-gpu",
    "intel-tensorflow",
    "intel-tensorflow-avx512",
    "tensorflow-rocm",
    "tensorflow-macos",
)
for package_name in _tf_candidates:
    try:
        _tf_version = importlib_metadata.version(package_name)
        _tf_available = True
        break
    except importlib_metadata.PackageNotFoundError:
        pass

_fastai_version = "N/A"
_fastai_available = False
try:
    _fastai_version: str = importlib_metadata.version("fastai")
    _fastai_available = True
except importlib_metadata.PackageNotFoundError:
    pass

_fastcore_version = "N/A"
_fastcore_available = False
try:
    _fastcore_version: str = importlib_metadata.version("fastcore")
    _fastcore_available = True
except importlib_metadata.PackageNotFoundError:
    pass


def is_torch_available():
    return _torch_available


def is_tf_available():
    return _tf_available


def get_tf_version():
    return _tf_version


def is_fastai_available():
    return _fastai_available


def get_fastai_version():
    return _fastai_version


def is_fastcore_available():
    return _fastcore_available


def get_fastcore_version():
    return _fastcore_version


@_deprecate_positional_args
def hf_hub_url(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """Construct the URL of a file from the given information.

    The resolved address can either be a huggingface.co-hosted url, or a link to
    Cloudfront (a Content Delivery Network, or CDN) for large files which are
    more than a few MBs.

    Args:
        repo_id (`str`):
            A namespace (user or an organization) name and a repo name separated
            by a `/`.
        filename (`str`):
            The name of the file in the repo.
        subfolder (`str`, *optional*):
            An optional value corresponding to a folder inside the repo.
        repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if uploading to a dataset or space,
            `None` or `"model"` if uploading to a model. Default is `None`.
        revision (`str`, *optional*):
            An optional Git revision id which can be a branch name, a tag, or a
            commit hash.

    Example:

    ```python
    >>> from huggingface_hub import hf_hub_url

    >>> hf_hub_url(
    ...     repo_id="julien-c/EsperBERTo-small", filename="pytorch_model.bin"
    ... )
    'https://huggingface.co/julien-c/EsperBERTo-small/resolve/main/pytorch_model.bin'
    ```

    <Tip>

    Notes:

        Cloudfront is replicated over the globe so downloads are way faster for
        the end user (and it also lowers our bandwidth costs).

        Cloudfront aggressively caches files by default (default TTL is 24
        hours), however this is not an issue here because we implement a
        git-based versioning system on huggingface.co, which means that we store
        the files on S3/Cloudfront in a content-addressable way (i.e., the file
        name is its hash). Using content-addressable filenames means cache can't
        ever be stale.

        In terms of client-side caching from this library, we base our caching
        on the objects' entity tag (`ETag`), which is an identifier of a
        specific version of a resource [1]_. An object's ETag is: its git-sha1
        if stored in git, or its sha256 if stored in git-lfs.

    </Tip>

    References:

    -  [1] https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag
    """
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    if repo_type not in REPO_TYPES:
        raise ValueError("Invalid repo type")

    if repo_type in REPO_TYPES_URL_PREFIXES:
        repo_id = REPO_TYPES_URL_PREFIXES[repo_type] + repo_id

    if revision is None:
        revision = DEFAULT_REVISION
    return HUGGINGFACE_CO_URL_TEMPLATE.format(
        repo_id=repo_id, revision=revision, filename=filename
    )


def url_to_filename(url: str, etag: Optional[str] = None) -> str:
    """Generate a local filename from a url.

    Convert `url` into a hashed filename in a reproducible way. If `etag` is
    specified, append its hash to the url's, delimited by a period. If the url
    ends with .h5 (Keras HDF5 weights) adds '.h5' to the name so that TF 2.0 can
    identify it as a HDF5 file (see
    https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)

    Args:
        url (`str`):
            The address to the file.
        etag (`str`, *optional*):
            The ETag of the file.

    Returns:
        The generated filename.
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
    Return the url and etag (which may be `None`) stored for `filename`. Raise
    `EnvironmentError` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError(f"file {cache_path} not found")

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise EnvironmentError(f"file {meta_path} not found")

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag


@_deprecate_positional_args
def http_user_agent(
    *,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    user_agent: Union[Dict, str, None] = None,
) -> str:
    """Formats a user-agent string with basic info about a request.

    Args:
        library_name (`str`, *optional*):
            The name of the library to which the object corresponds.
        library_version (`str`, *optional*):
            The version of the library.
        user_agent (`str`, `dict`, *optional*):
            The user agent info in the form of a dictionary or a single string.

    Returns:
        The formatted user-agent string.
    """
    if library_name is not None:
        ua = f"{library_name}/{library_version}"
    else:
        ua = "unknown/None"
    ua += f"; hf_hub/{__version__}"
    ua += f"; python/{_PY_VERSION}"
    if is_torch_available():
        ua += f"; torch/{_torch_version}"
    if is_tf_available():
        ua += f"; tensorflow/{_tf_version}"
    if is_fastai_available():
        ua += f"; fastai/{_fastai_version}"
    if is_fastcore_available():
        ua += f"; fastcore/{_fastcore_version}"
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


class OfflineModeIsEnabled(ConnectionError):
    pass


def _raise_if_offline_mode_is_enabled(msg: Optional[str] = None):
    """Raise a OfflineModeIsEnabled error (subclass of ConnectionError) if
    HF_HUB_OFFLINE is True."""
    if constants.HF_HUB_OFFLINE:
        raise OfflineModeIsEnabled(
            "Offline mode is enabled."
            if msg is None
            else "Offline mode is enabled. " + str(msg)
        )


def _request_with_retry(
    method: str,
    url: str,
    *,
    max_retries: int = 0,
    base_wait_time: float = 0.5,
    max_wait_time: float = 2,
    timeout: float = 10.0,
    **params,
) -> requests.Response:
    """Wrapper around requests to retry in case it fails with a `ConnectTimeout`, with
    exponential backoff.

        Note that if the environment variable HF_HUB_OFFLINE is set to 1, then a
        `OfflineModeIsEnabled` error is raised.

        Args:
            method (`str`):
                HTTP method, such as 'GET' or 'HEAD'.
            url (`str`):
                The URL of the resource to fetch.
            max_retries (`int`, *optional*, defaults to `0`):
                Maximum number of retries, defaults to 0 (no retries).
            base_wait_time (`float`, *optional*, defaults to `0.5`):
                Duration (in seconds) to wait before retrying the first time.
                Wait time between retries then grows exponentially, capped by
                `max_wait_time`.
            max_wait_time (`float`, *optional*, defaults to `2`):
                Maximum amount of time between two retries, in seconds.
            timeout (`float`, *optional*, defaults to `10`):
                How many seconds to wait for the server to send data before
                giving up which is passed to `requests.request`.
            **params (`dict`, *optional*):
                Params to pass to `requests.request`.
    """
    _raise_if_offline_mode_is_enabled(f"Tried to reach {url}")
    tries, success = 0, False
    while not success:
        tries += 1
        try:
            response = requests.request(
                method=method.upper(), url=url, timeout=timeout, **params
            )
            success = True
        except requests.exceptions.ConnectTimeout as err:
            if tries > max_retries:
                raise err
            else:
                logger.info(
                    f"{method} request to {url} timed out, retrying..."
                    f" [{tries/max_retries}]"
                )
                sleep_time = min(
                    max_wait_time, base_wait_time * 2 ** (tries - 1)
                )  # Exponential backoff
                time.sleep(sleep_time)
    return response


@_deprecate_positional_args
def http_get(
    url: str,
    temp_file: BinaryIO,
    *,
    proxies=None,
    resume_size=0,
    headers: Optional[Dict[str, str]] = None,
    timeout=10.0,
    max_retries=0,
):
    """
    Donwload remote file. Do not gobble up errors.
    """
    headers = copy.deepcopy(headers)
    if resume_size > 0:
        headers["Range"] = "bytes=%d-" % (resume_size,)
    r = _request_with_retry(
        method="GET",
        url=url,
        stream=True,
        proxies=proxies,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
    )
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


@_deprecate_positional_args
def cached_download(
    url: str,
    *,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    user_agent: Union[Dict, str, None] = None,
    force_download: Optional[bool] = False,
    force_filename: Optional[str] = None,
    proxies: Optional[Dict] = None,
    etag_timeout: Optional[float] = 10,
    resume_download: Optional[bool] = False,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only: Optional[bool] = False,
) -> Optional[str]:  # pragma: no cover
    """
    Download from a given URL and cache it if it's not already present in the
    local cache.

    Given a URL, this function looks for the corresponding file in the local
    cache. If it's not there, download it. Then return the path to the cached
    file.

    Args:
        url (`str`):
            The path to the file to be downloaded.
        library_name (`str`, *optional*):
            The name of the library to which the object corresponds.
        library_version (`str`, *optional*):
            The version of the library.
        cache_dir (`str`, `Path`, *optional*):
            Path to the folder where cached files are stored.
        user_agent (`dict`, `str`, *optional*):
            The user-agent info in the form of a dictionary or a string.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether the file should be downloaded even if it already exists in
            the local cache.
        force_filename (`str`, *optional*):
            Use this name instead of a generated file name.
        proxies (`dict`, *optional*):
            Dictionary mapping protocol to the URL of the proxy passed to
            `requests.request`.
        etag_timeout (`float`, *optional* defaults to `10`):
            When fetching ETag, how many seconds to wait for the server to send
            data before giving up which is passed to `requests.request`.
        resume_download (`bool`, *optional*, defaults to `False`):
            If `True`, resume a previously interrupted download.
        use_auth_token (`bool`, `str`, *optional*):
            A token to be used for the download.
                - If `True`, the token is read from the HuggingFace config
                  folder.
                - If a string, it's used as the authentication token.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, avoid downloading the file and return the path to the
            local cached file if it exists.

    Returns:
        Local path (string) of file or if networking is off, last version of
        file cached on disk.

    <Tip>

    Raises the following errors:

        - [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
          if `use_auth_token=True` and the token cannot be found.
        - [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError)
          if ETag cannot be determined.
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
          if some parameter value is invalid

    </Tip>
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
        headers["authorization"] = f"Bearer {use_auth_token}"
    elif use_auth_token:
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError(
                "You specified use_auth_token=True, but a huggingface token was not"
                " found."
            )
        headers["authorization"] = f"Bearer {token}"

    url_to_download = url
    etag = None
    if not local_files_only:
        try:
            r = _request_with_retry(
                method="HEAD",
                url=url,
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
                    "Distant resource does not have an ETag, we won't be able to"
                    " reliably ensure reproducibility."
                )
            # In case of a redirect,
            # save an extra redirect on the request.get call,
            # and ensure we download the exact atomic version even if it changed
            # between the HEAD and the GET (unlikely, but hey).
            if 300 <= r.status_code <= 399:
                url_to_download = r.headers["Location"]
        except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
            # Actually raise for those subclasses of ConnectionError
            raise
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            OfflineModeIsEnabled,
        ):
            # Otherwise, our Internet connection is down.
            # etag is None
            pass

    filename = (
        force_filename if force_filename is not None else url_to_filename(url, etag)
    )

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # etag is None == we don't have a connection or we passed local_files_only.
    # try to get the last downloaded one
    if etag is None:
        if os.path.exists(cache_path) and not force_download:
            return cache_path
        else:
            matching_files = [
                file
                for file in fnmatch.filter(
                    os.listdir(cache_dir), filename.split(".")[0] + ".*"
                )
                if not file.endswith(".json") and not file.endswith(".lock")
            ]
            if (
                len(matching_files) > 0
                and not force_download
                and force_filename is None
            ):
                return os.path.join(cache_dir, matching_files[-1])
            else:
                # If files cannot be found and local_files_only=True,
                # the models might've been found if local_files_only=False
                # Notify the user about that
                if local_files_only:
                    raise ValueError(
                        "Cannot find the requested files in the cached path and"
                        " outgoing traffic has been disabled. To enable model look-ups"
                        " and downloads online, set 'local_files_only' to False."
                    )
                else:
                    raise ValueError(
                        "Connection error, and we cannot find the requested files in"
                        " the cached path. Please try again or make sure your Internet"
                        " connection is on."
                    )

    # From now on, etag is not None.
    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"

    # Some Windows versions do not allow for paths longer than 255 characters.
    # In this case, we must specify it is an extended path by using the "\\?\" prefix.
    if os.name == "nt" and len(os.path.abspath(lock_path)) > 255:
        lock_path = "\\\\?\\" + os.path.abspath(lock_path)

    if os.name == "nt" and len(os.path.abspath(cache_path)) > 255:
        cache_path = "\\\\?\\" + os.path.abspath(cache_path)

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

        if force_filename is None:
            logger.info("creating metadata file for %s", cache_path)
            meta = {"url": url, "etag": etag}
            meta_path = cache_path + ".json"
            with open(meta_path, "w") as meta_file:
                json.dump(meta, meta_file)

    return cache_path


@_deprecate_positional_args
def hf_hub_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    user_agent: Union[Dict, str, None] = None,
    force_download: Optional[bool] = False,
    force_filename: Optional[str] = None,
    proxies: Optional[Dict] = None,
    etag_timeout: Optional[float] = 10,
    resume_download: Optional[bool] = False,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only: Optional[bool] = False,
):
    """Download a given file if it's not already present in the local cache.

    Args:
        repo_id (`str`):
            A user or an organization name and a repo name separated by a `/`.
        filename (`str`):
            The name of the file in the repo.
        subfolder (`str`, *optional*):
            An optional value corresponding to a folder inside the model repo.
        repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if uploading to a dataset or space,
            `None` or `"model"` if uploading to a model. Default is `None`.
        revision (`str`, *optional*):
            An optional Git revision id which can be a branch name, a tag, or a
            commit hash.
        library_name (`str`, *optional*):
            The name of the library to which the object corresponds.
        library_version (`str`, *optional*):
            The version of the library.
        cache_dir (`str`, `Path`, *optional*):
            Path to the folder where cached files are stored.
        user_agent (`dict`, `str`, *optional*):
            The user-agent info in the form of a dictionary or a string.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether the file should be downloaded even if it already exists in
            the local cache.
        force_filename (`str`, *optional*):
            Use this name instead of a generated file name.
        proxies (`dict`, *optional*):
            Dictionary mapping protocol to the URL of the proxy passed to
            `requests.request`.
        etag_timeout (`float`, *optional*, defaults to `10`):
            When fetching ETag, how many seconds to wait for the server to send
            data before giving up which is passed to `requests.request`.
        resume_download (`bool`, *optional*, defaults to `False`):
            If `True`, resume a previously interrupted download.
        use_auth_token (`str`, `bool`, *optional*):
            A token to be used for the download.
                - If `True`, the token is read from the HuggingFace config
                  folder.
                - If a string, it's used as the authentication token.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, avoid downloading the file and return the path to the
            local cached file if it exists.

    Returns:
        Local path (string) of file or if networking is off, last version of
        file cached on disk.

    <Tip>

    Raises the following errors:

        - [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
          if `use_auth_token=True` and the token cannot be found.
        - [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError)
          if ETag cannot be determined.
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
          if some parameter value is invalid

    </Tip>
    """
    url = hf_hub_url(
        repo_id, filename, subfolder=subfolder, repo_type=repo_type, revision=revision
    )

    return cached_download(
        url,
        library_name=library_name,
        library_version=library_version,
        cache_dir=cache_dir,
        user_agent=user_agent,
        force_download=force_download,
        force_filename=force_filename,
        proxies=proxies,
        etag_timeout=etag_timeout,
        resume_download=resume_download,
        use_auth_token=use_auth_token,
        local_files_only=local_files_only,
    )
