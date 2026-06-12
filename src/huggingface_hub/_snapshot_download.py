import os
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, overload
from urllib.parse import quote, urlparse

import httpx
from tqdm.auto import tqdm as base_tqdm
from tqdm.contrib.concurrent import thread_map

from . import constants
from .errors import (
    DryRunError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from ._tree_cache import TreeCacheEntry, read_tree_cache, write_tree_cache
from .file_download import (
    REGEX_COMMIT_HASH,
    DryRunFileInfo,
    HfFileMetadata,
    hf_hub_download,
    hf_hub_url,
    repo_folder_name,
)
from .hf_api import DatasetInfo, HfApi, KernelInfo, ModelInfo, RepoFile, SpaceInfo
from .utils import OfflineModeIsEnabled, filter_repo_objects, logging, validate_hf_hub_args
from .utils._xet import XetFileData, XetTokenType, xet_connection_info_refresh_url
from .utils.tqdm import _create_progress_bar
from .utils.tqdm import tqdm as hf_tqdm


logger = logging.get_logger(__name__)

LARGE_REPO_THRESHOLD = 1000  # After this limit, we don't consider `repo_info.siblings` to be reliable enough


@overload
def snapshot_download(
    repo_id: str,
    *,
    repo_type: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    library_name: str | None = None,
    library_version: str | None = None,
    user_agent: dict | str | None = None,
    etag_timeout: float = constants.DEFAULT_ETAG_TIMEOUT,
    force_download: bool = False,
    token: bool | str | None = None,
    local_files_only: bool = False,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    max_workers: int = 8,
    tqdm_class: type[base_tqdm] | None = None,
    headers: dict[str, str] | None = None,
    endpoint: str | None = None,
    dry_run: Literal[False] = False,
) -> str: ...


@overload
def snapshot_download(
    repo_id: str,
    *,
    repo_type: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    library_name: str | None = None,
    library_version: str | None = None,
    user_agent: dict | str | None = None,
    etag_timeout: float = constants.DEFAULT_ETAG_TIMEOUT,
    force_download: bool = False,
    token: bool | str | None = None,
    local_files_only: bool = False,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    max_workers: int = 8,
    tqdm_class: type[base_tqdm] | None = None,
    headers: dict[str, str] | None = None,
    endpoint: str | None = None,
    dry_run: Literal[True] = True,
) -> list[DryRunFileInfo]: ...


@overload
def snapshot_download(
    repo_id: str,
    *,
    repo_type: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    library_name: str | None = None,
    library_version: str | None = None,
    user_agent: dict | str | None = None,
    etag_timeout: float = constants.DEFAULT_ETAG_TIMEOUT,
    force_download: bool = False,
    token: bool | str | None = None,
    local_files_only: bool = False,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    max_workers: int = 8,
    tqdm_class: type[base_tqdm] | None = None,
    headers: dict[str, str] | None = None,
    endpoint: str | None = None,
    dry_run: bool = False,
) -> str | list[DryRunFileInfo]: ...


@validate_hf_hub_args
def snapshot_download(
    repo_id: str,
    *,
    repo_type: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    library_name: str | None = None,
    library_version: str | None = None,
    user_agent: dict | str | None = None,
    etag_timeout: float = constants.DEFAULT_ETAG_TIMEOUT,
    force_download: bool = False,
    token: bool | str | None = None,
    local_files_only: bool = False,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    max_workers: int = 8,
    tqdm_class: type[base_tqdm] | None = None,
    headers: dict[str, str] | None = None,
    endpoint: str | None = None,
    dry_run: bool = False,
) -> str | list[DryRunFileInfo]:
    """Download repo files.

    Download a whole snapshot of a repo's files at the specified revision. This is useful when you want all files from
    a repo because you don't know which ones you will need _a priori_. All files are nested in a folder to keep their
    path and filename relative to that folder. You can also filter which files to download by using `allow_patterns`
    and `ignore_patterns`.

    If `local_dir` is provided, the file structure from the repo will be replicated in this location. When using this
    option, the `cache_dir` will not be used, and a `.cache/huggingface/` folder will be created at the root of `local_dir`
    to store some metadata related to the downloaded files. While this mechanism is not as robust as the main
    cache system, it's optimized for regularly pulling the latest version of a repository.

    An alternative would be to clone the repo, but this requires git and git-lfs to be installed and properly
    configured. It is also not possible to filter which files to download when cloning a repository using git.

    Args:
        repo_id (`str`):
            A user or an organization name and a repo name separated by a `/`.
        repo_type (`str`, *optional*):
            Set to `"dataset"`, `"space"` or `"kernel"` if downloading from a dataset, space or kernel repo,
            `None` or `"model"` if downloading from a model. Default is `None`.
        revision (`str`, *optional*):
            An optional Git revision id, which can be a branch name, a tag, or a
            commit hash.
        cache_dir (`str`, `Path`, *optional*):
            Path to the folder where cached files are stored.
        local_dir (`str` or `Path`, *optional*):
            If provided, the downloaded files will be placed under this directory.
        library_name (`str`, *optional*):
            The name of the library to which the object corresponds.
        library_version (`str`, *optional*):
            The version of the library.
        user_agent (`str`, `dict`, *optional*):
            The user-agent info in the form of a dictionary or a string.
        etag_timeout (`float`, *optional*, defaults to `10`):
            When fetching ETag, how many seconds to wait for the server to send
            data before giving up, which is passed to `httpx.request`.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether the file should be downloaded even if it already exists in the local cache.
        token (`str`, `bool`, *optional*):
            A token to be used for the download.
                - If `True`, the token is read from the HuggingFace config
                  folder.
                - If a string, it's used as the authentication token.
        headers (`dict`, *optional*):
            Additional headers to include in the request. Those headers take precedence over the others.
        endpoint (`str`, *optional*):
            The Hub endpoint to send the request to. Defaults to the value of `HF_ENDPOINT`.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, do not download any files even if they are not in `cache_dir` or `local_dir`.
        allow_patterns (`list[str]` or `str`, *optional*):
            If provided, only files matching at least one pattern are downloaded.
        ignore_patterns (`list[str]` or `str`, *optional*):
            If provided, files matching any of the patterns are not downloaded.
        max_workers (`int`, *optional*):
            Number of concurrent threads to download files (1 thread = 1 file download).
            Defaults to 8.
        tqdm_class (`tqdm`, *optional*):
            If provided, overwrites the default behavior for the progress bar. Passed
            argument must inherit from `tqdm.auto.tqdm` or at least mimic its behavior.
            Note that the `tqdm_class` is not passed to each individual download.
            Defaults to the custom HF progress bar that can be disabled by setting
            `HF_HUB_DISABLE_PROGRESS_BARS` environment variable.
        dry_run (`bool`, *optional*, defaults to `False`):
            If `True`, perform a dry run without actually downloading the files. Returns a list of
            [`DryRunFileInfo`] objects containing information about what would be downloaded.

    Returns:
        `str` or list of [`DryRunFileInfo`]:
            - If `dry_run=False`: Local snapshot path.
            - If `dry_run=True`: A list of [`DryRunFileInfo`] objects containing download information.

    Raises:
        [`~utils.RepositoryNotFoundError`]
            If the repository to download from cannot be found. This may be because it doesn't exist
            or because it is set to `private` and you do not have access.
        [`~utils.RevisionNotFoundError`]
            If the revision to download from cannot be found.
        [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
            If `token=True` and the token cannot be found.
        [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError) if
            ETag cannot be determined.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If some parameter value is invalid.
    """
    if cache_dir is None:
        cache_dir = constants.HF_HUB_CACHE
    if revision is None:
        revision = constants.DEFAULT_REVISION
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if repo_type is None:
        repo_type = "model"
    if repo_type not in constants.REPO_TYPES_WITH_KERNEL:
        raise ValueError(
            f"Invalid repo type: {repo_type}. Accepted repo types are: {str(constants.REPO_TYPES_WITH_KERNEL)}"
        )

    storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))

    api = HfApi(
        library_name=library_name,
        library_version=library_version,
        user_agent=user_agent,
        endpoint=endpoint,
        headers=headers,
        token=token,
    )

    repo_info: ModelInfo | DatasetInfo | SpaceInfo | KernelInfo | None = None
    api_call_error: Exception | None = None
    if not local_files_only:
        # try/except logic to handle different errors => taken from `hf_hub_download`
        try:
            # if we have internet connection we want to list files to download
            repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
        except httpx.ProxyError:
            # Actually raise on proxy error
            raise
        except (httpx.ConnectError, httpx.TimeoutException, OfflineModeIsEnabled) as error:
            # Internet connection is down
            # => will try to use local files only
            api_call_error = error
            pass
        except RevisionNotFoundError:
            # The repo was found but the revision doesn't exist on the Hub (never existed or got deleted)
            raise
        except HfHubHTTPError as error:
            # Multiple reasons for an http error:
            # - Repository is private and invalid/missing token sent
            # - Repository is gated and invalid/missing token sent
            # - Hub is down (error 500 or 504)
            # => let's switch to 'local_files_only=True' to check if the files are already cached.
            #    (if it's not the case, the error will be re-raised)
            api_call_error = error
            pass

    # At this stage, if `repo_info` is None it means either:
    # - internet connection is down
    # - internet connection is deactivated (local_files_only=True or HF_HUB_OFFLINE=True)
    # - repo is private/gated and invalid/missing token sent
    # - Hub is down
    # => let's look if we can find the appropriate folder in the cache:
    #    - if the specified revision is a commit hash, look inside "snapshots".
    #    - f the specified revision is a branch or tag, look inside "refs".
    # => if local_dir is not None, we will return the path to the local folder if it exists.
    if repo_info is None:
        if dry_run:
            raise DryRunError(
                "Dry run cannot be performed as the repository cannot be accessed. Please check your internet connection or authentication token."
            ) from api_call_error

        # Try to get which commit hash corresponds to the specified revision
        commit_hash = None
        if REGEX_COMMIT_HASH.match(revision):
            commit_hash = revision
        else:
            ref_path = os.path.join(storage_folder, "refs", revision)
            if os.path.exists(ref_path):
                # retrieve commit_hash from refs file
                with open(ref_path) as f:
                    commit_hash = f.read()

        # Try to locate snapshot folder for this commit hash
        if commit_hash is not None and local_dir is None:
            snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
            if os.path.exists(snapshot_folder):
                # Snapshot folder exists => let's return it
                # (but we can't check if all the files are actually there)
                return snapshot_folder

        # If local_dir is not None, return it if it exists and is not empty
        if local_dir is not None:
            local_dir = Path(local_dir)
            if local_dir.is_dir() and any(local_dir.iterdir()):
                logger.warning(
                    f"Returning existing local_dir `{local_dir}` as remote repo cannot be accessed in `snapshot_download` ({api_call_error})."
                )
                return str(local_dir.resolve())
        # If we couldn't find the appropriate folder on disk, raise an error.
        if local_files_only:
            raise LocalEntryNotFoundError(
                "Cannot find an appropriate cached snapshot folder for the specified revision on the local disk and "
                "outgoing traffic has been disabled. To enable repo look-ups and downloads online, pass "
                "'local_files_only=False' as input."
            )
        elif isinstance(api_call_error, OfflineModeIsEnabled):
            raise LocalEntryNotFoundError(
                "Cannot find an appropriate cached snapshot folder for the specified revision on the local disk and "
                "outgoing traffic has been disabled. To enable repo look-ups and downloads online, set "
                "'HF_HUB_OFFLINE=0' as environment variable."
            ) from api_call_error
        elif isinstance(api_call_error, (RepositoryNotFoundError, GatedRepoError)) or (
            isinstance(api_call_error, HfHubHTTPError) and api_call_error.response.status_code == 401
        ):
            # Repo not found, gated, or specific authentication error => let's raise the actual error
            raise api_call_error
        else:
            # Otherwise: most likely a connection issue or Hub downtime => let's warn the user
            raise LocalEntryNotFoundError(
                f"Got: {api_call_error.__class__.__name__}: {api_call_error}"
                "\nAn error happened while trying to locate the files on the Hub, and we cannot find the appropriate"
                " snapshot folder for the specified revision on the local disk. Please check your internet connection"
                " and try again."
            ) from api_call_error

    # At this stage, internet connection is up and running
    # => let's download the files!
    assert repo_info.sha is not None, "Repo info returned from server must have a revision sha."

    commit_hash = repo_info.sha

    # Always work from the full tree listing of the resolved commit. The tree of a commit is immutable,
    # so the listing is cached on disk (under `<storage_folder>/trees/<commit_hash>.json`) and fetched at
    # most once per commit. The listing provides each file's metadata (size, etag, xet hash), which lets
    # `hf_hub_download` skip its per-file HEAD call.
    tree_entries = read_tree_cache(storage_folder, commit_hash)
    if tree_entries is None:
        tree_entries = {
            f.path: TreeCacheEntry(
                path=f.path,
                size=f.size,
                blob_id=f.blob_id,
                lfs_sha256=f.lfs.sha256 if f.lfs is not None else None,
                lfs_size=f.lfs.size if f.lfs is not None else None,
                xet_hash=f.xet_hash,
            )
            for f in api.list_repo_tree(repo_id=repo_id, recursive=True, revision=commit_hash, repo_type=repo_type)
            if isinstance(f, RepoFile)
        }
        write_tree_cache(storage_folder, commit_hash, repo_id, repo_type, tree_entries)

    filtered_repo_files = list(
        filter_repo_objects(
            items=tree_entries.keys(),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    )

    tqdm_desc = f"Fetching {len(filtered_repo_files)} files"
    if dry_run:
        tqdm_desc = "[dry-run] " + tqdm_desc

    # On public repos, `GET /resolve/...` always answers with a 307 redirect to the resolve-cache
    # endpoint, costing 2 requests per file. The redirect target is deterministic (commit hash, path
    # and etag — all known from the tree listing), so build it directly and save the redirect hop.
    is_public = getattr(repo_info, "private", None) is False

    def _download_location(repo_file: str, entry: TreeCacheEntry) -> str:
        resolve_url = hf_hub_url(repo_id, repo_file, repo_type=repo_type, revision=commit_hash, endpoint=endpoint)
        if not is_public or entry.lfs_sha256 is not None:
            # Private repos serve content directly from /resolve (no redirect). LFS files redirect
            # to the CDN with a signed URL we cannot build ourselves.
            return resolve_url
        base = endpoint if endpoint is not None else constants.ENDPOINT
        resolve_path = urlparse(resolve_url).path
        return (
            f"{base.rstrip('/')}/api/resolve-cache/{repo_type}s/{repo_id}/{commit_hash}/{repo_file}"
            f"?{quote(resolve_path, safe='')}=&etag=%22{entry.etag}%22"
        )

    def _file_metadata(repo_file: str) -> HfFileMetadata:
        """Build the same metadata the `/resolve` HEAD endpoint would return, from the tree listing entry."""
        entry = tree_entries[repo_file]
        xet_file_data = None
        if entry.xet_hash is not None:
            xet_file_data = XetFileData(
                file_hash=entry.xet_hash,
                refresh_route=xet_connection_info_refresh_url(
                    token_type=XetTokenType.READ,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=commit_hash,
                    endpoint=endpoint,
                ),
            )
        return HfFileMetadata(
            commit_hash=commit_hash,
            etag=entry.etag,
            location=_download_location(repo_file, entry),
            size=entry.file_size,
            xet_file_data=xet_file_data,
        )
    snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
    # if passed revision is not identical to commit_hash
    # then revision has to be a branch name or tag name.
    # In that case store a ref.
    if revision != commit_hash:
        ref_path = os.path.join(storage_folder, "refs", revision)
        try:
            os.makedirs(os.path.dirname(ref_path), exist_ok=True)
            with open(ref_path, "w") as f:
                f.write(commit_hash)
        except OSError as e:
            logger.warning(f"Ignored error while writing commit hash to {ref_path}: {e}.")

    # Download all missing xet files in a single xet download group, so the xet access token is
    # fetched once for the whole snapshot instead of once per file (hf_xet requests one token per
    # download group). Blobs land directly in the cache; the per-file loop below then only creates
    # symlinks for them (no HTTP call). Best-effort: falls back to per-file downloads on any error.
    if not dry_run and local_dir is None:
        _prefetch_xet_blobs(
            storage_folder=storage_folder,
            tree_entries=tree_entries,
            filenames=filtered_repo_files,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_hash=commit_hash,
            endpoint=endpoint,
            token=token,
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
            headers=headers,
            force_download=force_download,
        )

    results: list[str | DryRunFileInfo] = []

    # User can use its own tqdm class or the default one from `huggingface_hub.utils`
    tqdm_class = tqdm_class or hf_tqdm

    # Create a progress bar for the bytes downloaded
    # This progress bar is shared across threads/files and gets updated each time we fetch
    # metadata for a file.
    bytes_progress = _create_progress_bar(
        cls=tqdm_class,
        log_level=logger.getEffectiveLevel(),
        name="huggingface_hub.snapshot_download",
        desc="Downloading (incomplete total...)",
        total=0,
        initial=0,
        unit="B",
        unit_scale=True,
    )

    class _AggregatedTqdm:
        """Fake tqdm object to aggregate progress into the parent `bytes_progress` bar.

        In practice, the `_AggregatedTqdm` object won't be displayed; it's just used to update
        the `bytes_progress` bar from each thread/file download.
        """

        def __init__(self, *args, **kwargs):
            # Adjust the total of the parent progress bar
            total = kwargs.pop("total", None)
            if total is not None:
                bytes_progress.total += total
                bytes_progress.refresh()

            # Adjust initial of the parent progress bar
            initial = kwargs.pop("initial", 0)
            if initial:
                bytes_progress.update(initial)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        def update(self, n: int | float | None = 1) -> None:
            bytes_progress.update(n)

    # we pass the commit_hash to hf_hub_download
    # so no network call happens if we already
    # have the file locally.
    def _inner_hf_hub_download(repo_file: str) -> None:
        results.append(
            hf_hub_download(  # type: ignore
                repo_id,
                filename=repo_file,
                repo_type=repo_type,
                revision=commit_hash,
                endpoint=endpoint,
                cache_dir=cache_dir,
                local_dir=local_dir,
                library_name=library_name,
                library_version=library_version,
                user_agent=user_agent,
                etag_timeout=etag_timeout,
                force_download=force_download,
                token=token,
                headers=headers,
                tqdm_class=_AggregatedTqdm,  # type: ignore
                dry_run=dry_run,
                file_metadata=_file_metadata(repo_file),
            )
        )

    thread_map(
        _inner_hf_hub_download,
        filtered_repo_files,
        desc=tqdm_desc,
        max_workers=max_workers,
        tqdm_class=tqdm_class,
    )

    bytes_progress.set_description("Download complete")

    if dry_run:
        assert all(isinstance(r, DryRunFileInfo) for r in results)
        return results  # type: ignore

    if local_dir is not None:
        return str(os.path.realpath(local_dir))
    return snapshot_folder
def _prefetch_xet_blobs(
    *,
    storage_folder: str,
    tree_entries: dict[str, TreeCacheEntry],
    filenames: list[str],
    repo_id: str,
    repo_type: str,
    commit_hash: str,
    endpoint: str | None,
    token: bool | str | None,
    library_name: str | None,
    library_version: str | None,
    user_agent: dict | str | None,
    headers: dict[str, str] | None,
    force_download: bool,
) -> None:
    """Download all missing xet blobs in a single xet download group (one access token fetch). Best-effort."""
    from .utils import build_hf_headers
    from .utils._xet import get_xet_session, xet_headers_without_auth

    to_fetch: dict[str, TreeCacheEntry] = {}  # blob path -> entry (deduplicated by etag)
    for filename in filenames:
        entry = tree_entries[filename]
        if entry.xet_hash is None:
            continue
        blob_path = os.path.join(storage_folder, "blobs", entry.etag)
        if not force_download and os.path.exists(blob_path):
            continue
        to_fetch[blob_path] = entry
    if not to_fetch:
        return
    try:
        from hf_xet import XetFileInfo

        hf_headers = build_hf_headers(
            token=token,
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
            headers=headers,
        )
        refresh_route = xet_connection_info_refresh_url(
            token_type=XetTokenType.READ,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=commit_hash,
            endpoint=endpoint,
        )
        os.makedirs(os.path.join(storage_folder, "blobs"), exist_ok=True)
        incomplete_paths = {blob_path: f"{blob_path}.{os.getpid()}.xet.incomplete" for blob_path in to_fetch}
        session = get_xet_session()
        with session.new_file_download_group(
            token_refresh_url=refresh_route,
            token_refresh_headers=hf_headers,
            custom_headers=xet_headers_without_auth(hf_headers),
        ) as group:
            for blob_path, entry in to_fetch.items():
                group.start_download_file(
                    XetFileInfo(entry.xet_hash, entry.file_size), os.path.abspath(incomplete_paths[blob_path])
                )
        for blob_path, incomplete_path in incomplete_paths.items():
            os.replace(incomplete_path, blob_path)
        logger.info(f"Fetched {len(to_fetch)} xet files in a single download group.")
    except Exception as e:
        logger.warning(f"Batch xet download failed, falling back to per-file downloads: {e}")
