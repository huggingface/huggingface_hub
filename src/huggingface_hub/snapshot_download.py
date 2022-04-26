import os
from fnmatch import fnmatch
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

from .constants import DEFAULT_REVISION, HUGGINGFACE_HUB_CACHE
from .file_download import cached_download, hf_hub_url
from .hf_api import HfApi, HfFolder
from .utils import logging
from .utils._deprecation import _deprecate_positional_args


REPO_ID_SEPARATOR = "--"
# ^ this substring is not allowed in repo_ids on hf.co
# and is the canonical one we use for serialization of repo ids elsewhere.


logger = logging.get_logger(__name__)


@_deprecate_positional_args
def snapshot_download(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    user_agent: Optional[Union[Dict, str]] = None,
    proxies: Optional[Dict] = None,
    etag_timeout: Optional[float] = 10,
    resume_download: Optional[bool] = False,
    use_auth_token: Optional[Union[bool, str]] = None,
    local_files_only: Optional[bool] = False,
    allow_regex: Optional[Union[List[str], str]] = None,
    ignore_regex: Optional[Union[List[str], str]] = None,
) -> str:
    """Download all files of a repo.

    Downloads a whole snapshot of a repo's files at the specified revision. This
    is useful when you want all files from a repo, because you don't know which
    ones you will need a priori. All files are nested inside a folder in order
    to keep their actual filename relative to that folder.

    An alternative would be to just clone a repo but this would require that the
    user always has git and git-lfs installed, and properly configured.

    Args:
        repo_id (`str`):
            A user or an organization name and a repo name separated by a `/`.
        revision (`str`, *optional*):
            An optional Git revision id which can be a branch name, a tag, or a
            commit hash.
        cache_dir (`str`, `Path`, *optional*):
            Path to the folder where cached files are stored.
        library_name (`str`, *optional*):
            The name of the library to which the object corresponds.
        library_version (`str`, *optional*):
            The version of the library.
        user_agent (`str`, `dict`, *optional*):
            The user-agent info in the form of a dictionary or a string.
        proxies (`dict`, *optional*):
            Dictionary mapping protocol to the URL of the proxy passed to
            `requests.request`.
        etag_timeout (`float`, *optional*, defaults to `10`):
            When fetching ETag, how many seconds to wait for the server to send
            data before giving up which is passed to `requests.request`.
        resume_download (`bool`, *optional*, defaults to `False):
            If `True`, resume a previously interrupted download.
        use_auth_token (`str`, `bool`, *optional*):
            A token to be used for the download.
                - If `True`, the token is read from the HuggingFace config
                  folder.
                - If a string, it's used as the authentication token.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, avoid downloading the file and return the path to the
            local cached file if it exists.
        allow_regex (`list of str`, `str`, *optional*):
            If provided, only files matching this regex are downloaded.
        ignore_regex (`list of str`, `str`, *optional*):
            If provided, files matching this regex are not downloaded.

    Returns:
        Local folder path (string) of repo snapshot

    <Tip>

    Raises the following errors:

    - [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
      if `use_auth_token=True` and the token cannot be found.
    - [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError) if
      ETag cannot be determined.
    - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
      if some parameter value is invalid

    </Tip>
    """
    # Note: at some point maybe this format of storage should actually replace
    # the flat storage structure we've used so far (initially from allennlp
    # if I remember correctly).

    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE
    if revision is None:
        revision = DEFAULT_REVISION
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if isinstance(use_auth_token, str):
        token = use_auth_token
    elif use_auth_token:
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError(
                "You specified use_auth_token=True, but a Hugging Face token was not"
                " found."
            )
    else:
        token = None

    # remove all `/` occurrences to correctly convert repo to directory name
    repo_id_flattened = repo_id.replace("/", REPO_ID_SEPARATOR)

    # if we have no internet connection we will look for the
    # last modified folder in the cache
    if local_files_only:
        # possible repos have <path/to/cache_dir>/<flatten_repo_id> prefix
        repo_folders_prefix = os.path.join(cache_dir, repo_id_flattened)

        # list all possible folders that can correspond to the repo_id
        # and are of the format <flattened-repo-id>.<revision>.<commit-sha>
        # now let's list all cached repos that have to be included in the revision.
        # There are 3 cases that we have to consider.

        # 1) cached repos of format <repo_id>.{revision}.<any-hash>
        # -> in this case {revision} has to be a branch
        repo_folders_branch = glob(repo_folders_prefix + "." + revision + ".*")

        # 2) cached repos of format <repo_id>.{revision}
        # -> in this case {revision} has to be a commit sha
        repo_folders_commit_only = glob(repo_folders_prefix + "." + revision)

        # 3) cached repos of format <repo_id>.<any-branch>.{revision}
        # -> in this case {revision} also has to be a commit sha
        repo_folders_branch_commit = glob(repo_folders_prefix + ".*." + revision)

        # combine all possible fetched cached repos
        repo_folders = (
            repo_folders_branch + repo_folders_commit_only + repo_folders_branch_commit
        )

        if len(repo_folders) == 0:
            raise ValueError(
                "Cannot find the requested files in the cached path and outgoing"
                " traffic has been disabled. To enable model look-ups and downloads"
                " online, set 'local_files_only' to False."
            )

        # check if repo id was previously cached from a commit sha revision
        # and passed {revision} is not a commit sha
        # in this case snapshotting repos locally might lead to unexpected
        # behavior the user should be warned about

        # get all folders that were cached with just a sha commit revision
        all_repo_folders_from_sha = set(glob(repo_folders_prefix + ".*")) - set(
            glob(repo_folders_prefix + ".*.*")
        )
        # 1) is there any repo id that was previously cached from a commit sha?
        has_a_sha_revision_been_cached = len(all_repo_folders_from_sha) > 0
        # 2) is the passed {revision} is a branch
        is_revision_a_branch = (
            len(repo_folders_commit_only + repo_folders_branch_commit) == 0
        )

        if has_a_sha_revision_been_cached and is_revision_a_branch:
            # -> in this case let's warn the user
            logger.warn(
                f"The repo {repo_id} was previously downloaded from a commit hash"
                " revision and has created the following cached directories"
                f" {all_repo_folders_from_sha}. In this case, trying to load a repo"
                f" from the branch {revision} in offline mode might lead to unexpected"
                " behavior by not taking into account the latest commits."
            )

        # find last modified folder
        storage_folder = max(repo_folders, key=os.path.getmtime)

        # get commit sha
        repo_id_sha = storage_folder.split(".")[-1]
        model_files = os.listdir(storage_folder)
    else:
        # if we have internet connection we retrieve the correct folder name from the huggingface api
        _api = HfApi()
        model_info = _api.model_info(repo_id=repo_id, revision=revision, token=token)

        storage_folder = os.path.join(cache_dir, repo_id_flattened + "." + revision)

        # if passed revision is not identical to the commit sha
        # then revision has to be a branch name, e.g. "main"
        # in this case make sure that the branch name is included
        # cached storage folder name
        if revision != model_info.sha:
            storage_folder += f".{model_info.sha}"

        repo_id_sha = model_info.sha
        model_files = [f.rfilename for f in model_info.siblings]

    allow_regex = [allow_regex] if isinstance(allow_regex, str) else allow_regex
    ignore_regex = [ignore_regex] if isinstance(ignore_regex, str) else ignore_regex

    for model_file in model_files:
        # if there's an allowlist, skip download if file does not match any regex
        if allow_regex is not None and not any(
            fnmatch(model_file, r) for r in allow_regex
        ):
            continue

        # if there's a denylist, skip download if file does matches any regex
        if ignore_regex is not None and any(
            fnmatch(model_file, r) for r in ignore_regex
        ):
            continue

        url = hf_hub_url(repo_id, filename=model_file, revision=repo_id_sha)
        relative_filepath = os.path.join(*model_file.split("/"))

        # Create potential nested dir
        nested_dirname = os.path.dirname(
            os.path.join(storage_folder, relative_filepath)
        )
        os.makedirs(nested_dirname, exist_ok=True)

        path = cached_download(
            url,
            cache_dir=storage_folder,
            force_filename=relative_filepath,
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
            proxies=proxies,
            etag_timeout=etag_timeout,
            resume_download=resume_download,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
        )

        if os.path.exists(path + ".lock"):
            os.remove(path + ".lock")

    return storage_folder
