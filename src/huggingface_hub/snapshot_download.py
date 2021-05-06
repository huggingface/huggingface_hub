import os
from pathlib import Path
from typing import Dict, Optional, Union

from .constants import HUGGINGFACE_HUB_CACHE
from .file_download import cached_download, hf_hub_url
from .hf_api import HfApi


REPO_ID_SEPARATOR = "__"
# ^ make sure this substring is not allowed in repo_ids on hf.co


def snapshot_download(
    repo_id: str,
    revision: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    user_agent: Union[Dict, str, None] = None,
) -> str:
    """
    Downloads a whole snapshot of a repo's files at the specified revision.
    This is useful when you want all files from a repo, because you don't know
    which ones you will need a priori.
    All files are nested inside a folder in order to keep their actual filename
    relative to that folder.

    An alternative would be to just clone a repo but this would require that
    the user always has git and git-lfs installed, and properly configured.

    Note: at some point maybe this format of storage should actually replace
    the flat storage structure we've used so far (initially from allennlp
    if I remember correctly).

    Return:
        Local folder path (string) of repo snapshot
    """
    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    _api = HfApi()
    model_info = _api.model_info(repo_id=repo_id, revision=revision)

    storage_folder = os.path.join(
        cache_dir, repo_id.replace("/", REPO_ID_SEPARATOR) + "." + model_info.sha
    )

    for model_file in model_info.siblings:
        url = hf_hub_url(
            repo_id, filename=model_file.rfilename, revision=model_info.sha
        )
        relative_filepath = os.path.join(*model_file.rfilename.split("/"))

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
        )

        if os.path.exists(path + ".lock"):
            os.remove(path + ".lock")

    return storage_folder
