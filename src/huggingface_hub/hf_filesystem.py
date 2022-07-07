import tempfile
from functools import partial
from pathlib import PurePosixPath
from typing import Optional

import fsspec

from .constants import (
    HUGGINGFACE_HUB_CACHE,
    REPO_TYPE_DATASET,
    REPO_TYPE_MODEL,
    REPO_TYPE_SPACE,
    REPO_TYPES,
)
from .file_download import hf_hub_url
from .hf_api import (
    HfFolder,
    dataset_info,
    delete_file,
    model_info,
    space_info,
    upload_file,
)


def _repo_type_to_info_func(repo_type):
    if repo_type == REPO_TYPE_DATASET:
        return dataset_info
    elif repo_type == REPO_TYPE_MODEL:
        return model_info
    elif repo_type == REPO_TYPE_SPACE:
        return space_info
    else:  # None
        return model_info


class HfFileSystem(fsspec.AbstractFileSystem):
    """
    Access a remote Hugging Face Hub repository as if were a local file system.

    Args:
        repo_id (`str`):
            The remote repository to access as if were a local file system,
            for example: `"username/custom_transformers"`
        token (`str`, *optional*):
            Authentication token, obtained with `HfApi.login` method. Will
            default to the stored token.
        repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if the remote repositry is a dataset or
            space repositroy, `None` or `"model"` if it is a model repository. Default is
            `None`.
        revision (`str`, *optional*):
            An optional Git revision id which can be a branch name, a tag, or a
            commit hash. Defaults to the head of the `"main"` branch.

    Example usage (direct):

    ```python
    >>> from huggingface_hub import HfFileSystem

    >>> hffs = HfFileSystem("username/my-dataset", repo_type="dataset")

    >>> # Read a remote file
    >>> with hffs.open("remote/file/in/repo.bin") as f:
    ...     data = f.read()

    >>> # Write a remote file
    >>> with hffs.open("remote/file/in/repo.bin", "wb") as f:
    ...     f.write(data)
    ```

    Example usage (via [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/)):

    ```python
    >>> import fsspec

    >>> # Read a remote file
    >>> with fsspec.open("hf:username/my-dataset:/remote/file/in/repo.bin", repo_type="dataset") as f:
    ...     data = f.read()

    >>> # Write a remote file
    >>> with fsspec.open("hf:username/my-dataset:/remote/file/in/repo.bin", "wb", repo_type="dataset") as f:
    ...     f.write(data)
    ```
    """

    root_marker = ""
    protocol = "hf"

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(self, **kwargs)

        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")

        self.repo_id = repo_id
        self.token = token if token is not None else HfFolder.get_token()
        self.repo_type = repo_type
        self.revision = revision
        # Cached attributes
        self._repo_info = None
        self._repo_entries_spec = None

    def _get_repo_info(self):
        if self._repo_info is None:
            self._repo_info = _repo_type_to_info_func(self.repo_type)(
                self.repo_id, revision=self.revision, token=self.token
            )

    def _get_repo_entries_spec(self):
        if self._repo_entries_spec is None:
            self._get_repo_info()
            self._repo_entries_spec = {}
            for hf_file in self._repo_info.siblings:
                # TODO(QL): add sizes
                self._repo_entries_spec[hf_file.rfilename] = {
                    "name": hf_file.rfilename,
                    "size": None,
                    "type": "file",
                }
                self._repo_entries_spec.update(
                    {
                        str(d): {"name": str(d), "size": None, "type": "directory"}
                        for d in list(PurePosixPath(hf_file.rfilename).parents)[:-1]
                    }
                )

    def _invalidate_repo_cache(self):
        self._repo_info = None
        self._repo_entries_spec = None

    @classmethod
    def _strip_protocol(cls, path):
        path = super()._strip_protocol(path).lstrip("/")
        if ":/" in path:
            path = path.split(":", 1)[1]
        return path.lstrip("/")

    @staticmethod
    def _get_kwargs_from_urls(path):
        if path.startswith("hf://"):
            path = path[5:]
        out = {"repo_id": path}
        if ":/" in path:
            out["repo_id"], out["path"] = path.split(":/", 1)
        if "@" in out["repo_id"]:
            out["repo_id"], out["revision"] = out["repo_id"].split("@", 1)
        return out

    def _open(
        self,
        path: str,
        mode: str = "rb",
        **kwargs,
    ):
        if mode == "rb":
            self._get_repo_info()
            url = hf_hub_url(
                self.repo_id,
                path,
                repo_type=self.repo_type,
                revision=self.revision,
            )
            return fsspec.open(
                url,
                mode=mode,
                headers={"authorization": f"Bearer {self.token}"},
            ).open()
        else:
            return TempFileUploader(self, path, mode=mode)

    def _rm(self, path):
        path = self._strip_protocol(path)
        delete_file(
            path_in_repo=path,
            repo_id=self.repo_id,
            token=self.token,
            repo_type=self.repo_type,
            revision=self.revision,
        )
        self._invalidate_repo_cache()

    def info(self, path, **kwargs):
        self._get_repo_entries_spec()
        path = self._strip_protocol(path)
        if path in self._repo_entries_spec:
            return self._repo_entries_spec[path]
        else:
            raise FileNotFoundError(path)

    def ls(self, path, detail=False, **kwargs):
        self._get_repo_entries_spec()
        path = PurePosixPath(path.strip("/"))
        paths = {}
        for p, f in self._repo_entries_spec.items():
            p = PurePosixPath(p.strip("/"))
            root = p.parent
            if root == path:
                paths[str(p)] = f
        out = list(paths.values())
        if detail:
            return out
        else:
            return list(sorted(f["name"] for f in out))


class TempFileUploader(fsspec.spec.AbstractBufferedFile):
    def _initiate_upload(self):
        self.temp_file = tempfile.TemporaryFile(dir=HUGGINGFACE_HUB_CACHE)
        if self.mode == "ab":
            with self.fs.open(self.path, "rb") as f:
                for block in iter(partial(f.read, self.blocksize), b""):
                    self.temp_file.write(block)

    def _upload_chunk(self, final=False):
        self.buffer.seek(0)
        block = self.buffer.read()
        self.temp_file.write(block)
        if final:
            upload_file(
                path_or_fileobj=self.temp_file.file,
                path_in_repo=self.path,
                repo_id=self.fs.repo_id,
                token=self.fs.token,
                repo_type=self.fs.repo_type,
                revision=self.fs.revision,
            )
            self.fs._invalidate_repo_cache()
            self.temp_file.close()
