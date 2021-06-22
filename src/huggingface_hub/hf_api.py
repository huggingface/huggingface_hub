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


import os
import re
import sys
import warnings
from io import BufferedIOBase, RawIOBase
from os.path import expanduser
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union

import requests
from requests.exceptions import HTTPError

from .constants import REPO_TYPES, REPO_TYPES_URL_PREFIXES


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


ENDPOINT = "https://huggingface.co"
REMOTE_FILEPATH_REGEX = re.compile(r"^\w[\w\/]*(\.\w+)?$")
# ^^ No trailing slash, no backslash, no spaces, no relative parts ("." or "..")
#    Only word characters and an optional extension


class RepoObj:
    """
    HuggingFace git-based system, data structure that represents a file belonging to the current user.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        items = (f"{k}='{v}'" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({', '.join(items)})"


class ModelFile:
    """
    Data structure that represents a public file inside a model, accessible from huggingface.co
    """

    def __init__(self, rfilename: str, **kwargs):
        self.rfilename = rfilename  # filename relative to the model root
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        items = (f"{k}='{v}'" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({', '.join(items)})"


class ModelInfo:
    """
    Info about a public model accessible from huggingface.co
    """

    def __init__(
        self,
        modelId: Optional[str] = None,  # id of model
        sha: Optional[str] = None,  # commit sha at the specified revision
        lastModified: Optional[str] = None,  # date of last commit to repo
        tags: List[str] = [],
        pipeline_tag: Optional[str] = None,
        siblings: Optional[
            List[Dict]
        ] = None,  # list of files that constitute the model
        config: Optional[Dict] = None,  # information about model configuration
        **kwargs,
    ):
        self.modelId = modelId
        self.sha = sha
        self.lastModified = lastModified
        self.tags = tags
        self.pipeline_tag = pipeline_tag
        self.siblings = (
            [ModelFile(**x) for x in siblings] if siblings is not None else None
        )
        self.config = config
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        s = f"{self.__class__.__name__}:" + " {"
        for key, val in self.__dict__.items():
            s += f"\n\t{key}: {val}"
        return s + "\n}"

    def __str__(self):
        r = f"Model Name: {self.modelId}, Tags: {self.tags}"
        if self.pipeline_tag:
            r += f", Task: {self.pipeline_tag}"
        return r


class HfApi:
    def __init__(self, endpoint=None):
        self.endpoint = endpoint if endpoint is not None else ENDPOINT

    def login(self, username: str, password: str) -> str:
        """
        Call HF API to sign in a user and get a token if credentials are valid.

        Outputs: token if credentials are valid

        Throws: requests.exceptions.HTTPError if credentials are invalid
        """
        path = "{}/api/login".format(self.endpoint)
        r = requests.post(path, json={"username": username, "password": password})
        r.raise_for_status()
        d = r.json()
        return d["token"]

    def whoami(self, token: str) -> Tuple[str, List[str]]:
        """
        Call HF API to know "whoami"
        """
        path = "{}/api/whoami".format(self.endpoint)
        r = requests.get(path, headers={"authorization": "Bearer {}".format(token)})
        r.raise_for_status()
        d = r.json()
        return d["user"], d["orgs"]

    def logout(self, token: str) -> None:
        """
        Call HF API to log out.
        """
        path = "{}/api/logout".format(self.endpoint)
        r = requests.post(path, headers={"authorization": "Bearer {}".format(token)})
        r.raise_for_status()

    def list_models(
        self,
        filter: Union[str, Iterable[str], None] = None,
        sort: Union[Literal["lastModified"], str, None] = None,
        direction: Optional[Literal[-1]] = None,
        limit: Optional[int] = None,
        full: Optional[bool] = None,
    ) -> List[ModelInfo]:
        """
        Get the public list of all the models on huggingface.co

        Args:
            filter (:obj:`str` or :class:`Iterable`, `optional`):
                A string which can be used to identify models on the hub by their tags.
                Example usage:

                    >>> from huggingface_hub import HfApi
                    >>> api = HfApi()

                    >>> # List all models
                    >>> api.list_models()

                    >>> # List only the text classification models
                    >>> api.list_models(filter="text-classification")

                    >>> # List only the russian models compatible with pytorch
                    >>> api.list_models(filter=("ru", "pytorch"))

                    >>> # List only the models trained on the "common_voice" dataset
                    >>> api.list_models(filter="dataset:common_voice")

                    >>> # List only the models from the AllenNLP library
                    >>> api.list_models(filter="allennlp")
            sort (:obj:`Literal["lastModified"]` or :obj:`str`, `optional`):
                The key with which to sort the resulting models. Possible values are the properties of the `ModelInfo`
                class.
            direction (:obj:`Literal[-1]` or :obj:`int`, `optional`):
                Direction in which to sort. The value `-1` sorts by descending order while all other values
                sort by ascending order.
            limit (:obj:`int`, `optional`):
                The limit on the number of models fetched. Leaving this option to `None` fetches all models.
            full (:obj:`bool`, `optional`):
                Whether to fetch all model data, including the `lastModified`, the `sha`, the files and the `tags`.
                This is set to `True` by default when using a filter.

        """
        path = "{}/api/models".format(self.endpoint)
        params = {}
        if filter is not None:
            params.update({"filter": filter})
            params.update({"full": True})
        if sort is not None:
            params.update({"sort": sort})
        if direction is not None:
            params.update({"direction": direction})
        if limit is not None:
            params.update({"limit": limit})
        if full is not None:
            if full:
                params.update({"full": True})
            elif "full" in params:
                del params["full"]
        r = requests.get(path, params=params)
        r.raise_for_status()
        d = r.json()
        return [ModelInfo(**x) for x in d]

    def model_list(self) -> List[ModelInfo]:
        """
        Deprecated method name, renamed to `list_models`.

        Get the public list of all the models on huggingface.co
        """
        warnings.warn(
            "This method has been renamed to `list_models` for consistency and will be removed in a future version."
        )
        return self.list_models()

    def model_info(
        self, repo_id: str, revision: Optional[str] = None, token: Optional[str] = None
    ) -> ModelInfo:
        """
        Get info on one specific model on huggingface.co

        Model can be private if you pass an acceptable token.
        """
        path = (
            "{}/api/models/{repo_id}".format(self.endpoint, repo_id=repo_id)
            if revision is None
            else "{}/api/models/{repo_id}/revision/{revision}".format(
                self.endpoint, repo_id=repo_id, revision=revision
            )
        )
        headers = (
            {"authorization": "Bearer {}".format(token)} if token is not None else None
        )
        r = requests.get(path, headers=headers)
        r.raise_for_status()
        d = r.json()
        return ModelInfo(**d)

    def list_repos_objs(
        self, token: str, organization: Optional[str] = None
    ) -> List[RepoObj]:
        """
        HuggingFace git-based system, used for models, datasets, and spaces.

        Call HF API to list all stored files for user (or one of their organizations).
        """
        path = "{}/api/repos/ls".format(self.endpoint)
        params = {"organization": organization} if organization is not None else None
        r = requests.get(
            path, params=params, headers={"authorization": "Bearer {}".format(token)}
        )
        r.raise_for_status()
        d = r.json()
        return [RepoObj(**x) for x in d]

    def create_repo(
        self,
        token: str,
        name: str,
        organization: Optional[str] = None,
        private: Optional[bool] = None,
        repo_type: Optional[str] = None,
        exist_ok=False,
        lfsmultipartthresh: Optional[int] = None,
    ) -> str:
        """
        HuggingFace git-based system, used for models, datasets, and spaces.

        Call HF API to create a whole repo.

        Params:
            private: Whether the model repo should be private (requires a paid huggingface.co account)

            repo_type: Set to "dataset" or "space" if creating a dataset or space, default is model

            exist_ok: Do not raise an error if repo already exists

            lfsmultipartthresh: Optional: internal param for testing purposes.

        Returns:
            URL to the newly created repo.
        """
        path = "{}/api/repos/create".format(self.endpoint)

        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        json = {"name": name, "organization": organization, "private": private}
        if repo_type is not None:
            json["type"] = repo_type
        if lfsmultipartthresh is not None:
            json["lfsmultipartthresh"] = lfsmultipartthresh
        r = requests.post(
            path,
            headers={"authorization": "Bearer {}".format(token)},
            json=json,
        )

        try:
            r.raise_for_status()
        except HTTPError as err:
            if not (exist_ok and err.response.status_code == 409):
                try:
                    additional_info = r.json().get("error", None)
                    if additional_info:
                        new_err = f"{err.args[0]} - {additional_info}"
                        err.args = (new_err,) + err.args[1:]
                except ValueError:
                    pass

                raise err

        d = r.json()
        return d["url"]

    def delete_repo(
        self,
        token: str,
        name: str,
        organization: Optional[str] = None,
        repo_type: Optional[str] = None,
    ):
        """
        HuggingFace git-based system, used for models, datasets, and spaces.

        Call HF API to delete a whole repo.

        CAUTION(this is irreversible).
        """
        path = "{}/api/repos/delete".format(self.endpoint)

        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        json = {"name": name, "organization": organization}
        if repo_type is not None:
            json["type"] = repo_type

        r = requests.delete(
            path,
            headers={"authorization": "Bearer {}".format(token)},
            json=json,
        )
        r.raise_for_status()

    def update_repo_visibility(
        self,
        token: str,
        name: str,
        private: bool,
        organization: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Update the visibility setting of a repository.
        """
        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        if organization is None:
            namespace, _ = self.whoami(token)
        else:
            namespace = organization

        path_prefix = "{}/api/".format(self.endpoint)
        if repo_type in REPO_TYPES_URL_PREFIXES:
            path_prefix += REPO_TYPES_URL_PREFIXES[repo_type]

        path = "{}{}/{}/settings".format(path_prefix, namespace, name)
        json = {"private": private}

        r = requests.put(
            path,
            headers={"authorization": "Bearer {}".format(token)},
            json=json,
        )
        r.raise_for_status()
        return r.json()

    def upload_file(
        self,
        token: str,
        path_or_fileobj: Union[str, BinaryIO],
        path_in_repo: str,
        repo_id: str,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        identical_ok: bool = True,
    ) -> str:
        """
        Upload a local file (up to 5GB) to the given repo. The upload is done through a HTTP post request, and
        doesn't require git or git-lfs to be installed.

        Params:
            token (``str``):
                Authentication token, obtained with :function:`HfApi.login` method

            path_or_fileobj (``str`` or ``BinaryIO``):
                Path to a file on the local machine or binary data stream / fileobj.

            path_in_repo (``str``):
                Relative filepath in the repo, for example: :obj:`"checkpoints/1fec34a/weights.bin"`

            repo_id (``str``):
                The repository to which the file will be uploaded, for example: :obj:`"username/custom_transformers"`

            repo_type (``str``, Optional):
                Set to :obj:`"dataset"` or :obj:`"space"` if uploading to a dataset or space, :obj:`None` if uploading to a model. Default is :obj:`None`.

            revision (``str``, Optional):
                The git revision to commit from. Defaults to the :obj:`"main"` branch.

            identical_ok (``bool``, defaults to ``True``):
                When set to false, will raise an HTTPError when the file you're trying to upload already exists on the hub
                and its content did not change.

        Returns:
            ``str``: The URL to visualize the uploaded file on the hub

        Raises:
            :class:`ValueError`: if some parameter value is invalid

            :class:`requests.HTTPError`: if the HuggingFace API returned an error

        Examples:
            >>> with open("./local/filepath", "rb") as fobj:
            ...     upload_file(
            ...         path_or_fileobj=fileobj,
            ...         path_in_repo="remote/file/path.h5",
            ...         repo_id="username/my-dataset",
            ...         repo_type="datasets",
            ...         token="my_token",
            ...    )
            "https://huggingface.co/datasets/username/my-dataset/blob/main/remote/file/path.h5"

            >>> upload_file(
            ...     path_or_fileobj=".\\\\local\\\\file\\\\path",
            ...     path_in_repo="remote/file/path.h5",
            ...     repo_id="username/my-model",
            ...     token="my_token",
            ... )
            "https://huggingface.co/username/my-model/blob/main/remote/file/path.h5"


        """
        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type, must be one of {}".format(REPO_TYPES))

        # Validate path_or_fileobj
        if isinstance(path_or_fileobj, str):
            path_or_fileobj = os.path.normpath(os.path.expanduser(path_or_fileobj))
            if not os.path.isfile(path_or_fileobj):
                raise ValueError(
                    "Provided path: '{}' is not a file".format(path_or_fileobj)
                )
        elif not isinstance(path_or_fileobj, (RawIOBase, BufferedIOBase)):
            # ^^ Test from: https://stackoverflow.com/questions/44584829/how-to-determine-if-file-is-opened-in-binary-or-text-mode
            raise ValueError(
                "path_or_fileobj must be either an instance of str or BinaryIO. "
                "If you passed a fileobj, make sure you've opened the file in binary mode."
            )

        # Normalize path separators and strip leading slashes
        if not REMOTE_FILEPATH_REGEX.match(path_in_repo):
            raise ValueError(
                "Invalid path_in_repo '{}', path_in_repo must match regex {}".format(
                    path_in_repo, REMOTE_FILEPATH_REGEX.pattern
                )
            )

        if repo_type in REPO_TYPES_URL_PREFIXES:
            repo_id = REPO_TYPES_URL_PREFIXES[repo_type] + repo_id

        revision = revision if revision is not None else "main"

        path = "{}/api/{repo_id}/upload/{revision}/{path_in_repo}".format(
            self.endpoint,
            repo_id=repo_id,
            revision=revision,
            path_in_repo=path_in_repo,
        )

        headers = (
            {"authorization": "Bearer {}".format(token)} if token is not None else None
        )

        if isinstance(path_or_fileobj, str):
            with open(path_or_fileobj, "rb") as bytestream:
                r = requests.post(path, headers=headers, data=bytestream)
        else:
            r = requests.post(path, headers=headers, data=path_or_fileobj)

        try:
            r.raise_for_status()
        except HTTPError as err:
            if not (identical_ok and err.response.status_code == 409):
                raise err

        d = r.json()
        return d["url"]


class HfFolder:
    path_token = expanduser("~/.huggingface/token")

    @classmethod
    def save_token(cls, token):
        """
        Save token, creating folder as needed.
        """
        os.makedirs(os.path.dirname(cls.path_token), exist_ok=True)
        with open(cls.path_token, "w+") as f:
            f.write(token)

    @classmethod
    def get_token(cls):
        """
        Get token or None if not existent.
        """
        try:
            with open(cls.path_token, "r") as f:
                return f.read()
        except FileNotFoundError:
            pass

    @classmethod
    def delete_token(cls):
        """
        Delete token. Do not fail if token does not exist.
        """
        try:
            os.remove(cls.path_token)
        except FileNotFoundError:
            pass
