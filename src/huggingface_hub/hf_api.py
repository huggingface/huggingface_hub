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
import subprocess
import sys
import warnings
from io import BufferedIOBase, RawIOBase
from os.path import expanduser
from typing import IO, Dict, Iterable, List, Optional, Tuple, Union

import requests
from requests.exceptions import HTTPError

from .constants import (
    ENDPOINT,
    REPO_TYPES,
    REPO_TYPES_MAPPING,
    REPO_TYPES_URL_PREFIXES,
    SPACES_SDK_TYPES,
)


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


REMOTE_FILEPATH_REGEX = re.compile(r"^\w[\w\/\-]*(\.\w+)?$")
# ^^ No trailing slash, no backslash, no spaces, no relative parts ("." or "..")
#    Only word characters and an optional extension


def repo_type_and_id_from_hf_id(hf_id: str):
    """
    Returns the repo type and ID from a huggingface.co URL linking to a repository

    Args:
        hf_id (``str``):
            An URL or ID of a repository on the HF hub. Accepted values are:
            - https://huggingface.co/<repo_type>/<namespace>/<repo_id>
            - https://huggingface.co/<namespace>/<repo_id>
            - <repo_type>/<namespace>/<repo_id>
            - <namespace>/<repo_id>
            - <repo_id>
    """
    is_hf_url = "huggingface.co" in hf_id and "@" not in hf_id
    url_segments = hf_id.split("/")
    is_hf_id = len(url_segments) <= 3

    if is_hf_url:
        namespace, repo_id = url_segments[-2:]
        if namespace == "huggingface.co":
            namespace = None
        if len(url_segments) > 2 and "huggingface.co" not in url_segments[-3]:
            repo_type = url_segments[-3]
        else:
            repo_type = None
    elif is_hf_id:
        if len(url_segments) == 3:
            # Passed <repo_type>/<user>/<model_id> or <repo_type>/<org>/<model_id>
            repo_type, namespace, repo_id = url_segments[-3:]
        elif len(url_segments) == 2:
            # Passed <user>/<model_id> or <org>/<model_id>
            namespace, repo_id = hf_id.split("/")[-2:]
            repo_type = None
        else:
            # Passed <model_id>
            repo_id = url_segments[0]
            namespace, repo_type = None, None
    else:
        raise ValueError(
            f"Unable to retrieve user and repo ID from the passed HF ID: {hf_id}"
        )

    repo_type = (
        repo_type if repo_type in REPO_TYPES else REPO_TYPES_MAPPING.get(repo_type)
    )

    return repo_type, namespace, repo_id


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


class DatasetFile:
    """
    Data structure that represents a public file inside a dataset, accessible from huggingface.co
    """

    def __init__(self, rfilename: str, **kwargs):
        self.rfilename = rfilename  # filename relative to the dataset root
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


class DatasetInfo:
    """
    Info about a public dataset accessible from huggingface.co
    """

    def __init__(
        self,
        id: Optional[str] = None,  # id of dataset
        lastModified: Optional[str] = None,  # date of last commit to repo
        tags: List[str] = [],  # tags of the dataset
        siblings: Optional[
            List[Dict]
        ] = None,  # list of files that constitute the dataset
        private: Optional[bool] = None,  # community datasets only
        author: Optional[str] = None,  # community datasets only
        description: Optional[str] = None,
        citation: Optional[str] = None,
        card_data: Optional[dict] = None,
        **kwargs,
    ):
        self.id = id
        self.lastModified = lastModified
        self.tags = tags
        self.private = private
        self.author = author
        self.description = description
        self.citation = citation
        self.card_data = card_data
        self.siblings = (
            [DatasetFile(**x) for x in siblings] if siblings is not None else None
        )
        # Legacy stuff, "key" is always returned with an empty string
        # because of old versions of the datasets lib that need this field
        kwargs.pop("key", None)
        # Store all the other fields returned by the API
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        s = f"{self.__class__.__name__}:" + " {"
        for key, val in self.__dict__.items():
            s += f"\n\t{key}: {val}"
        return s + "\n}"

    def __str__(self):
        r = f"Dataset Name: {self.id}, Tags: {self.tags}"
        return r


class MetricInfo:
    """
    Info about a public metric accessible from huggingface.co
    """

    def __init__(
        self,
        id: Optional[str] = None,  # id of metric
        description: Optional[str] = None,
        citation: Optional[str] = None,
        **kwargs,
    ):
        self.id = id
        self.description = description
        self.citation = citation
        # Legacy stuff, "key" is always returned with an empty string
        # because of old versions of the datasets lib that need this field
        kwargs.pop("key", None)
        # Store all the other fields returned by the API
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        s = f"{self.__class__.__name__}:" + " {"
        for key, val in self.__dict__.items():
            s += f"\n\t{key}: {val}"
        return s + "\n}"

    def __str__(self):
        r = f"Metric Name: {self.id}"
        return r


def write_to_credential_store(username: str, password: str):
    with subprocess.Popen(
        "git credential-store store".split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as process:
        input_username = f"username={username.lower()}"
        input_password = f"password={password}"

        process.stdin.write(
            f"url={ENDPOINT}\n{input_username}\n{input_password}\n\n".encode("utf-8")
        )
        process.stdin.flush()


def read_from_credential_store(
    username=None,
) -> Tuple[Union[str, None], Union[str, None]]:
    """
    Reads the credential store relative to huggingface.co. If no `username` is specified, will read the first
    entry for huggingface.co, otherwise will read the entry corresponding to the username specified.

    The username returned will be all lowercase.
    """
    with subprocess.Popen(
        "git credential-store get".split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as process:
        standard_input = f"url={ENDPOINT}\n"

        if username is not None:
            standard_input += f"username={username.lower()}\n"

        standard_input += "\n"

        process.stdin.write(standard_input.encode("utf-8"))
        process.stdin.flush()
        output = process.stdout.read()
        output = output.decode("utf-8")

    if len(output) == 0:
        return None, None

    username, password = [line for line in output.split("\n") if len(line) != 0]
    return username.split("=")[1], password.split("=")[1]


def erase_from_credential_store(username=None):
    """
    Erases the credential store relative to huggingface.co. If no `username` is specified, will erase the first
    entry for huggingface.co, otherwise will erase the entry corresponding to the username specified.
    """
    with subprocess.Popen(
        "git credential-store erase".split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as process:
        standard_input = f"url={ENDPOINT}\n"

        if username is not None:
            standard_input += f"username={username.lower()}\n"

        standard_input += "\n"

        process.stdin.write(standard_input.encode("utf-8"))
        print(standard_input)
        process.stdin.flush()


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

        write_to_credential_store(username, password)
        return d["token"]

    def whoami(self, token: Optional[str] = None) -> Dict:
        """
        Call HF API to know "whoami".

        Args:
            token (``str``, `optional`):
                Hugging Face token. Will default to the locally saved token if not provided.
        """
        if token is None:
            token = HfFolder.get_token()
        if token is None:
            raise ValueError(
                "You need to pass a valid `token` or login by using `huggingface-cli login`"
            )

        path = "{}/api/whoami-v2".format(self.endpoint)
        r = requests.get(path, headers={"authorization": "Bearer {}".format(token)})
        try:
            r.raise_for_status()
        except HTTPError as e:
            raise HTTPError(
                "Invalid user token. If you didn't pass a user token, make sure you are properly logged in by "
                "executing `huggingface-cli login`, and if you did pass a user token, double-check it's correct."
            ) from e
        return r.json()

    def logout(self, token: Optional[str] = None) -> None:
        """
        Call HF API to log out.

        Args:
            token (``str``, `optional`):
                Hugging Face token. Will default to the locally saved token if not provided.
        """
        if token is None:
            token = HfFolder.get_token()
        if token is None:
            raise ValueError(
                "You need to pass a valid `token` or login by using `huggingface-cli login`"
            )

        username = self.whoami(token)["name"]
        erase_from_credential_store(username)

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
        fetch_config: Optional[bool] = None,
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
            fetch_config (:obj:`bool`, `optional`):
                Whether to fetch the model configs as well. This is not included in `full` due to its size.

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
        if fetch_config is not None:
            params.update({"config": fetch_config})
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

    def list_datasets(
        self,
        filter: Union[str, Iterable[str], None] = None,
        sort: Union[Literal["lastModified"], str, None] = None,
        direction: Optional[Literal[-1]] = None,
        limit: Optional[int] = None,
        full: Optional[bool] = None,
    ) -> List[DatasetInfo]:
        """
        Get the public list of all the datasets on huggingface.co

        Args:
            filter (:obj:`str` or :class:`Iterable`, `optional`):
                A string which can be used to identify datasets on the hub by their tags.
                Example usage:

                    >>> from huggingface_hub import HfApi
                    >>> api = HfApi()

                    >>> # List all datasets
                    >>> api.list_datasets()

                    >>> # List only the text classification datasets
                    >>> api.list_datasets(filter="task_categories:text-classification")

                    >>> # List only the datasets in russian for language modeling
                    >>> api.list_datasets(filter=("languages:ru", "task_ids:language-modeling"))
            sort (:obj:`Literal["lastModified"]` or :obj:`str`, `optional`):
                The key with which to sort the resulting datasets. Possible values are the properties of the `DatasetInfo`
                class.
            direction (:obj:`Literal[-1]` or :obj:`int`, `optional`):
                Direction in which to sort. The value `-1` sorts by descending order while all other values
                sort by ascending order.
            limit (:obj:`int`, `optional`):
                The limit on the number of datasets fetched. Leaving this option to `None` fetches all datasets.
            full (:obj:`bool`, `optional`):
                Whether to fetch all dataset data, including the `lastModified` and the `card_data`.

        """
        path = "{}/api/datasets".format(self.endpoint)
        params = {}
        if filter is not None:
            params.update({"filter": filter})
        if sort is not None:
            params.update({"sort": sort})
        if direction is not None:
            params.update({"direction": direction})
        if limit is not None:
            params.update({"limit": limit})
        if full is not None:
            if full:
                params.update({"full": True})
        r = requests.get(path, params=params)
        r.raise_for_status()
        d = r.json()
        return [DatasetInfo(**x) for x in d]

    def list_metrics(self) -> List[MetricInfo]:
        """
        Get the public list of all the metrics on huggingface.co
        """
        path = "{}/api/metrics".format(self.endpoint)
        params = {}
        r = requests.get(path, params=params)
        r.raise_for_status()
        d = r.json()
        return [MetricInfo(**x) for x in d]

    def model_info(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ModelInfo:
        """
        Get info on one specific model on huggingface.co

        Model can be private if you pass an acceptable token or are logged in.
        """
        if token is None:
            token = HfFolder.get_token()

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
        r = requests.get(path, headers=headers, timeout=timeout)
        r.raise_for_status()
        d = r.json()
        return ModelInfo(**d)

    def list_repo_files(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> List[str]:
        """
        Get the list of files in a given repo.
        """
        if repo_type is None:
            info = self.model_info(
                repo_id, revision=revision, token=token, timeout=timeout
            )
        elif repo_type == "dataset":
            info = self.dataset_info(
                repo_id, revision=revision, token=token, timeout=timeout
            )
        else:
            raise ValueError("Spaces are not available yet.")

        return [f.rfilename for f in info.siblings]

    def list_repos_objs(
        self, token: Optional[str] = None, organization: Optional[str] = None
    ) -> List[RepoObj]:
        """
        HuggingFace git-based system, used for models, datasets, and spaces.

        Call HF API to list all stored files for user (or one of their organizations).
        """
        if token is None:
            token = HfFolder.get_token()
        if token is None:
            raise ValueError(
                "You need to pass a valid `token` or login by using `huggingface-cli login`"
            )

        path = "{}/api/repos/ls".format(self.endpoint)
        params = {"organization": organization} if organization is not None else None
        r = requests.get(
            path, params=params, headers={"authorization": "Bearer {}".format(token)}
        )
        r.raise_for_status()
        d = r.json()
        return [RepoObj(**x) for x in d]

    def dataset_info(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> DatasetInfo:
        """
        Get info on one specific dataset on huggingface.co

        Dataset can be private if you pass an acceptable token.
        """
        path = (
            "{}/api/datasets/{repo_id}".format(self.endpoint, repo_id=repo_id)
            if revision is None
            else "{}/api/datasets/{repo_id}/revision/{revision}".format(
                self.endpoint, repo_id=repo_id, revision=revision
            )
        )
        headers = (
            {"authorization": "Bearer {}".format(token)} if token is not None else None
        )
        params = {"full": "true"}
        r = requests.get(path, headers=headers, params=params, timeout=timeout)
        r.raise_for_status()
        d = r.json()
        return DatasetInfo(**d)

    def _is_valid_token(self, token: str):
        """
        Determines whether `token` is a valid token or not.
        """
        try:
            self.whoami(token=token)
            return True
        except HTTPError:
            return False

    def create_repo(
        self,
        name: str,
        token: Optional[str] = None,
        organization: Optional[str] = None,
        private: Optional[bool] = None,
        repo_type: Optional[str] = None,
        exist_ok=False,
        lfsmultipartthresh: Optional[int] = None,
        space_sdk: Optional[str] = None,
    ) -> str:
        """
        HuggingFace git-based system, used for models, datasets, and spaces.

        Call HF API to create a whole repo.

        Params:
            private: Whether the model repo should be private (requires a paid huggingface.co account)

            repo_type: Set to "dataset" or "space" if creating a dataset or space, default is model

            exist_ok: Do not raise an error if repo already exists

            lfsmultipartthresh: Optional: internal param for testing purposes.

            space_sdk: Choice of SDK to use if repo_type is "space". Can be "streamlit", "gradio", or "static".

        Returns:
            URL to the newly created repo.
        """
        path = "{}/api/repos/create".format(self.endpoint)
        if token is None:
            token = HfFolder.get_token()
            if token is None:
                raise EnvironmentError(
                    "You need to provide a `token` or be logged in to Hugging Face with "
                    "`huggingface-cli login`."
                )
        elif not self._is_valid_token(token):
            if self._is_valid_token(name):
                warnings.warn(
                    "`create_repo` now takes `token` as an optional positional argument. "
                    "Be sure to adapt your code!",
                    FutureWarning,
                )
                token, name = name, token
            else:
                raise ValueError("Invalid token passed!")

        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        json = {"name": name, "organization": organization, "private": private}
        if repo_type is not None:
            json["type"] = repo_type
            if repo_type == "space":
                if space_sdk is None:
                    raise ValueError(
                        "No space_sdk provided. `create_repo` expects space_sdk to be one of "
                        f"{SPACES_SDK_TYPES} when repo_type is 'space'`"
                    )
                if space_sdk not in SPACES_SDK_TYPES:
                    raise ValueError(
                        f"Invalid space_sdk. Please choose one of {SPACES_SDK_TYPES}."
                    )
                json["sdk"] = space_sdk
        if space_sdk is not None and repo_type != "space":
            warnings.warn(
                "Ignoring provided space_sdk because repo_type is not 'space'."
            )

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
        name: str,
        token: Optional[str] = None,
        organization: Optional[str] = None,
        repo_type: Optional[str] = None,
    ):
        """
        HuggingFace git-based system, used for models, datasets, and spaces.

        Call HF API to delete a whole repo.

        CAUTION(this is irreversible).
        """
        path = "{}/api/repos/delete".format(self.endpoint)
        if token is None:
            token = HfFolder.get_token()
            if token is None:
                raise EnvironmentError(
                    "You need to provide a `token` or be logged in to Hugging Face with "
                    "`huggingface-cli login`."
                )
        elif not self._is_valid_token(token):
            if self._is_valid_token(name):
                warnings.warn(
                    "`delete_repo` now takes `token` as an optional positional argument. "
                    "Be sure to adapt your code!",
                    FutureWarning,
                )
                token, name = name, token
            else:
                raise ValueError("Invalid token passed!")

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
        name: str,
        private: bool,
        token: Optional[str] = None,
        organization: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Update the visibility setting of a repository.
        """
        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        if token is None:
            token = HfFolder.get_token()
            if token is None:
                raise EnvironmentError(
                    "You need to provide a `token` or be logged in to Hugging Face with "
                    "`huggingface-cli login`."
                )
        elif not self._is_valid_token(token):
            if self._is_valid_token(name):
                warnings.warn(
                    "`update_repo_visibility` now takes `token` as an optional positional argument. "
                    "Be sure to adapt your code!",
                    FutureWarning,
                )
                token, name, private = name, private, token
            else:
                raise ValueError("Invalid token passed!")

        if organization is None:
            namespace = self.whoami(token)["name"]
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
        path_or_fileobj: Union[str, bytes, IO],
        path_in_repo: str,
        repo_id: str,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        identical_ok: bool = True,
    ) -> str:
        """
        Upload a local file (up to 5GB) to the given repo. The upload is done through a HTTP post request, and
        doesn't require git or git-lfs to be installed.

        Params:
            path_or_fileobj (``str``, ``bytes``, or ``IO``):
                Path to a file on the local machine or binary data stream / fileobj / buffer.

            path_in_repo (``str``):
                Relative filepath in the repo, for example: :obj:`"checkpoints/1fec34a/weights.bin"`

            repo_id (``str``):
                The repository to which the file will be uploaded, for example: :obj:`"username/custom_transformers"`

            token (``str``):
                Authentication token, obtained with :function:`HfApi.login` method. Will default to the stored token.

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

        if token is None:
            token = HfFolder.get_token()
            if token is None:
                raise EnvironmentError(
                    "You need to provide a `token` or be logged in to Hugging Face with "
                    "`huggingface-cli login`."
                )
        elif not self._is_valid_token(token):
            if self._is_valid_token(path_or_fileobj):
                warnings.warn(
                    "`upload_file` now takes `token` as an optional positional argument. "
                    "Be sure to adapt your code!",
                    FutureWarning,
                )
                token, path_or_fileobj, path_in_repo, repo_id = (
                    path_or_fileobj,
                    path_in_repo,
                    repo_id,
                    token,
                )
            else:
                raise ValueError("Invalid token passed!")

        # Validate path_or_fileobj
        if isinstance(path_or_fileobj, str):
            path_or_fileobj = os.path.normpath(os.path.expanduser(path_or_fileobj))
            if not os.path.isfile(path_or_fileobj):
                raise ValueError(
                    "Provided path: '{}' is not a file".format(path_or_fileobj)
                )
        elif not isinstance(path_or_fileobj, (RawIOBase, BufferedIOBase, bytes)):
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
            if identical_ok and err.response.status_code == 409:
                from .file_download import hf_hub_url

                return hf_hub_url(
                    repo_id, path_in_repo, revision=revision, repo_type=repo_type
                )
            else:
                raise err

        d = r.json()
        return d["url"]

    def delete_file(
        self,
        path_in_repo: str,
        repo_id: str,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
    ):
        """
        Deletes a file in the given repo.

        Params:
            path_in_repo (``str``):
                Relative filepath in the repo, for example: :obj:`"checkpoints/1fec34a/weights.bin"`

            repo_id (``str``):
                The repository from which the file will be deleted, for example: :obj:`"username/custom_transformers"`

            token (``str``):
                Authentication token, obtained with :function:`HfApi.login` method. Will default to the stored token.

            repo_type (``str``, Optional):
                Set to :obj:`"dataset"` or :obj:`"space"` if the file is in a dataset or space repository, :obj:`None` if in a model. Default is :obj:`None`.

            revision (``str``, Optional):
                The git revision to commit from. Defaults to the :obj:`"main"` branch.

        Raises:
            :class:`ValueError`: if some parameter value is invalid

            :class:`requests.HTTPError`: if the HuggingFace API returned an error

        """
        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type, must be one of {}".format(REPO_TYPES))

        if token is None:
            token = HfFolder.get_token()
            if token is None:
                raise EnvironmentError(
                    "You need to provide a `token` or be logged in to Hugging Face with "
                    "`huggingface-cli login`."
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

        path = "{}/api/{repo_id}/delete/{revision}/{path_in_repo}".format(
            self.endpoint,
            repo_id=repo_id,
            revision=revision,
            path_in_repo=path_in_repo,
        )

        headers = {"authorization": "Bearer {}".format(token)}
        r = requests.delete(path, headers=headers)

        r.raise_for_status()

    def get_full_repo_name(
        self,
        model_id: str,
        organization: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Returns the repository name for a given model ID and optional organization.

        Args:
            model_id (``str``):
                The name of the model.
            organization (``str``, `optional`):
                If passed, the repository name will be in the organization namespace instead of the
                user namespace.
            token (``str``, `optional`):
                The Hugging Face authentication token

        Returns:
            ``str``: The repository name in the user's namespace ({username}/{model_id}) if no
            organization is passed, and under the organization namespace ({organization}/{model_id})
            otherwise.
        """
        if organization is None:
            username = self.whoami(token=token)["name"]
            return f"{username}/{model_id}"
        else:
            return f"{organization}/{model_id}"


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


api = HfApi()

login = api.login
logout = api.logout
whoami = api.whoami

list_models = api.list_models
model_info = api.model_info
list_repo_files = api.list_repo_files
list_repos_objs = api.list_repos_objs

list_datasets = api.list_datasets
dataset_info = api.dataset_info

list_metrics = api.list_metrics

create_repo = api.create_repo
delete_repo = api.delete_repo
update_repo_visibility = api.update_repo_visibility
upload_file = api.upload_file
delete_file = api.delete_file
get_full_repo_name = api.get_full_repo_name
