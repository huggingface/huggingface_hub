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
from http import HTTPStatus
from os.path import expanduser
from typing import Dict, List, Literal, Optional, Tuple, Union

import requests

from .constants import REPO_TYPES


ENDPOINT = "https://huggingface.co/api/"


class RepoObj:
    """
    HuggingFace git-based system, data structure that represents a file belonging to the current user.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class ObjectInfo:
    """
    Info about a public dataset or Metric accessible from our S3.
    """

    def __init__(
        self,
        id: str,
        key: str,
        lastModified: Optional[str] = None,
        description: Optional[str] = None,
        citation: Optional[str] = None,
        size: Optional[int] = None,
        etag: Optional[str] = None,
        siblings: List[Dict] = None,
        author: str = None,
        **kwargs,
    ):
        self.id = id  # id of dataset
        self.key = key  # S3 object key of config.json
        self.lastModified = lastModified
        self.description = description
        self.citation = citation
        self.size = size
        self.etag = etag
        self.siblings = siblings  # list of files that constitute the dataset
        self.author = author
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        single_line_description = self.description.replace("\n", "") if self.description is not None else ""
        return f"datasets.ObjectInfo(\n\tid='{self.id}',\n\tdescription='{single_line_description}',\n\tfiles={self.siblings}\n)"


class ModelSibling:
    """
    Data structure that represents a public file inside a model, accessible from huggingface.co
    """

    def __init__(self, rfilename: str, **kwargs):
        self.rfilename = rfilename  # filename relative to the model root
        for k, v in kwargs.items():
            setattr(self, k, v)


class ModelInfo:
    """
    Info about a public model accessible from huggingface.co
    """

    def __init__(
        self,
        modelId: Optional[str] = None,  # id of model
        tags: List[str] = [],
        pipeline_tag: Optional[str] = None,
        siblings: Optional[List[Dict]] = None,  # list of files that constitute the model
        **kwargs,
    ):
        self.modelId = modelId
        self.tags = tags
        self.pipeline_tag = pipeline_tag
        self.siblings = [ModelSibling(**x) for x in siblings] if siblings is not None else None
        for k, v in kwargs.items():
            setattr(self, k, v)


class HfApi:
    def __init__(self, endpoint=None):
        self.endpoint: str = endpoint if endpoint is not None else ENDPOINT
        self.token: Optional[str] = None

    @property
    def auth_headers(self):
        if self.token is None:
            raise ValueError("you need to login before using this -> `.login(username, password)`")
        return dict(authorization=f"Bearer {self.token}")

    def _api(
        self, path: str, method: Literal["get", "post"] = "get", ignore_status: List[int] = None, **kwargs
    ) -> Union[Dict, List]:
        response = requests.request(method, f"{self.endpoint}/{path}", **kwargs)
        if ignore_status is not None and response.status_code in ignore_status:
            return response.json()
        response.raise_for_status()
        return response.json()

    def login(self, username: str, password: str) -> str:
        """
        Call HF API to sign in a user and get a token if credentials are valid.

        Outputs: token if credentials are valid

        Throws: requests.exceptions.HTTPError if credentials are invalid
        """
        response = self._api("login", "post", json=dict(username=username, password=password))
        self.token = response["token"]
        return response["token"]

    def whoami(self) -> Tuple[str, List[str]]:
        """
        Call HF API to know "whoami"
        """
        response = self._api("whoami", headers=self.auth_headers)
        return response["user"], response["orgs"]

    def logout(self) -> None:
        """
        Call HF API to log out.
        """
        self._api("logout", "post", headers=self.auth_headers)

    def model_list(self) -> List[ModelInfo]:
        """
        Get the public list of all the models on huggingface.co
        """
        response = self._api("models")
        return [ModelInfo(**x) for x in response]

    def dataset_list(self, with_community_datasets=True, id_only=False) -> Union[List[ObjectInfo], List[str]]:
        """
        Get the public list of all the datasets on huggingface, including the community datasets
        """
        response = self._api("datasets")
        datasets = [ObjectInfo(**x) for x in response]
        if not with_community_datasets:
            datasets = [d for d in datasets if "/" not in d.id]
        if id_only:
            datasets = [d.id for d in datasets]
        return datasets

    def metric_list(self, with_community_metrics=True, id_only=False) -> Union[List[ObjectInfo], List[str]]:
        """
        Get the public list of all the metrics on huggingface, including the community metrics
        """
        response = self._api("metrics")
        metrics = [ObjectInfo(**x) for x in response]
        if not with_community_metrics:
            metrics = [m for m in metrics if "/" not in m.id]
        if id_only:
            metrics = [m.id for m in metrics]
        return metrics

    def list_repos_objs(self, organization: Optional[str] = None) -> List[RepoObj]:
        """
        HuggingFace git-based system, used for models.

        Call HF API to list all stored files for user (or one of their organizations).
        """
        response = self._api(
            "repos/ls",
            headers=self.auth_headers,
            params=dict(organization=organization) if organization is not None else None,
        )
        return [RepoObj(**x) for x in response]

    def create_repo(
        self,
        name: str,
        organization: Optional[str] = None,
        private: Optional[bool] = None,
        repo_type: Optional[str] = None,
        exist_ok=False,
        lfsmultipartthresh: Optional[int] = None,
    ) -> str:
        """
        HuggingFace git-based system, used for models.

        Call HF API to create a whole repo.

        Params:
            private: Whether the model repo should be private (requires a paid huggingface.co account)

            repo_type: Set to "dataset" if creating a dataset, default is model

            exist_ok: Do not raise an error if repo already exists

            lfsmultipartthresh: Optional: internal param for testing purposes.

        Returns:
            URL to the newly created repo.
        """
        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        json = {"name": name, "organization": organization, "private": private}
        if repo_type is not None:
            json["type"] = repo_type
        if lfsmultipartthresh is not None:
            json["lfsmultipartthresh"] = lfsmultipartthresh
        response = self._api(
            "repos/create",
            ignore_status=[HTTPStatus.CONFLICT] if exist_ok else 0,
            headers=self.auth_headers,
            json=json,
        )
        return response["url"]

    def delete_repo(
        self,
        name: str,
        organization: Optional[str] = None,
        repo_type: Optional[str] = None,
    ):
        """
        HuggingFace git-based system, used for models.

        Call HF API to delete a whole repo.

        CAUTION(this is irreversible).
        """
        path = "{}/api/repos/delete".format(self.endpoint)

        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        json = {"name": name, "organization": organization}
        if repo_type is not None:
            json["type"] = repo_type

        self._api("repos/delete", headers=self.auth_headers, json=json)


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
