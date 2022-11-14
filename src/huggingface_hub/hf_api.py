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
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Iterator, List, Optional, Tuple, Union
from urllib.parse import quote

import requests
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from requests.exceptions import HTTPError

from ._commit_api import (
    CommitOperation,
    CommitOperationAdd,
    CommitOperationDelete,
    fetch_upload_modes,
    prepare_commit_payload,
    upload_lfs_files,
    warn_on_overwriting_operations,
)
from .community import (
    Discussion,
    DiscussionComment,
    DiscussionStatusChange,
    DiscussionTitleChange,
    DiscussionWithDetails,
    deserialize_event,
)
from .constants import (
    DEFAULT_REVISION,
    ENDPOINT,
    REGEX_COMMIT_OID,
    REPO_TYPE_MODEL,
    REPO_TYPES,
    REPO_TYPES_MAPPING,
    REPO_TYPES_URL_PREFIXES,
    SPACES_SDK_TYPES,
)
from .utils import (  # noqa: F401 # imported for backward compatibility
    HfFolder,
    HfHubHTTPError,
    build_hf_headers,
    erase_from_credential_store,
    filter_repo_objects,
    hf_raise_for_status,
    logging,
    parse_datetime,
    read_from_credential_store,
    validate_hf_hub_args,
    write_to_credential_store,
)
from .utils._deprecation import (
    _deprecate_arguments,
    _deprecate_list_output,
    _deprecate_method,
    _deprecate_positional_args,
)
from .utils._pagination import paginate
from .utils._typing import Literal, TypedDict
from .utils.endpoint_helpers import (
    AttributeDictionary,
    DatasetFilter,
    DatasetTags,
    ModelFilter,
    ModelTags,
    _filter_emissions,
)


USERNAME_PLACEHOLDER = "hf_user"
_REGEX_DISCUSSION_URL = re.compile(r".*/discussions/(\d+)$")

logger = logging.get_logger(__name__)


def repo_type_and_id_from_hf_id(
    hf_id: str, hub_url: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Returns the repo type and ID from a huggingface.co URL linking to a
    repository

    Args:
        hf_id (`str`):
            An URL or ID of a repository on the HF hub. Accepted values are:

            - https://huggingface.co/<repo_type>/<namespace>/<repo_id>
            - https://huggingface.co/<namespace>/<repo_id>
            - <repo_type>/<namespace>/<repo_id>
            - <namespace>/<repo_id>
            - <repo_id>
        hub_url (`str`, *optional*):
            The URL of the HuggingFace Hub, defaults to https://huggingface.co

    Returns:
        A tuple with three items: repo_type (`str` or `None`), namespace (`str` or
        `None`) and repo_id (`str`).
    """
    hub_url = re.sub(r"https?://", "", hub_url if hub_url is not None else ENDPOINT)
    is_hf_url = hub_url in hf_id and "@" not in hf_id
    url_segments = hf_id.split("/")
    is_hf_id = len(url_segments) <= 3

    namespace: Optional[str]
    if is_hf_url:
        namespace, repo_id = url_segments[-2:]
        if namespace == hub_url:
            namespace = None
        if len(url_segments) > 2 and hub_url not in url_segments[-3]:
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

    if repo_type not in REPO_TYPES:
        assert repo_type is not None, "repo_type `None` do not have mapping"
        repo_type = REPO_TYPES_MAPPING.get(repo_type)

    return repo_type, namespace, repo_id


class BlobLfsInfo(TypedDict, total=False):
    size: int
    sha256: str


@dataclass
class CommitInfo:
    """Data structure containing information about a newly created commit.

    Returned by [`create_commit`].

    Args:
        commit_url (`str`):
            Url where to find the commit.

        commit_message (`str`):
            The summary (first line) of the commit that has been created.

        commit_description (`str`):
            Description of the commit that has been created. Can be empty.

        oid (`str`):
            Commit hash id. Example: `"91c54ad1727ee830252e457677f467be0bfd8a57"`.

        pr_url (`str`, *optional*):
            Url to the PR that has been created, if any. Populated when `create_pr=True`
            is passed.

        pr_revision (`str`, *optional*):
            Revision of the PR that has been created, if any. Populated when
            `create_pr=True` is passed. Example: `"refs/pr/1"`.

        pr_num (`int`, *optional*):
            Number of the PR discussion that has been created, if any. Populated when
            `create_pr=True` is passed. Can be passed as `discussion_num` in
            [`get_discussion_details`]. Example: `1`.
    """

    commit_url: str
    commit_message: str
    commit_description: str
    oid: str
    pr_url: Optional[str] = None

    # Computed from `pr_url` in `__post_init__`
    pr_revision: Optional[str] = field(init=False)
    pr_num: Optional[str] = field(init=False)

    def __post_init__(self):
        """Populate pr-related fields after initialization.

        See https://docs.python.org/3.10/library/dataclasses.html#post-init-processing.
        """
        if self.pr_url is not None:
            self.pr_revision = _parse_revision_from_pr_url(self.pr_url)
            self.pr_num = int(self.pr_revision.split("/")[-1])
        else:
            self.pr_revision = None
            self.pr_num = None


class RepoFile:
    """
    Data structure that represents a public file inside a repo, accessible from
    huggingface.co

    Args:
        rfilename (str):
            file name, relative to the repo root. This is the only attribute
            that's guaranteed to be here, but under certain conditions there can
            certain other stuff.
        size (`int`, *optional*):
            The file's size, in bytes. This attribute is present when `files_metadata` argument
            of [`repo_info`] is set to `True`. It's `None` otherwise.
        blob_id (`str`, *optional*):
            The file's git OID. This attribute is present when `files_metadata` argument
            of [`repo_info`] is set to `True`. It's `None` otherwise.
        lfs (`BlobLfsInfo`, *optional*):
            The file's LFS metadata. This attribute is present when`files_metadata` argument
            of [`repo_info`] is set to `True` and the file is stored with Git LFS. It's `None` otherwise.
    """

    def __init__(
        self,
        rfilename: str,
        size: Optional[int] = None,
        blobId: Optional[str] = None,
        lfs: Optional[BlobLfsInfo] = None,
        **kwargs,
    ):
        self.rfilename = rfilename  # filename relative to the repo root

        # Optional file metadata
        self.size = size
        self.blob_id = blobId
        self.lfs = lfs

        # Hack to ensure backward compatibility with future versions of the API.
        # See discussion in https://github.com/huggingface/huggingface_hub/pull/951#discussion_r926460408
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        items = (f"{k}='{v}'" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({', '.join(items)})"


class ModelInfo:
    """
    Info about a model accessible from huggingface.co

    Attributes:
        modelId (`str`, *optional*):
            ID of model repository.
        sha (`str`, *optional*):
            repo sha at this particular revision
        lastModified (`str`, *optional*):
            date of last commit to repo
        tags (`List[str]`, *optional*):
            List of tags.
        pipeline_tag (`str`, *optional*):
            Pipeline tag to identify the correct widget.
        siblings (`List[RepoFile]`, *optional*):
            list of ([`huggingface_hub.hf_api.RepoFile`]) objects that constitute the model.
        private (`bool`, *optional*, defaults to `False`):
            is the repo private
        author (`str`, *optional*):
            repo author
        config (`Dict`, *optional*):
            Model configuration information
        securityStatus (`Dict`, *optional*):
            Security status of the model.
            Example: `{"containsInfected": False}`
        kwargs (`Dict`, *optional*):
            Kwargs that will be become attributes of the class.
    """

    def __init__(
        self,
        *,
        modelId: Optional[str] = None,
        sha: Optional[str] = None,
        lastModified: Optional[str] = None,
        tags: Optional[List[str]] = None,
        pipeline_tag: Optional[str] = None,
        siblings: Optional[List[Dict]] = None,
        private: bool = False,
        author: Optional[str] = None,
        config: Optional[Dict] = None,
        securityStatus: Optional[Dict] = None,
        **kwargs,
    ):
        self.modelId = modelId
        self.sha = sha
        self.lastModified = lastModified
        self.tags = tags
        self.pipeline_tag = pipeline_tag
        self.siblings = (
            [RepoFile(**x) for x in siblings] if siblings is not None else []
        )
        self.private = private
        self.author = author
        self.config = config
        self.securityStatus = securityStatus
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
    Info about a dataset accessible from huggingface.co

    Attributes:
        id (`str`, *optional*):
            ID of dataset repository.
        sha (`str`, *optional*):
            repo sha at this particular revision
        lastModified (`str`, *optional*):
            date of last commit to repo
        tags (`Listr[str]`, *optional*):
            List of tags.
        siblings (`List[RepoFile]`, *optional*):
            list of [`huggingface_hub.hf_api.RepoFile`] objects that constitute the dataset.
        private (`bool`, *optional*, defaults to `False`):
            is the repo private
        author (`str`, *optional*):
            repo author
        description (`str`, *optional*):
            Description of the dataset
        citation (`str`, *optional*):
            Dataset citation
        cardData (`Dict`, *optional*):
            Metadata of the model card as a dictionary.
        kwargs (`Dict`, *optional*):
            Kwargs that will be become attributes of the class.
    """

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        sha: Optional[str] = None,
        lastModified: Optional[str] = None,
        tags: Optional[List[str]] = None,
        siblings: Optional[List[Dict]] = None,
        private: bool = False,
        author: Optional[str] = None,
        description: Optional[str] = None,
        citation: Optional[str] = None,
        cardData: Optional[dict] = None,
        **kwargs,
    ):
        self.id = id
        self.sha = sha
        self.lastModified = lastModified
        self.tags = tags
        self.private = private
        self.author = author
        self.description = description
        self.citation = citation
        self.cardData = cardData
        self.siblings = (
            [RepoFile(**x) for x in siblings] if siblings is not None else []
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


class SpaceInfo:
    """
    Info about a Space accessible from huggingface.co

    This is a "dataclass" like container that just sets on itself any attribute
    passed by the server.

    Attributes:
        id (`str`, *optional*):
            id of space
        sha (`str`, *optional*):
            repo sha at this particular revision
        lastModified (`str`, *optional*):
            date of last commit to repo
        siblings (`List[RepoFile]`, *optional*):
            list of [`huggingface_hub.hf_api.RepoFIle`] objects that constitute the Space
        private (`bool`, *optional*, defaults to `False`):
            is the repo private
        author (`str`, *optional*):
            repo author
        kwargs (`Dict`, *optional*):
            Kwargs that will be become attributes of the class.
    """

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        sha: Optional[str] = None,
        lastModified: Optional[str] = None,
        siblings: Optional[List[Dict]] = None,
        private: bool = False,
        author: Optional[str] = None,
        **kwargs,
    ):
        self.id = id
        self.sha = sha
        self.lastModified = lastModified
        self.siblings = (
            [RepoFile(**x) for x in siblings] if siblings is not None else []
        )
        self.private = private
        self.author = author
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        s = f"{self.__class__.__name__}:" + " {"
        for key, val in self.__dict__.items():
            s += f"\n\t{key}: {val}"
        return s + "\n}"


class MetricInfo:
    """
    Info about a public metric accessible from huggingface.co
    """

    def __init__(
        self,
        *,
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


class ModelSearchArguments(AttributeDictionary):
    """
    A nested namespace object holding all possible values for properties of
    models currently hosted in the Hub with tab-completion. If a value starts
    with a number, it will only exist in the dictionary

    Example:

    ```python
    >>> args = ModelSearchArguments()
    >>> args.author_or_organization.huggingface
    >>> args.language.en
    ```
    """

    def __init__(self, api: Optional["HfApi"] = None):
        self._api = api if api is not None else HfApi()
        tags = self._api.get_model_tags()
        super().__init__(tags)
        self._process_models()

    def _process_models(self):
        def clean(s: str) -> str:
            return s.replace(" ", "").replace("-", "_").replace(".", "_")

        models = self._api.list_models()
        author_dict, model_name_dict = AttributeDictionary(), AttributeDictionary()
        for model in models:
            if "/" in model.modelId:
                author, name = model.modelId.split("/")
                author_dict[author] = clean(author)
            else:
                name = model.modelId
            model_name_dict[name] = clean(name)
        self["model_name"] = model_name_dict
        self["author"] = author_dict


class DatasetSearchArguments(AttributeDictionary):
    """
    A nested namespace object holding all possible values for properties of
    datasets currently hosted in the Hub with tab-completion. If a value starts
    with a number, it will only exist in the dictionary

    Example:

    ```python
    >>> args = DatasetSearchArguments()
    >>> args.author_or_organization.huggingface
    >>> args.language.en
    ```
    """

    def __init__(self, api: Optional["HfApi"] = None):
        self._api = api if api is not None else HfApi()
        tags = self._api.get_dataset_tags()
        super().__init__(tags)
        self._process_models()

    def _process_models(self):
        def clean(s: str):
            return s.replace(" ", "").replace("-", "_").replace(".", "_")

        datasets = self._api.list_datasets()
        author_dict, dataset_name_dict = AttributeDictionary(), AttributeDictionary()
        for dataset in datasets:
            if "/" in dataset.id:
                author, name = dataset.id.split("/")
                author_dict[author] = clean(author)
            else:
                name = dataset.id
            dataset_name_dict[name] = clean(name)
        self["dataset_name"] = dataset_name_dict
        self["author"] = author_dict


class HfApi:
    def __init__(
        self, endpoint: Optional[str] = None, token: Optional[str] = None
    ) -> None:
        self.endpoint = endpoint if endpoint is not None else ENDPOINT
        self.token = token

    def whoami(self, token: Optional[str] = None) -> Dict:
        """
        Call HF API to know "whoami".

        Args:
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if
                not provided.
        """
        r = requests.get(
            f"{self.endpoint}/api/whoami-v2",
            headers=self._build_hf_headers(
                # If `token` is provided and not `None`, it will be used by default.
                # Otherwise, the token must be retrieved from cache or env variable.
                token=(token or self.token or True),
            ),
        )
        try:
            hf_raise_for_status(r)
        except HTTPError as e:
            raise HTTPError(
                "Invalid user token. If you didn't pass a user token, make sure you "
                "are properly logged in by executing `huggingface-cli login`, and "
                "if you did pass a user token, double-check it's correct."
            ) from e
        return r.json()

    def _is_valid_token(self, token: str) -> bool:
        """
        Determines whether `token` is a valid token or not.

        Args:
            token (`str`):
                The token to check for validity.

        Returns:
            `bool`: `True` if valid, `False` otherwise.
        """
        try:
            self.whoami(token=token)
            return True
        except HTTPError:
            return False

    @staticmethod
    @_deprecate_method(
        version="0.14",
        message=(
            "`HfApi.set_access_token` is deprecated as it is very ambiguous. Use"
            " `login` or `set_git_credential` instead."
        ),
    )
    def set_access_token(access_token: str):
        """
        Saves the passed access token so git can correctly authenticate the
        user.

        Args:
            access_token (`str`):
                The access token to save.
        """
        write_to_credential_store(USERNAME_PLACEHOLDER, access_token)

    @staticmethod
    @_deprecate_method(
        version="0.14",
        message=(
            "`HfApi.unset_access_token` is deprecated as it is very ambiguous. Use"
            " `login` or `unset_git_credential` instead."
        ),
    )
    def unset_access_token():
        """
        Resets the user's access token.
        """
        erase_from_credential_store(USERNAME_PLACEHOLDER)

    def get_model_tags(self) -> ModelTags:
        "Gets all valid model tags as a nested namespace object"
        path = f"{self.endpoint}/api/models-tags-by-type"
        r = requests.get(path)
        hf_raise_for_status(r)
        d = r.json()
        return ModelTags(d)

    def get_dataset_tags(self) -> DatasetTags:
        """
        Gets all valid dataset tags as a nested namespace object.
        """
        path = f"{self.endpoint}/api/datasets-tags-by-type"
        r = requests.get(path)
        hf_raise_for_status(r)
        d = r.json()
        return DatasetTags(d)

    @_deprecate_list_output(version="0.14")
    @validate_hf_hub_args
    def list_models(
        self,
        *,
        filter: Union[ModelFilter, str, Iterable[str], None] = None,
        author: Optional[str] = None,
        search: Optional[str] = None,
        emissions_thresholds: Optional[Tuple[float, float]] = None,
        sort: Union[Literal["lastModified"], str, None] = None,
        direction: Optional[Literal[-1]] = None,
        limit: Optional[int] = None,
        full: Optional[bool] = None,
        cardData: bool = False,
        fetch_config: bool = False,
        token: Optional[Union[bool, str]] = None,
    ) -> List[ModelInfo]:
        """
        Get the list of all the models on huggingface.co

        Args:
            filter ([`ModelFilter`] or `str` or `Iterable`, *optional*):
                A string or [`ModelFilter`] which can be used to identify models
                on the Hub.
            author (`str`, *optional*):
                A string which identify the author (user or organization) of the
                returned models
            search (`str`, *optional*):
                A string that will be contained in the returned models Example
                usage:
            emissions_thresholds (`Tuple`, *optional*):
                A tuple of two ints or floats representing a minimum and maximum
                carbon footprint to filter the resulting models with in grams.
            sort (`Literal["lastModified"]` or `str`, *optional*):
                The key with which to sort the resulting models. Possible values
                are the properties of the [`huggingface_hub.hf_api.ModelInfo`] class.
            direction (`Literal[-1]` or `int`, *optional*):
                Direction in which to sort. The value `-1` sorts by descending
                order while all other values sort by ascending order.
            limit (`int`, *optional*):
                The limit on the number of models fetched. Leaving this option
                to `None` fetches all models.
            full (`bool`, *optional*):
                Whether to fetch all model data, including the `lastModified`,
                the `sha`, the files and the `tags`. This is set to `True` by
                default when using a filter.
            cardData (`bool`, *optional*):
                Whether to grab the metadata for the model as well. Can contain
                useful information such as carbon emissions, metrics, and
                datasets trained on.
            fetch_config (`bool`, *optional*):
                Whether to fetch the model configs as well. This is not included
                in `full` due to its size.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            `List[ModelInfo]`: a list of [`huggingface_hub.hf_api.ModelInfo`] objects.
             To anticipate future pagination, please consider the return value to be a
             simple iterator.

        Example usage with the `filter` argument:

        ```python
        >>> from huggingface_hub import HfApi

        >>> api = HfApi()

        >>> # List all models
        >>> api.list_models()

        >>> # Get all valid search arguments
        >>> args = ModelSearchArguments()

        >>> # List only the text classification models
        >>> api.list_models(filter="text-classification")
        >>> # Using the `ModelFilter`
        >>> filt = ModelFilter(task="text-classification")
        >>> # With `ModelSearchArguments`
        >>> filt = ModelFilter(task=args.pipeline_tags.TextClassification)
        >>> api.list_models(filter=filt)

        >>> # Using `ModelFilter` and `ModelSearchArguments` to find text classification in both PyTorch and TensorFlow
        >>> filt = ModelFilter(
        ...     task=args.pipeline_tags.TextClassification,
        ...     library=[args.library.PyTorch, args.library.TensorFlow],
        ... )
        >>> api.list_models(filter=filt)

        >>> # List only models from the AllenNLP library
        >>> api.list_models(filter="allennlp")
        >>> # Using `ModelFilter` and `ModelSearchArguments`
        >>> filt = ModelFilter(library=args.library.allennlp)
        ```

        Example usage with the `search` argument:

        ```python
        >>> from huggingface_hub import HfApi

        >>> api = HfApi()

        >>> # List all models with "bert" in their name
        >>> api.list_models(search="bert")

        >>> # List all models with "bert" in their name made by google
        >>> api.list_models(search="bert", author="google")
        ```
        """
        path = f"{self.endpoint}/api/models"
        headers = self._build_hf_headers(token=token)
        params = {}
        if filter is not None:
            if isinstance(filter, ModelFilter):
                params = self._unpack_model_filter(filter)
            else:
                params.update({"filter": filter})
            params.update({"full": True})
        if author is not None:
            params.update({"author": author})
        if search is not None:
            params.update({"search": search})
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
        if fetch_config:
            params.update({"config": True})
        if cardData:
            params.update({"cardData": True})

        data = paginate(path, params=params, headers=headers)
        if limit is not None:
            data = islice(data, limit)  # Do not iterate over all pages
        items = [ModelInfo(**x) for x in data]

        if emissions_thresholds is not None:
            if cardData is None:
                raise ValueError(
                    "`emissions_thresholds` were passed without setting"
                    " `cardData=True`."
                )
            else:
                return _filter_emissions(items, *emissions_thresholds)

        return items

    def _unpack_model_filter(self, model_filter: ModelFilter):
        """
        Unpacks a [`ModelFilter`] into something readable for `list_models`
        """
        model_str = ""
        tags = []

        # Handling author
        if model_filter.author is not None:
            model_str = f"{model_filter.author}/"

        # Handling model_name
        if model_filter.model_name is not None:
            model_str += model_filter.model_name

        filter_list: List[str] = []

        # Handling tasks
        if model_filter.task is not None:
            filter_list.extend(
                [model_filter.task]
                if isinstance(model_filter.task, str)
                else model_filter.task
            )

        # Handling dataset
        if model_filter.trained_dataset is not None:
            if not isinstance(model_filter.trained_dataset, (list, tuple)):
                model_filter.trained_dataset = [model_filter.trained_dataset]
            for dataset in model_filter.trained_dataset:
                if "dataset:" not in dataset:
                    dataset = f"dataset:{dataset}"
                filter_list.append(dataset)

        # Handling library
        if model_filter.library:
            filter_list.extend(
                [model_filter.library]
                if isinstance(model_filter.library, str)
                else model_filter.library
            )

        # Handling tags
        if model_filter.tags:
            tags.extend(
                [model_filter.tags]
                if isinstance(model_filter.tags, str)
                else model_filter.tags
            )

        query_dict: Dict[str, Any] = {}
        if model_str is not None:
            query_dict["search"] = model_str
        if len(tags) > 0:
            query_dict["tags"] = tags
        if isinstance(model_filter.language, list):
            filter_list.extend(model_filter.language)
        elif isinstance(model_filter.language, str):
            filter_list.append(model_filter.language)
        query_dict["filter"] = tuple(filter_list)
        return query_dict

    @_deprecate_arguments(
        version="0.14",
        deprecated_args={"cardData"},
        custom_message="Use 'full' instead.",
    )
    @_deprecate_list_output(version="0.14")
    @validate_hf_hub_args
    def list_datasets(
        self,
        *,
        filter: Union[DatasetFilter, str, Iterable[str], None] = None,
        author: Optional[str] = None,
        search: Optional[str] = None,
        sort: Union[Literal["lastModified"], str, None] = None,
        direction: Optional[Literal[-1]] = None,
        limit: Optional[int] = None,
        cardData: Optional[bool] = None,  # deprecated
        full: Optional[bool] = None,
        token: Optional[str] = None,
    ) -> List[DatasetInfo]:
        """
        Get the list of all the datasets on huggingface.co

        Args:
            filter ([`DatasetFilter`] or `str` or `Iterable`, *optional*):
                A string or [`DatasetFilter`] which can be used to identify
                datasets on the hub.
            author (`str`, *optional*):
                A string which identify the author of the returned datasets.
            search (`str`, *optional*):
                A string that will be contained in the returned datasets.
            sort (`Literal["lastModified"]` or `str`, *optional*):
                The key with which to sort the resulting datasets. Possible
                values are the properties of the [`huggingface_hub.hf_api.DatasetInfo`] class.
            direction (`Literal[-1]` or `int`, *optional*):
                Direction in which to sort. The value `-1` sorts by descending
                order while all other values sort by ascending order.
            limit (`int`, *optional*):
                The limit on the number of datasets fetched. Leaving this option
                to `None` fetches all datasets.
            full (`bool`, *optional*):
                Whether to fetch all dataset data, including the `lastModified`
                and the `cardData`. Can contain useful information such as the
                PapersWithCode ID.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            `List[DatasetInfo]`: a list of [`huggingface_hub.hf_api.DatasetInfo`] objects.
             To anticipate future pagination, please consider the return value to be a
             simple iterator.

        Example usage with the `filter` argument:

        ```python
        >>> from huggingface_hub import HfApi

        >>> api = HfApi()

        >>> # List all datasets
        >>> api.list_datasets()

        >>> # Get all valid search arguments
        >>> args = DatasetSearchArguments()

        >>> # List only the text classification datasets
        >>> api.list_datasets(filter="task_categories:text-classification")
        >>> # Using the `DatasetFilter`
        >>> filt = DatasetFilter(task_categories="text-classification")
        >>> # With `DatasetSearchArguments`
        >>> filt = DatasetFilter(task=args.task_categories.text_classification)
        >>> api.list_models(filter=filt)

        >>> # List only the datasets in russian for language modeling
        >>> api.list_datasets(
        ...     filter=("language:ru", "task_ids:language-modeling")
        ... )
        >>> # Using the `DatasetFilter`
        >>> filt = DatasetFilter(language="ru", task_ids="language-modeling")
        >>> # With `DatasetSearchArguments`
        >>> filt = DatasetFilter(
        ...     language=args.language.ru,
        ...     task_ids=args.task_ids.language_modeling,
        ... )
        >>> api.list_datasets(filter=filt)
        ```

        Example usage with the `search` argument:

        ```python
        >>> from huggingface_hub import HfApi

        >>> api = HfApi()

        >>> # List all datasets with "text" in their name
        >>> api.list_datasets(search="text")

        >>> # List all datasets with "text" in their name made by google
        >>> api.list_datasets(search="text", author="google")
        ```
        """
        path = f"{self.endpoint}/api/datasets"
        headers = self._build_hf_headers(token=token)
        params = {}
        if filter is not None:
            if isinstance(filter, DatasetFilter):
                params = self._unpack_dataset_filter(filter)
            else:
                params.update({"filter": filter})
        if author is not None:
            params.update({"author": author})
        if search is not None:
            params.update({"search": search})
        if sort is not None:
            params.update({"sort": sort})
        if direction is not None:
            params.update({"direction": direction})
        if limit is not None:
            params.update({"limit": limit})
        if full or cardData:
            params.update({"full": True})

        data = paginate(path, params=params, headers=headers)
        if limit is not None:
            data = islice(data, limit)  # Do not iterate over all pages
        return [DatasetInfo(**x) for x in data]

    def _unpack_dataset_filter(self, dataset_filter: DatasetFilter):
        """
        Unpacks a [`DatasetFilter`] into something readable for `list_datasets`
        """
        dataset_str = ""

        # Handling author
        if dataset_filter.author is not None:
            dataset_str = f"{dataset_filter.author}/"

        # Handling dataset_name
        if dataset_filter.dataset_name is not None:
            dataset_str += dataset_filter.dataset_name

        filter_list = []
        data_attributes = [
            "benchmark",
            "language_creators",
            "language",
            "multilinguality",
            "size_categories",
            "task_categories",
            "task_ids",
        ]

        for attr in data_attributes:
            curr_attr = getattr(dataset_filter, attr)
            if curr_attr is not None:
                if not isinstance(curr_attr, (list, tuple)):
                    curr_attr = [curr_attr]
                for data in curr_attr:
                    if f"{attr}:" not in data:
                        data = f"{attr}:{data}"
                    filter_list.append(data)

        query_dict: Dict[str, Any] = {}
        if dataset_str is not None:
            query_dict["search"] = dataset_str
        query_dict["filter"] = tuple(filter_list)
        return query_dict

    def list_metrics(self) -> List[MetricInfo]:
        """
        Get the public list of all the metrics on huggingface.co

        Returns:
            `List[MetricInfo]`: a list of [`MetricInfo`] objects which.
        """
        path = f"{self.endpoint}/api/metrics"
        r = requests.get(path)
        hf_raise_for_status(r)
        d = r.json()
        return [MetricInfo(**x) for x in d]

    @_deprecate_list_output(version="0.14")
    @validate_hf_hub_args
    def list_spaces(
        self,
        *,
        filter: Union[str, Iterable[str], None] = None,
        author: Optional[str] = None,
        search: Optional[str] = None,
        sort: Union[Literal["lastModified"], str, None] = None,
        direction: Optional[Literal[-1]] = None,
        limit: Optional[int] = None,
        datasets: Union[str, Iterable[str], None] = None,
        models: Union[str, Iterable[str], None] = None,
        linked: bool = False,
        full: Optional[bool] = None,
        token: Optional[str] = None,
    ) -> List[SpaceInfo]:
        """
        Get the public list of all Spaces on huggingface.co

        Args:
            filter `str` or `Iterable`, *optional*):
                A string tag or list of tags that can be used to identify Spaces on the Hub.
            author (`str`, *optional*):
                A string which identify the author of the returned Spaces.
            search (`str`, *optional*):
                A string that will be contained in the returned Spaces.
            sort (`Literal["lastModified"]` or `str`, *optional*):
                The key with which to sort the resulting Spaces. Possible
                values are the properties of the [`huggingface_hub.hf_api.SpaceInfo`]` class.
            direction (`Literal[-1]` or `int`, *optional*):
                Direction in which to sort. The value `-1` sorts by descending
                order while all other values sort by ascending order.
            limit (`int`, *optional*):
                The limit on the number of Spaces fetched. Leaving this option
                to `None` fetches all Spaces.
            datasets (`str` or `Iterable`, *optional*):
                Whether to return Spaces that make use of a dataset.
                The name of a specific dataset can be passed as a string.
            models (`str` or `Iterable`, *optional*):
                Whether to return Spaces that make use of a model.
                The name of a specific model can be passed as a string.
            linked (`bool`, *optional*):
                Whether to return Spaces that make use of either a model or a dataset.
            full (`bool`, *optional*):
                Whether to fetch all Spaces data, including the `lastModified`
                and the `cardData`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            `List[SpaceInfo]`: a list of [`huggingface_hub.hf_api.SpaceInfo`] objects.
             To anticipate future pagination, please consider the return value to be a
             simple iterator.
        """
        path = f"{self.endpoint}/api/spaces"
        headers = self._build_hf_headers(token=token)
        params: Dict[str, Any] = {}
        if filter is not None:
            params.update({"filter": filter})
        if author is not None:
            params.update({"author": author})
        if search is not None:
            params.update({"search": search})
        if sort is not None:
            params.update({"sort": sort})
        if direction is not None:
            params.update({"direction": direction})
        if limit is not None:
            params.update({"limit": limit})
        if full:
            params.update({"full": True})
        if linked:
            params.update({"linked": True})
        if datasets is not None:
            params.update({"datasets": datasets})
        if models is not None:
            params.update({"models": models})

        data = paginate(path, params=params, headers=headers)
        if limit is not None:
            data = islice(data, limit)  # Do not iterate over all pages
        return [SpaceInfo(**x) for x in data]

    @validate_hf_hub_args
    def model_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        timeout: Optional[float] = None,
        securityStatus: Optional[bool] = None,
        files_metadata: bool = False,
        token: Optional[Union[bool, str]] = None,
    ) -> ModelInfo:
        """
        Get info on one specific model on huggingface.co

        Model can be private if you pass an acceptable token or are logged in.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            revision (`str`, *optional*):
                The revision of the model repository from which to get the
                information.
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            securityStatus (`bool`, *optional*):
                Whether to retrieve the security status from the model
                repository as well.
            files_metadata (`bool`, *optional*):
                Whether or not to retrieve metadata for files in the repository
                (size, LFS metadata, etc). Defaults to `False`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            [`huggingface_hub.hf_api.ModelInfo`]: The model repository information.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.

        </Tip>
        """
        headers = self._build_hf_headers(token=token)
        path = (
            f"{self.endpoint}/api/models/{repo_id}"
            if revision is None
            else (
                f"{self.endpoint}/api/models/{repo_id}/revision/{quote(revision, safe='')}"
            )
        )
        params = {}
        if securityStatus:
            params["securityStatus"] = True
        if files_metadata:
            params["blobs"] = True
        r = requests.get(
            path,
            headers=headers,
            timeout=timeout,
            params=params,
        )
        hf_raise_for_status(r)
        d = r.json()
        return ModelInfo(**d)

    @validate_hf_hub_args
    def dataset_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        timeout: Optional[float] = None,
        files_metadata: bool = False,
        token: Optional[Union[bool, str]] = None,
    ) -> DatasetInfo:
        """
        Get info on one specific dataset on huggingface.co.

        Dataset can be private if you pass an acceptable token.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            revision (`str`, *optional*):
                The revision of the dataset repository from which to get the
                information.
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            files_metadata (`bool`, *optional*):
                Whether or not to retrieve metadata for files in the repository
                (size, LFS metadata, etc). Defaults to `False`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            [`hf_api.DatasetInfo`]: The dataset repository information.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.

        </Tip>
        """
        headers = self._build_hf_headers(token=token)
        path = (
            f"{self.endpoint}/api/datasets/{repo_id}"
            if revision is None
            else (
                f"{self.endpoint}/api/datasets/{repo_id}/revision/{quote(revision, safe='')}"
            )
        )
        params = {}
        if files_metadata:
            params["blobs"] = True

        r = requests.get(path, headers=headers, timeout=timeout, params=params)
        hf_raise_for_status(r)
        d = r.json()
        return DatasetInfo(**d)

    @validate_hf_hub_args
    def space_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        timeout: Optional[float] = None,
        files_metadata: bool = False,
        token: Optional[Union[bool, str]] = None,
    ) -> SpaceInfo:
        """
        Get info on one specific Space on huggingface.co.

        Space can be private if you pass an acceptable token.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            revision (`str`, *optional*):
                The revision of the space repository from which to get the
                information.
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            files_metadata (`bool`, *optional*):
                Whether or not to retrieve metadata for files in the repository
                (size, LFS metadata, etc). Defaults to `False`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            [`~hf_api.SpaceInfo`]: The space repository information.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.

        </Tip>
        """
        headers = self._build_hf_headers(token=token)
        path = (
            f"{self.endpoint}/api/spaces/{repo_id}"
            if revision is None
            else (
                f"{self.endpoint}/api/spaces/{repo_id}/revision/{quote(revision, safe='')}"
            )
        )
        params = {}
        if files_metadata:
            params["blobs"] = True

        r = requests.get(path, headers=headers, timeout=timeout, params=params)
        hf_raise_for_status(r)
        d = r.json()
        return SpaceInfo(**d)

    @validate_hf_hub_args
    def repo_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        repo_type: Optional[str] = None,
        timeout: Optional[float] = None,
        files_metadata: bool = False,
        token: Optional[Union[bool, str]] = None,
    ) -> Union[ModelInfo, DatasetInfo, SpaceInfo]:
        """
        Get the info object for a given repo of a given type.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            revision (`str`, *optional*):
                The revision of the repository from which to get the
                information.
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            files_metadata (`bool`, *optional*):
                Whether or not to retrieve metadata for files in the repository
                (size, LFS metadata, etc). Defaults to `False`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            `Union[SpaceInfo, DatasetInfo, ModelInfo]`: The repository information, as a
            [`huggingface_hub.hf_api.DatasetInfo`], [`huggingface_hub.hf_api.ModelInfo`]
            or [`huggingface_hub.hf_api.SpaceInfo`] object.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.

        </Tip>
        """
        if repo_type is None or repo_type == "model":
            method = self.model_info
        elif repo_type == "dataset":
            method = self.dataset_info  # type: ignore
        elif repo_type == "space":
            method = self.space_info  # type: ignore
        else:
            raise ValueError("Unsupported repo type.")
        return method(
            repo_id,
            revision=revision,
            token=token,
            timeout=timeout,
            files_metadata=files_metadata,
        )

    @validate_hf_hub_args
    def list_repo_files(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        repo_type: Optional[str] = None,
        timeout: Optional[float] = None,
        token: Optional[Union[bool, str]] = None,
    ) -> List[str]:
        """
        Get the list of files in a given repo.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            revision (`str`, *optional*):
                The revision of the model repository from which to get the
                information.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            `List[str]`: the list of files in a given repository.
        """
        repo_info = self.repo_info(
            repo_id,
            revision=revision,
            repo_type=repo_type,
            token=token,
            timeout=timeout,
        )
        return [f.rfilename for f in repo_info.siblings]

    @validate_hf_hub_args
    @_deprecate_positional_args(version="0.12")
    def create_repo(
        self,
        repo_id: str,
        *,
        token: Optional[str] = None,
        private: bool = False,
        repo_type: Optional[str] = None,
        exist_ok: bool = False,
        space_sdk: Optional[str] = None,
    ) -> str:
        """Create an empty repo on the HuggingFace Hub.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            private (`bool`, *optional*, defaults to `False`):
                Whether the model repo should be private.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            exist_ok (`bool`, *optional*, defaults to `False`):
                If `True`, do not raise an error if repo already exists.
            space_sdk (`str`, *optional*):
                Choice of SDK to use if repo_type is "space". Can be
                "streamlit", "gradio", or "static".

        Returns:
            `str`: URL to the newly created repo.
        """
        organization, name = repo_id.split("/") if "/" in repo_id else (None, repo_id)

        path = f"{self.endpoint}/api/repos/create"

        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        json = {"name": name, "organization": organization, "private": private}
        if repo_type is not None:
            json["type"] = repo_type
        if repo_type == "space":
            if space_sdk is None:
                raise ValueError(
                    "No space_sdk provided. `create_repo` expects space_sdk to be one"
                    f" of {SPACES_SDK_TYPES} when repo_type is 'space'`"
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

        if getattr(self, "_lfsmultipartthresh", None):
            # Testing purposes only.
            # See https://github.com/huggingface/huggingface_hub/pull/733/files#r820604472
            json["lfsmultipartthresh"] = self._lfsmultipartthresh  # type: ignore
        headers = self._build_hf_headers(token=token, is_write_action=True)
        r = requests.post(path, headers=headers, json=json)

        try:
            hf_raise_for_status(r)
        except HTTPError as err:
            if exist_ok and err.response.status_code == 409:
                # Repo already exists and `exist_ok=True`
                pass
            else:
                raise

        d = r.json()
        return d["url"]

    @validate_hf_hub_args
    def delete_repo(
        self,
        repo_id: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ):
        """
        Delete a repo from the HuggingFace Hub. CAUTION: this is irreversible.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        organization, name = repo_id.split("/") if "/" in repo_id else (None, repo_id)

        path = f"{self.endpoint}/api/repos/delete"

        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        json = {"name": name, "organization": organization}
        if repo_type is not None:
            json["type"] = repo_type

        headers = self._build_hf_headers(token=token, is_write_action=True)
        r = requests.delete(path, headers=headers, json=json)
        hf_raise_for_status(r)

    @validate_hf_hub_args
    def update_repo_visibility(
        self,
        repo_id: str,
        private: bool = False,
        *,
        token: Optional[str] = None,
        organization: Optional[str] = None,
        repo_type: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, bool]:
        """Update the visibility setting of a repository.

        Args:
            repo_id (`str`, *optional*):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            private (`bool`, *optional*, defaults to `False`):
                Whether the model repo should be private.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.

        Returns:
            The HTTP response in json.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        organization, name = repo_id.split("/") if "/" in repo_id else (None, repo_id)

        if organization is None:
            namespace = self.whoami(token)["name"]
        else:
            namespace = organization

        if repo_type is None:
            repo_type = REPO_TYPE_MODEL  # default repo type

        r = requests.put(
            url=f"{self.endpoint}/api/{repo_type}s/{namespace}/{name}/settings",
            headers=self._build_hf_headers(token=token, is_write_action=True),
            json={"private": private},
        )
        hf_raise_for_status(r)
        return r.json()

    def move_repo(
        self,
        from_id: str,
        to_id: str,
        *,
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Moving a repository from namespace1/repo_name1 to namespace2/repo_name2

        Note there are certain limitations. For more information about moving
        repositories, please see
        https://hf.co/docs/hub/main#how-can-i-rename-or-transfer-a-repo.

        Args:
            from_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`. Original repository identifier.
            to_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`. Final repository identifier.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        if len(from_id.split("/")) != 2:
            raise ValueError(
                f"Invalid repo_id: {from_id}. It should have a namespace"
                " (:namespace:/:repo_name:)"
            )

        if len(to_id.split("/")) != 2:
            raise ValueError(
                f"Invalid repo_id: {to_id}. It should have a namespace"
                " (:namespace:/:repo_name:)"
            )

        if repo_type is None:
            repo_type = REPO_TYPE_MODEL  # Hub won't accept `None`.

        json = {"fromRepo": from_id, "toRepo": to_id, "type": repo_type}

        path = f"{self.endpoint}/api/repos/move"
        headers = self._build_hf_headers(token=token, is_write_action=True)
        r = requests.post(path, headers=headers, json=json)
        try:
            hf_raise_for_status(r)
        except HfHubHTTPError as e:
            e.append_to_message(
                "\nFor additional documentation please see"
                " https://hf.co/docs/hub/main#how-can-i-rename-or-transfer-a-repo."
            )
            raise

    @validate_hf_hub_args
    def create_commit(
        self,
        repo_id: str,
        operations: Iterable[CommitOperation],
        *,
        commit_message: str,
        commit_description: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: Optional[bool] = None,
        num_threads: int = 5,
        parent_commit: Optional[str] = None,
    ) -> CommitInfo:
        """
        Creates a commit in the given repo, deleting & uploading files as needed.

        Args:
            repo_id (`str`):
                The repository in which the commit will be created, for example:
                `"username/custom_transformers"`

            operations (`Iterable` of [`~hf_api.CommitOperation`]):
                An iterable of operations to include in the commit, either:

                    - [`~hf_api.CommitOperationAdd`] to upload a file
                    - [`~hf_api.CommitOperationDelete`] to delete a file

            commit_message (`str`):
                The summary (first line) of the commit that will be created.

            commit_description (`str`, *optional*):
                The description of the commit that will be created

            token (`str`, *optional*):
                Authentication token, obtained with `HfApi.login` method. Will
                default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.

            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the `"main"` branch.

            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request with that commit. Defaults to `False`.
                If `revision` is not set, PR is opened against the `"main"` branch. If
                `revision` is set and is a branch, PR is opened against this branch. If
                `revision` is set and is not a branch name (example: a commit oid), an
                `RevisionNotFoundError` is returned by the server.

            num_threads (`int`, *optional*):
                Number of concurrent threads for uploading files. Defaults to 5.
                Setting it to 2 means at most 2 files will be uploaded concurrently.

            parent_commit (`str`, *optional*):
                The OID / SHA of the parent commit, as a hexadecimal string.
                Shorthands (7 first characters) are also supported.If specified and `create_pr` is `False`,
                the commit will fail if `revision` does not point to `parent_commit`. If specified and `create_pr`
                is `True`, the pull request will be created from `parent_commit`. Specifying `parent_commit`
                ensures the repo has not changed before committing the changes, and can be especially useful
                if the repo is updated / committed to concurrently.

        Returns:
            [`CommitInfo`]:
                Instance of [`CommitInfo`] containing information about the newly
                created commit (commit hash, commit url, pr url, commit message,...).

        Raises:
            [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
                If commit message is empty.
            [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
                If parent commit is not a valid commit OID.
            [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
                If the Hub API returns an HTTP 400 error (bad request)
            [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
                If `create_pr` is `True` and revision is neither `None` nor `"main"`.
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private
                but not authenticated or repo does not exist.

        <Tip warning={true}>

        `create_commit` assumes that the repo already exists on the Hub. If you get a
        Client error 404, please make sure you are authenticated and that `repo_id` and
        `repo_type` are set correctly. If repo does not exist, create it first using
        [`~hf_api.create_repo`].

        </Tip>

        <Tip warning={true}>

        `create_commit` is limited to 25k LFS files and a 1GB payload for regular files.

        </Tip>
        """
        _CREATE_COMMIT_NO_REPO_ERROR_MESSAGE = (
            "\nNote: Creating a commit assumes that the repo already exists on the"
            " Huggingface Hub. Please use `create_repo` if it's not the case."
        )

        if parent_commit is not None and not REGEX_COMMIT_OID.fullmatch(parent_commit):
            raise ValueError(
                "`parent_commit` is not a valid commit OID. It must match the"
                f" following regex: {REGEX_COMMIT_OID}"
            )

        if commit_message is None or len(commit_message) == 0:
            raise ValueError("`commit_message` can't be empty, please pass a value.")

        commit_description = (
            commit_description if commit_description is not None else ""
        )
        repo_type = repo_type if repo_type is not None else REPO_TYPE_MODEL
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
        revision = (
            quote(revision, safe="") if revision is not None else DEFAULT_REVISION
        )
        create_pr = create_pr if create_pr is not None else False

        operations = list(operations)
        additions = [op for op in operations if isinstance(op, CommitOperationAdd)]
        nb_additions = len(additions)
        nb_deletions = len(operations) - nb_additions

        logger.debug(
            f"About to commit to the hub: {len(additions)} addition(s) and"
            f" {nb_deletions} deletion(s)."
        )

        # If updating twice the same file or update then delete a file in a single commit
        warn_on_overwriting_operations(operations)

        try:
            upload_modes = fetch_upload_modes(
                additions=additions,
                repo_type=repo_type,
                repo_id=repo_id,
                token=token or self.token,
                revision=revision,
                endpoint=self.endpoint,
                create_pr=create_pr,
            )
        except RepositoryNotFoundError as e:
            e.append_to_message(_CREATE_COMMIT_NO_REPO_ERROR_MESSAGE)
            raise

        upload_lfs_files(
            additions=[
                addition
                for addition in additions
                if upload_modes[addition.path_in_repo] == "lfs"
            ],
            repo_type=repo_type,
            repo_id=repo_id,
            token=token or self.token,
            endpoint=self.endpoint,
            num_threads=num_threads,
        )
        commit_payload = prepare_commit_payload(
            operations=operations,
            upload_modes=upload_modes,
            commit_message=commit_message,
            commit_description=commit_description,
            parent_commit=parent_commit,
        )
        commit_url = f"{self.endpoint}/api/{repo_type}s/{repo_id}/commit/{revision}"

        def _payload_as_ndjson() -> Iterable[bytes]:
            for item in commit_payload:
                yield json.dumps(item).encode()
                yield b"\n"

        headers = {
            # See https://github.com/huggingface/huggingface_hub/issues/1085#issuecomment-1265208073
            "Content-Type": "application/x-ndjson",
            **self._build_hf_headers(token=token, is_write_action=True),
        }

        try:
            commit_resp = requests.post(
                url=commit_url,
                headers=headers,
                data=_payload_as_ndjson(),  # type: ignore
                params={"create_pr": "1"} if create_pr else None,
            )
            hf_raise_for_status(commit_resp, endpoint_name="commit")
        except RepositoryNotFoundError as e:
            e.append_to_message(_CREATE_COMMIT_NO_REPO_ERROR_MESSAGE)
            raise
        except EntryNotFoundError as e:
            if nb_deletions > 0 and "A file with this name doesn't exist" in str(e):
                e.append_to_message(
                    "\nMake sure to differentiate file and folder paths in delete"
                    " operations with a trailing '/' or using `is_folder=True/False`."
                )
            raise

        commit_data = commit_resp.json()
        return CommitInfo(
            commit_url=commit_data["commitUrl"],
            commit_message=commit_message,
            commit_description=commit_description,
            oid=commit_data["commitOid"],
            pr_url=commit_data["pullRequestUrl"] if create_pr else None,
        )

    @validate_hf_hub_args
    def upload_file(
        self,
        *,
        path_or_fileobj: Union[str, bytes, BinaryIO],
        path_in_repo: str,
        repo_id: str,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        create_pr: Optional[bool] = None,
        parent_commit: Optional[str] = None,
    ) -> str:
        """
        Upload a local file (up to 50 GB) to the given repo. The upload is done
        through a HTTP post request, and doesn't require git or git-lfs to be
        installed.

        Args:
            path_or_fileobj (`str`, `bytes`, or `IO`):
                Path to a file on the local machine or binary data stream /
                fileobj / buffer.
            path_in_repo (`str`):
                Relative filepath in the repo, for example:
                `"checkpoints/1fec34a/weights.bin"`
            repo_id (`str`):
                The repository to which the file will be uploaded, for example:
                `"username/custom_transformers"`
            token (`str`, *optional*):
                Authentication token, obtained with `HfApi.login` method. Will
                default to the stored token.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the `"main"` branch.
            commit_message (`str`, *optional*):
                The summary / title / first line of the generated commit
            commit_description (`str` *optional*)
                The description of the generated commit
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request with that commit. Defaults to `False`.
                If `revision` is not set, PR is opened against the `"main"` branch. If
                `revision` is set and is a branch, PR is opened against this branch. If
                `revision` is set and is not a branch name (example: a commit oid), an
                `RevisionNotFoundError` is returned by the server.
            parent_commit (`str`, *optional*):
                The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
                If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`.
                If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`.
                Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be
                especially useful if the repo is updated / committed to concurrently.


        Returns:
            `str`: The URL to visualize the uploaded file on the hub

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.

        </Tip>

        <Tip warning={true}>

        `upload_file` assumes that the repo already exists on the Hub. If you get a
        Client error 404, please make sure you are authenticated and that `repo_id` and
        `repo_type` are set correctly. If repo does not exist, create it first using
        [`~hf_api.create_repo`].

        </Tip>

        Example:

        ```python
        >>> from huggingface_hub import upload_file

        >>> with open("./local/filepath", "rb") as fobj:
        ...     upload_file(
        ...         path_or_fileobj=fileobj,
        ...         path_in_repo="remote/file/path.h5",
        ...         repo_id="username/my-dataset",
        ...         repo_type="dataset",
        ...         token="my_token",
        ...     )
        "https://huggingface.co/datasets/username/my-dataset/blob/main/remote/file/path.h5"

        >>> upload_file(
        ...     path_or_fileobj=".\\\\local\\\\file\\\\path",
        ...     path_in_repo="remote/file/path.h5",
        ...     repo_id="username/my-model",
        ...     token="my_token",
        ... )
        "https://huggingface.co/username/my-model/blob/main/remote/file/path.h5"

        >>> upload_file(
        ...     path_or_fileobj=".\\\\local\\\\file\\\\path",
        ...     path_in_repo="remote/file/path.h5",
        ...     repo_id="username/my-model",
        ...     token="my_token",
        ...     create_pr=True,
        ... )
        "https://huggingface.co/username/my-model/blob/refs%2Fpr%2F1/remote/file/path.h5"
        ```
        """
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")

        commit_message = (
            commit_message
            if commit_message is not None
            else f"Upload {path_in_repo} with huggingface_hub"
        )
        operation = CommitOperationAdd(
            path_or_fileobj=path_or_fileobj,
            path_in_repo=path_in_repo,
        )

        commit_info = self.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=[operation],
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            revision=revision,
            create_pr=create_pr,
            parent_commit=parent_commit,
        )

        if commit_info.pr_url is not None:
            revision = quote(_parse_revision_from_pr_url(commit_info.pr_url), safe="")
        if repo_type in REPO_TYPES_URL_PREFIXES:
            repo_id = REPO_TYPES_URL_PREFIXES[repo_type] + repo_id
        revision = revision if revision is not None else DEFAULT_REVISION
        # Similar to `hf_hub_url` but it's "blob" instead of "resolve"
        return f"{self.endpoint}/{repo_id}/blob/{revision}/{path_in_repo}"

    @validate_hf_hub_args
    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: Union[str, Path],
        path_in_repo: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: Optional[bool] = None,
        parent_commit: Optional[str] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
    ):
        """
        Upload a local folder to the given repo. The upload is done
        through a HTTP requests, and doesn't require git or git-lfs to be
        installed.

        The structure of the folder will be preserved. Files with the same name
        already present in the repository will be overwritten, others will be left untouched.

        Use the `allow_patterns` and `ignore_patterns` arguments to specify which files
        to upload. These parameters accept either a single pattern or a list of
        patterns. Patterns are Standard Wildcards (globbing patterns) as documented
        [here](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm). If both
        `allow_patterns` and `ignore_patterns` are provided, both constraints apply. By
        default, all files from the folder are uploaded.

        Uses `HfApi.create_commit` under the hood.

        Args:
            repo_id (`str`):
                The repository to which the file will be uploaded, for example:
                `"username/custom_transformers"`
            folder_path (`str` or `Path`):
                Path to the folder to upload on the local file system
            path_in_repo (`str`, *optional*):
                Relative path of the directory in the repo, for example:
                `"checkpoints/1fec34a/results"`. Will default to the root folder of the repository.
            token (`str`, *optional*):
                Authentication token, obtained with `HfApi.login` method. Will
                default to the stored token.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the `"main"` branch.
            commit_message (`str`, *optional*):
                The summary / title / first line of the generated commit. Defaults to:
                `f"Upload {path_in_repo} with huggingface_hub"`
            commit_description (`str` *optional*):
                The description of the generated commit
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request with that commit. Defaults to `False`.
                If `revision` is not set, PR is opened against the `"main"` branch. If
                `revision` is set and is a branch, PR is opened against this branch. If
                `revision` is set and is not a branch name (example: a commit oid), an
                `RevisionNotFoundError` is returned by the server.
            parent_commit (`str`, *optional*):
                The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
                If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`.
                If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`.
                Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be
                especially useful if the repo is updated / committed to concurrently.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are uploaded.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not uploaded.

        Returns:
            `str`: A URL to visualize the uploaded folder on the hub

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
            if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            if some parameter value is invalid

        </Tip>

        <Tip warning={true}>

        `upload_folder` assumes that the repo already exists on the Hub. If you get a
        Client error 404, please make sure you are authenticated and that `repo_id` and
        `repo_type` are set correctly. If repo does not exist, create it first using
        [`~hf_api.create_repo`].

        </Tip>

        Example:

        ```python
        >>> upload_folder(
        ...     folder_path="local/checkpoints",
        ...     path_in_repo="remote/experiment/checkpoints",
        ...     repo_id="username/my-dataset",
        ...     repo_type="datasets",
        ...     token="my_token",
        ...     ignore_patterns="**/logs/*.txt",
        ... )
        # "https://huggingface.co/datasets/username/my-dataset/tree/main/remote/experiment/checkpoints"

        >>> upload_folder(
        ...     folder_path="local/checkpoints",
        ...     path_in_repo="remote/experiment/checkpoints",
        ...     repo_id="username/my-dataset",
        ...     repo_type="datasets",
        ...     token="my_token",
        ...     create_pr=True,
        ... )
        # "https://huggingface.co/datasets/username/my-dataset/tree/refs%2Fpr%2F1/remote/experiment/checkpoints"

        ```
        """
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")

        # By default, upload folder to the root directory in repo.
        if path_in_repo is None:
            path_in_repo = ""

        commit_message = (
            commit_message
            if commit_message is not None
            else f"Upload {path_in_repo} with huggingface_hub"
        )

        files_to_add = _prepare_upload_folder_commit(
            folder_path,
            path_in_repo,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

        commit_info = self.create_commit(
            repo_type=repo_type,
            repo_id=repo_id,
            operations=files_to_add,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            revision=revision,
            create_pr=create_pr,
            parent_commit=parent_commit,
        )

        if commit_info.pr_url is not None:
            revision = quote(_parse_revision_from_pr_url(commit_info.pr_url), safe="")
        if repo_type in REPO_TYPES_URL_PREFIXES:
            repo_id = REPO_TYPES_URL_PREFIXES[repo_type] + repo_id
        revision = revision if revision is not None else DEFAULT_REVISION
        # Similar to `hf_hub_url` but it's "tree" instead of "resolve"
        return f"{self.endpoint}/{repo_id}/tree/{revision}/{path_in_repo}"

    @validate_hf_hub_args
    def delete_file(
        self,
        path_in_repo: str,
        repo_id: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        create_pr: Optional[bool] = None,
        parent_commit: Optional[str] = None,
    ) -> CommitInfo:
        """
        Deletes a file in the given repo.

        Args:
            path_in_repo (`str`):
                Relative filepath in the repo, for example:
                `"checkpoints/1fec34a/weights.bin"`
            repo_id (`str`):
                The repository from which the file will be deleted, for example:
                `"username/custom_transformers"`
            token (`str`, *optional*):
                Authentication token, obtained with `HfApi.login` method. Will
                default to the stored token.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if the file is in a dataset or
                space, `None` or `"model"` if in a model. Default is `None`.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the `"main"` branch.
            commit_message (`str`, *optional*):
                The summary / title / first line of the generated commit. Defaults to
                `f"Delete {path_in_repo} with huggingface_hub"`.
            commit_description (`str` *optional*)
                The description of the generated commit
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request with that commit. Defaults to `False`.
                If `revision` is not set, PR is opened against the `"main"` branch. If
                `revision` is set and is a branch, PR is opened against this branch. If
                `revision` is set and is not a branch name (example: a commit oid), an
                `RevisionNotFoundError` is returned by the server.
            parent_commit (`str`, *optional*):
                The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
                If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`.
                If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`.
                Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be
                especially useful if the repo is updated / committed to concurrently.


        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.
            - [`~utils.EntryNotFoundError`]
              If the file to download cannot be found.

        </Tip>

        """
        commit_message = (
            commit_message
            if commit_message is not None
            else f"Delete {path_in_repo} with huggingface_hub"
        )

        operations = [CommitOperationDelete(path_in_repo=path_in_repo)]

        return self.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            operations=operations,
            revision=revision,
            commit_message=commit_message,
            commit_description=commit_description,
            create_pr=create_pr,
            parent_commit=parent_commit,
        )

    @validate_hf_hub_args
    def delete_folder(
        self,
        path_in_repo: str,
        repo_id: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        create_pr: Optional[bool] = None,
        parent_commit: Optional[str] = None,
    ) -> CommitInfo:
        """
        Deletes a folder in the given repo.

        Simple wrapper around [`create_commit`] method.

        Args:
            path_in_repo (`str`):
                Relative folder path in the repo, for example: `"checkpoints/1fec34a"`.
            repo_id (`str`):
                The repository from which the folder will be deleted, for example:
                `"username/custom_transformers"`
            token (`str`, *optional*):
                Authentication token, obtained with `HfApi.login` method. Will default
                to the stored token.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if the folder is in a dataset or
                space, `None` or `"model"` if in a model. Default is `None`.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the `"main"` branch.
            commit_message (`str`, *optional*):
                The summary / title / first line of the generated commit. Defaults to
                `f"Delete folder {path_in_repo} with huggingface_hub"`.
            commit_description (`str` *optional*)
                The description of the generated commit.
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request with that commit. Defaults to `False`.
                If `revision` is not set, PR is opened against the `"main"` branch. If
                `revision` is set and is a branch, PR is opened against this branch. If
                `revision` is set and is not a branch name (example: a commit oid), an
                `RevisionNotFoundError` is returned by the server.
            parent_commit (`str`, *optional*):
                The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
                If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`.
                If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`.
                Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be
                especially useful if the repo is updated / committed to concurrently.
        """
        return self.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            operations=[
                CommitOperationDelete(path_in_repo=path_in_repo, is_folder=True)
            ],
            revision=revision,
            commit_message=(
                commit_message
                if commit_message is not None
                else f"Delete folder {path_in_repo} with huggingface_hub"
            ),
            commit_description=commit_description,
            create_pr=create_pr,
            parent_commit=parent_commit,
        )

    @validate_hf_hub_args
    def create_branch(
        self,
        repo_id: str,
        *,
        branch: str,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> None:
        """
        Create a new branch from `main` on a repo on the Hub.

        Args:
            repo_id (`str`):
                The repository in which the branch will be created.
                Example: `"user/my-cool-model"`.

            branch (`str`):
                The name of the branch to create.

            token (`str`, *optional*):
                Authentication token. Will default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if creating a branch on a dataset or
                space, `None` or `"model"` if tagging a model. Default is `None`.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private
                but not authenticated or repo does not exist.
            [`~utils.BadRequestError`]:
                If invalid reference for a branch. Ex: `refs/pr/5` or 'refs/foo/bar'.
            [`~utils.HfHubHTTPError`]:
                If the branch already exists on the repo (error 409).
        """
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL
        branch = quote(branch, safe="")

        # Prepare request
        branch_url = f"{self.endpoint}/api/{repo_type}s/{repo_id}/branch/{branch}"
        headers = self._build_hf_headers(token=token, is_write_action=True)

        # Create branch
        response = requests.post(url=branch_url, headers=headers)
        hf_raise_for_status(response)

    @validate_hf_hub_args
    def delete_branch(
        self,
        repo_id: str,
        *,
        branch: str,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> None:
        """
        Delete a branch from a repo on the Hub.

        Args:
            repo_id (`str`):
                The repository in which a branch will be deleted.
                Example: `"user/my-cool-model"`.

            branch (`str`):
                The name of the branch to delete.

            token (`str`, *optional*):
                Authentication token. Will default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if creating a branch on a dataset or
                space, `None` or `"model"` if tagging a model. Default is `None`.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private
                but not authenticated or repo does not exist.
            [`~utils.HfHubHTTPError`]:
                If trying to delete a protected branch. Ex: `main` cannot be deleted.
            [`~utils.HfHubHTTPError`]:
                If trying to delete a branch that does not exist.

        """
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL
        branch = quote(branch, safe="")

        # Prepare request
        branch_url = f"{self.endpoint}/api/{repo_type}s/{repo_id}/branch/{branch}"
        headers = self._build_hf_headers(token=token, is_write_action=True)

        # Delete branch
        response = requests.delete(url=branch_url, headers=headers)
        hf_raise_for_status(response)

    @validate_hf_hub_args
    def create_tag(
        self,
        repo_id: str,
        *,
        tag: str,
        tag_message: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> None:
        """
        Tag a given commit of a repo on the Hub.

        Args:
            repo_id (`str`):
                The repository in which a commit will be tagged.
                Example: `"user/my-cool-model"`.

            tag (`str`):
                The name of the tag to create.

            tag_message (`str`, *optional*):
                The description of the tag to create.

            revision (`str`, *optional*):
                The git revision to tag. It can be a branch name or the OID/SHA of a
                commit, as a hexadecimal string. Shorthands (7 first characters) are
                also supported. Defaults to the head of the `"main"` branch.

            token (`str`, *optional*):
                Authentication token. Will default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if tagging a dataset or
                space, `None` or `"model"` if tagging a model. Default is
                `None`.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private
                but not authenticated or repo does not exist.
            [`~utils.RevisionNotFoundError`]:
                If revision is not found (error 404) on the repo.
        """
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL
        revision = (
            quote(revision, safe="") if revision is not None else DEFAULT_REVISION
        )

        # Prepare request
        tag_url = f"{self.endpoint}/api/{repo_type}s/{repo_id}/tag/{revision}"
        headers = self._build_hf_headers(token=token, is_write_action=True)
        payload = {"tag": tag}
        if tag_message is not None:
            payload["message"] = tag_message

        # Tag
        response = requests.post(url=tag_url, headers=headers, json=payload)
        hf_raise_for_status(response)

    @validate_hf_hub_args
    def delete_tag(
        self,
        repo_id: str,
        *,
        tag: str,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> None:
        """
        Delete a tag from a repo on the Hub.

        Args:
            repo_id (`str`):
                The repository in which a tag will be deleted.
                Example: `"user/my-cool-model"`.

            tag (`str`):
                The name of the tag to delete.

            token (`str`, *optional*):
                Authentication token. Will default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if tagging a dataset or space, `None` or
                `"model"` if tagging a model. Default is `None`.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private
                but not authenticated or repo does not exist.
            [`~utils.RevisionNotFoundError`]:
                If tag is not found.
        """
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL
        tag = quote(tag, safe="")

        # Prepare request
        tag_url = f"{self.endpoint}/api/{repo_type}s/{repo_id}/tag/{tag}"
        headers = self._build_hf_headers(token=token, is_write_action=True)

        # Un-tag
        response = requests.delete(url=tag_url, headers=headers)
        hf_raise_for_status(response)

    @validate_hf_hub_args
    def get_full_repo_name(
        self,
        model_id: str,
        *,
        organization: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
    ):
        """
        Returns the repository name for a given model ID and optional
        organization.

        Args:
            model_id (`str`):
                The name of the model.
            organization (`str`, *optional*):
                If passed, the repository name will be in the organization
                namespace instead of the user namespace.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            `str`: The repository name in the user's namespace
            ({username}/{model_id}) if no organization is passed, and under the
            organization namespace ({organization}/{model_id}) otherwise.
        """
        if organization is None:
            if "/" in model_id:
                username = model_id.split("/")[0]
            else:
                username = self.whoami(token=token)["name"]  # type: ignore
            return f"{username}/{model_id}"
        else:
            return f"{organization}/{model_id}"

    @validate_hf_hub_args
    def get_repo_discussions(
        self,
        repo_id: str,
        *,
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Iterator[Discussion]:
        """
        Fetches Discussions and Pull Requests for the given repo.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if fetching from a dataset or
                space, `None` or `"model"` if fetching from a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token).

        Returns:
            `Iterator[Discussion]`: An iterator of [`Discussion`] objects.

        Example:
            Collecting all discussions of a repo in a list:

            ```python
            >>> from huggingface_hub import get_repo_discussions
            >>> discussions_list = list(get_repo_discussions(repo_id="bert-base-uncased"))
            ```

            Iterating over discussions of a repo:

            ```python
            >>> from huggingface_hub import get_repo_discussions
            >>> for discussion in get_repo_discussions(repo_id="bert-base-uncased"):
            ...     print(discussion.num, discussion.title)
            ```
        """
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL

        headers = self._build_hf_headers(token=token)

        def _fetch_discussion_page(page_index: int):
            path = (
                f"{self.endpoint}/api/{repo_type}s/{repo_id}/discussions?p={page_index}"
            )
            resp = requests.get(path, headers=headers)
            hf_raise_for_status(resp)
            paginated_discussions = resp.json()
            total = paginated_discussions["count"]
            start = paginated_discussions["start"]
            discussions = paginated_discussions["discussions"]
            has_next = (start + len(discussions)) < total
            return discussions, has_next

        has_next, page_index = True, 0

        while has_next:
            discussions, has_next = _fetch_discussion_page(page_index=page_index)
            for discussion in discussions:
                yield Discussion(
                    title=discussion["title"],
                    num=discussion["num"],
                    author=discussion.get("author", {}).get("name", "deleted"),
                    created_at=parse_datetime(discussion["createdAt"]),
                    status=discussion["status"],
                    repo_id=discussion["repo"]["name"],
                    repo_type=discussion["repo"]["type"],
                    is_pull_request=discussion["isPullRequest"],
                )
            page_index = page_index + 1

    @validate_hf_hub_args
    def get_discussion_details(
        self,
        repo_id: str,
        discussion_num: int,
        *,
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
    ) -> DiscussionWithDetails:
        """Fetches a Discussion's / Pull Request 's details from the Hub.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns: [`DiscussionWithDetails`]

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        if not isinstance(discussion_num, int) or discussion_num <= 0:
            raise ValueError("Invalid discussion_num, must be a positive integer")
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL

        path = (
            f"{self.endpoint}/api/{repo_type}s/{repo_id}/discussions/{discussion_num}"
        )
        headers = self._build_hf_headers(token=token)
        resp = requests.get(path, params={"diff": "1"}, headers=headers)
        hf_raise_for_status(resp)

        discussion_details = resp.json()
        is_pull_request = discussion_details["isPullRequest"]

        target_branch = (
            discussion_details["changes"]["base"] if is_pull_request else None
        )
        conflicting_files = (
            discussion_details["filesWithConflicts"] if is_pull_request else None
        )
        merge_commit_oid = (
            discussion_details["changes"].get("mergeCommitId", None)
            if is_pull_request
            else None
        )

        return DiscussionWithDetails(
            title=discussion_details["title"],
            num=discussion_details["num"],
            author=discussion_details.get("author", {}).get("name", "deleted"),
            created_at=parse_datetime(discussion_details["createdAt"]),
            status=discussion_details["status"],
            repo_id=discussion_details["repo"]["name"],
            repo_type=discussion_details["repo"]["type"],
            is_pull_request=discussion_details["isPullRequest"],
            events=[deserialize_event(evt) for evt in discussion_details["events"]],
            conflicting_files=conflicting_files,
            target_branch=target_branch,
            merge_commit_oid=merge_commit_oid,
            diff=discussion_details.get("diff"),
        )

    @validate_hf_hub_args
    def create_discussion(
        self,
        repo_id: str,
        title: str,
        *,
        token: Optional[str] = None,
        description: Optional[str] = None,
        repo_type: Optional[str] = None,
        pull_request: bool = False,
    ) -> DiscussionWithDetails:
        """Creates a Discussion or Pull Request.

        Pull Requests created programmatically will be in `"draft"` status.

        Creating a Pull Request with changes can also be done at once with [`HfApi.create_commit`].

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            title (`str`):
                The title of the discussion. It can be up to 200 characters long,
                and must be at least 3 characters long. Leading and trailing whitespaces
                will be stripped.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            description (`str`, *optional*):
                An optional description for the Pull Request.
                Defaults to `"Discussion opened with the huggingface_hub Python library"`
            pull_request (`bool`, *optional*):
                Whether to create a Pull Request or discussion. If `True`, creates a Pull Request.
                If `False`, creates a discussion. Defaults to `False`.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.

        Returns: [`DiscussionWithDetails`]

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>"""
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL

        if description is not None:
            description = description.strip()
        description = (
            description
            if description
            else (
                f"{'Pull Request' if pull_request else 'Discussion'} opened with the"
                " [huggingface_hub Python"
                " library](https://huggingface.co/docs/huggingface_hub)"
            )
        )

        headers = self._build_hf_headers(token=token, is_write_action=True)
        resp = requests.post(
            f"{self.endpoint}/api/{repo_type}s/{repo_id}/discussions",
            json={
                "title": title.strip(),
                "description": description,
                "pullRequest": pull_request,
            },
            headers=headers,
        )
        hf_raise_for_status(resp)
        num = resp.json()["num"]
        return self.get_discussion_details(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=num,
            token=token,
        )

    @validate_hf_hub_args
    def create_pull_request(
        self,
        repo_id: str,
        title: str,
        *,
        token: Optional[str] = None,
        description: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionWithDetails:
        """Creates a Pull Request . Pull Requests created programmatically will be in `"draft"` status.

        Creating a Pull Request with changes can also be done at once with [`HfApi.create_commit`];

        This is a wrapper around [`HfApi.create_discusssion`].

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            title (`str`):
                The title of the discussion. It can be up to 200 characters long,
                and must be at least 3 characters long. Leading and trailing whitespaces
                will be stripped.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            description (`str`, *optional*):
                An optional description for the Pull Request.
                Defaults to `"Discussion opened with the huggingface_hub Python library"`
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.

        Returns: [`DiscussionWithDetails`]

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>"""
        return self.create_discussion(
            repo_id=repo_id,
            title=title,
            token=token,
            description=description,
            repo_type=repo_type,
            pull_request=True,
        )

    def _post_discussion_changes(
        self,
        *,
        repo_id: str,
        discussion_num: int,
        resource: str,
        body: Optional[dict] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> requests.Response:
        """Internal utility to POST changes to a Discussion or Pull Request"""
        if not isinstance(discussion_num, int) or discussion_num <= 0:
            raise ValueError("Invalid discussion_num, must be a positive integer")
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL
        repo_id = f"{repo_type}s/{repo_id}"

        path = f"{self.endpoint}/api/{repo_id}/discussions/{discussion_num}/{resource}"

        headers = self._build_hf_headers(token=token, is_write_action=True)
        resp = requests.post(path, headers=headers, json=body)
        hf_raise_for_status(resp)
        return resp

    @validate_hf_hub_args
    def comment_discussion(
        self,
        repo_id: str,
        discussion_num: int,
        comment: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionComment:
        """Creates a new comment on the given Discussion.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            comment (`str`):
                The content of the comment to create. Comments support markdown formatting.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionComment`]: the newly created comment


        Examples:
            ```python

            >>> comment = \"\"\"
            ... Hello @otheruser!
            ...
            ... # This is a title
            ...
            ... **This is bold**, *this is italic* and ~this is strikethrough~
            ... And [this](http://url) is a link
            ... \"\"\"

            >>> HfApi().comment_discussion(
            ...     repo_id="username/repo_name",
            ...     discussion_num=34
            ...     comment=comment
            ... )
            # DiscussionComment(id='deadbeef0000000', type='comment', ...)

            ```

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        resp = self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource="comment",
            body={"comment": comment},
        )
        return deserialize_event(resp.json()["newMessage"])  # type: ignore

    @validate_hf_hub_args
    def rename_discussion(
        self,
        repo_id: str,
        discussion_num: int,
        new_title: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionTitleChange:
        """Renames a Discussion.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            new_title (`str`):
                The new title for the discussion
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionTitleChange`]: the title change event


        Examples:
            ```python
            >>> new_title = "New title, fixing a typo"
            >>> HfApi().rename_discussion(
            ...     repo_id="username/repo_name",
            ...     discussion_num=34
            ...     new_title=new_title
            ... )
            # DiscussionTitleChange(id='deadbeef0000000', type='title-change', ...)

            ```

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        resp = self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource="title",
            body={"title": new_title},
        )
        return deserialize_event(resp.json()["newTitle"])  # type: ignore

    @validate_hf_hub_args
    def change_discussion_status(
        self,
        repo_id: str,
        discussion_num: int,
        new_status: Literal["open", "closed"],
        *,
        token: Optional[str] = None,
        comment: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionStatusChange:
        """Closes or re-opens a Discussion or Pull Request.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            new_status (`str`):
                The new status for the discussion, either `"open"` or `"closed"`.
            comment (`str`, *optional*):
                An optional comment to post with the status change.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionStatusChange`]: the status change event


        Examples:
            ```python
            >>> new_title = "New title, fixing a typo"
            >>> HfApi().rename_discussion(
            ...     repo_id="username/repo_name",
            ...     discussion_num=34
            ...     new_title=new_title
            ... )
            # DiscussionStatusChange(id='deadbeef0000000', type='status-change', ...)

            ```

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        if new_status not in ["open", "closed"]:
            raise ValueError("Invalid status, valid statuses are: 'open' and 'closed'")
        body: Dict[str, str] = {"status": new_status}
        if comment and comment.strip():
            body["comment"] = comment.strip()
        resp = self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource="status",
            body=body,
        )
        return deserialize_event(resp.json()["newStatus"])  # type: ignore

    @validate_hf_hub_args
    def merge_pull_request(
        self,
        repo_id: str,
        discussion_num: int,
        *,
        token: Optional[str] = None,
        comment: Optional[str] = None,
        repo_type: Optional[str] = None,
    ):
        """Merges a Pull Request.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            comment (`str`, *optional*):
                An optional comment to post with the status change.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionStatusChange`]: the status change event

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource="merge",
            body={"comment": comment.strip()} if comment and comment.strip() else None,
        )

    @validate_hf_hub_args
    def edit_discussion_comment(
        self,
        repo_id: str,
        discussion_num: int,
        comment_id: str,
        new_content: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionComment:
        """Edits a comment on a Discussion / Pull Request.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            comment_id (`str`):
                The ID of the comment to edit.
            new_content (`str`):
                The new content of the comment. Comments support markdown formatting.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionComment`]: the edited comment

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        resp = self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource=f"comment/{comment_id.lower()}/edit",
            body={"content": new_content},
        )
        return deserialize_event(resp.json()["updatedComment"])  # type: ignore

    @validate_hf_hub_args
    def hide_discussion_comment(
        self,
        repo_id: str,
        discussion_num: int,
        comment_id: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionComment:
        """Hides a comment on a Discussion / Pull Request.

        <Tip warning={true}>
        Hidden comments' content cannot be retrieved anymore. Hiding a comment is irreversible.
        </Tip>

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            comment_id (`str`):
                The ID of the comment to edit.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionComment`]: the hidden comment

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        warnings.warn(
            "Hidden comments' content cannot be retrieved anymore. Hiding a comment is"
            " irreversible.",
            UserWarning,
        )
        resp = self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource=f"comment/{comment_id.lower()}/hide",
        )
        return deserialize_event(resp.json()["updatedComment"])  # type: ignore

    def _build_hf_headers(
        self,
        token: Optional[Union[bool, str]] = None,
        is_write_action: bool = False,
        library_name: Optional[str] = None,
        library_version: Optional[str] = None,
        user_agent: Union[Dict, str, None] = None,
    ) -> Dict[str, str]:
        """
        Alias for [`build_hf_headers`] that uses the token from [`HfApi`] client
        when `token` is not provided.
        """
        if token is None:
            token = self.token
        return build_hf_headers(
            token=token,
            is_write_action=is_write_action,
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
        )


def _prepare_upload_folder_commit(
    folder_path: Union[str, Path],
    path_in_repo: str,
    allow_patterns: Optional[Union[List[str], str]] = None,
    ignore_patterns: Optional[Union[List[str], str]] = None,
) -> List[CommitOperationAdd]:
    """Generate the list of Add operations for a commit to upload a folder.

    Files not matching the `allow_patterns` (allowlist) and `ignore_patterns` (denylist)
    constraints are discarded.
    """
    folder_path = os.path.normpath(os.path.expanduser(folder_path))
    if not os.path.isdir(folder_path):
        raise ValueError(f"Provided path: '{folder_path}' is not a directory")

    files_to_add: List[CommitOperationAdd] = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, folder_path)
            files_to_add.append(
                CommitOperationAdd(
                    path_or_fileobj=abs_path,
                    path_in_repo=os.path.normpath(
                        os.path.join(path_in_repo, rel_path)
                    ).replace(os.sep, "/"),
                )
            )

    return list(
        filter_repo_objects(
            files_to_add,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            key=lambda x: x.path_in_repo,
        )
    )


def _parse_revision_from_pr_url(pr_url: str) -> str:
    """Safely parse revision number from a PR url.

    Example:
    ```py
    >>> _parse_revision_from_pr_url("https://huggingface.co/bigscience/bloom/discussions/2")
    "refs/pr/2"
    ```
    """
    re_match = re.match(_REGEX_DISCUSSION_URL, pr_url)
    if re_match is None:
        raise RuntimeError(
            "Unexpected response from the hub, expected a Pull Request URL but got:"
            f" '{pr_url}'"
        )
    return f"refs/pr/{re_match[1]}"


api = HfApi()

set_access_token = api.set_access_token
unset_access_token = api.unset_access_token

whoami = api.whoami

list_models = api.list_models
model_info = api.model_info

list_datasets = api.list_datasets
dataset_info = api.dataset_info

list_spaces = api.list_spaces
space_info = api.space_info

repo_info = api.repo_info
list_repo_files = api.list_repo_files

list_metrics = api.list_metrics

get_model_tags = api.get_model_tags
get_dataset_tags = api.get_dataset_tags

create_commit = api.create_commit
create_repo = api.create_repo
delete_repo = api.delete_repo
update_repo_visibility = api.update_repo_visibility
move_repo = api.move_repo
upload_file = api.upload_file
upload_folder = api.upload_folder
delete_file = api.delete_file
delete_folder = api.delete_folder
create_branch = api.create_branch
delete_branch = api.delete_branch
create_tag = api.create_tag
delete_tag = api.delete_tag
get_full_repo_name = api.get_full_repo_name

get_discussion_details = api.get_discussion_details
get_repo_discussions = api.get_repo_discussions
create_discussion = api.create_discussion
create_pull_request = api.create_pull_request
change_discussion_status = api.change_discussion_status
comment_discussion = api.comment_discussion
edit_discussion_comment = api.edit_discussion_comment
rename_discussion = api.rename_discussion
merge_pull_request = api.merge_pull_request
