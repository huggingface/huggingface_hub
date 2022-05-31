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
"""
Helpful utility functions and classes in relation to exploring API endpoints
with the aim for a user-friendly interface
"""

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Union

from ..constants import (
    DEFAULT_REVISION,
    HUGGINGFACE_CO_URL_TEMPLATE,
    REPO_TYPES,
    REPO_TYPES_URL_PREFIXES,
)
from ._deprecation import _deprecate_positional_args


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


def _filter_emissions(
    models,
    minimum_threshold: float = None,
    maximum_threshold: float = None,
):
    """Filters a list of models for those that include an emission tag
    and limit them to between two thresholds

    Args:
        models (`ModelInfo` or `List`):
            A list of `ModelInfo`'s to filter by.
        minimum_threshold (`float`, *optional*):
            A minimum carbon threshold to filter by, such as 1.
        maximum_threshold (`float`, *optional*):
            A maximum carbon threshold to filter by, such as 10.
    """
    if minimum_threshold is None and maximum_threshold is None:
        raise ValueError(
            "Both `minimum_threshold` and `maximum_threshold` cannot both be `None`"
        )
    if minimum_threshold is None:
        minimum_threshold = -1
    if maximum_threshold is None:
        maximum_threshold = math.inf
    emissions = []
    for i, model in enumerate(models):
        if hasattr(model, "cardData"):
            if isinstance(model.cardData, dict):
                emission = model.cardData.get("co2_eq_emissions", None)
                if isinstance(emission, dict):
                    emission = emission["emissions"]
                if emission:
                    emission = str(emission)
                    if any(char.isdigit() for char in emission):
                        emission = re.search("\d+\.\d+|\d+", emission).group(0)
                        emissions.append((i, float(emission)))
    filtered_results = []
    for idx, emission in emissions:
        if emission >= minimum_threshold and emission <= maximum_threshold:
            filtered_results.append(models[idx])
    return filtered_results


@dataclass
class DatasetFilter:
    """
    A class that converts human-readable dataset search parameters into ones
    compatible with the REST API. For all parameters capitalization does not
    matter.

    Args:
        author (`str`, *optional*):
            A string or list of strings that can be used to identify datasets on
            the Hub by the original uploader (author or organization), such as
            `facebook` or `huggingface`.
        benchmark (`str` or `List`, *optional*):
            A string or list of strings that can be used to identify datasets on
            the Hub by their official benchmark.
        dataset_name (`str`, *optional*):
            A string or list of strings that can be used to identify datasets on
            the Hub by its name, such as `SQAC` or `wikineural`
        language_creators (`str` or `List`, *optional*):
            A string or list of strings that can be used to identify datasets on
            the Hub with how the data was curated, such as `crowdsourced` or
            `machine_generated`.
        languages (`str` or `List`, *optional*):
            A string or list of strings representing a two-character language to
            filter datasets by on the Hub.
        multilinguality (`str` or `List`, *optional*):
            A string or list of strings representing a filter for datasets that
            contain multiple languages.
        size_categories (`str` or `List`, *optional*):
            A string or list of strings that can be used to identify datasets on
            the Hub by the size of the dataset such as `100K<n<1M` or
            `1M<n<10M`.
        task_categories (`str` or `List`, *optional*):
            A string or list of strings that can be used to identify datasets on
            the Hub by the designed task, such as `audio_classification` or
            `named_entity_recognition`.
        task_ids (`str` or `List`, *optional*):
            A string or list of strings that can be used to identify datasets on
            the Hub by the specific task such as `speech_emotion_recognition` or
            `paraphrase`.

    Examples:

    ```py
    >>> from huggingface_hub import DatasetFilter

    >>> # Using author
    >>> new_filter = DatasetFilter(author="facebook")

    >>> # Using benchmark
    >>> new_filter = DatasetFilter(benchmark="raft")

    >>> # Using dataset_name
    >>> new_filter = DatasetFilter(dataset_name="wikineural")

    >>> # Using language_creator
    >>> new_filter = DatasetFilter(language_creator="crowdsourced")

    >>> # Using language
    >>> new_filter = DatasetFilter(language="en")

    >>> # Using multilinguality
    >>> new_filter = DatasetFilter(multilinguality="yes")

    >>> # Using size_categories
    >>> new_filter = DatasetFilter(size_categories="100K<n<1M")

    >>> # Using task_categories
    >>> new_filter = DatasetFilter(task_categories="audio_classification")

    >>> # Using task_ids
    >>> new_filter = DatasetFilter(task_ids="paraphrase")
    ```
    """

    author: str = None
    benchmark: Union[str, List[str]] = None
    dataset_name: str = None
    language_creators: Union[str, List[str]] = None
    languages: Union[str, List[str]] = None
    multilinguality: Union[str, List[str]] = None
    size_categories: Union[str, List[str]] = None
    task_categories: Union[str, List[str]] = None
    task_ids: Union[str, List[str]] = None


@dataclass
class ModelFilter:
    """
    A class that converts human-readable model search parameters into ones
    compatible with the REST API. For all parameters capitalization does not
    matter.

    Args:
        author (`str`, *optional*):
            A string that can be used to identify models on the Hub by the
            original uploader (author or organization), such as `facebook` or
            `huggingface`.
        library (`str` or `List`, *optional*):
            A string or list of strings of foundational libraries models were
            originally trained from, such as pytorch, tensorflow, or allennlp.
        language (`str` or `List`, *optional*):
            A string or list of strings of languages, both by name and country
            code, such as "en" or "English"
        model_name (`str`, *optional*):
            A string that contain complete or partial names for models on the
            Hub, such as "bert" or "bert-base-cased"
        task (`str` or `List`, *optional*):
            A string or list of strings of tasks models were designed for, such
            as: "fill-mask" or "automatic-speech-recognition"
        tags (`str` or `List`, *optional*):
            A string tag or a list of tags to filter models on the Hub by, such
            as `text-generation` or `spacy`.
        trained_dataset (`str` or `List`, *optional*):
            A string tag or a list of string tags of the trained dataset for a
            model on the Hub.


    ```python
    >>> from huggingface_hub import ModelFilter

    >>> # For the author_or_organization
    >>> new_filter = ModelFilter(author_or_organization="facebook")

    >>> # For the library
    >>> new_filter = ModelFilter(library="pytorch")

    >>> # For the language
    >>> new_filter = ModelFilter(language="french")

    >>> # For the model_name
    >>> new_filter = ModelFilter(model_name="bert")

    >>> # For the task
    >>> new_filter = ModelFilter(task="text-classification")

    >>> # Retrieving tags using the `HfApi.get_model_tags` method
    >>> from huggingface_hub import HfApi

    >>> api = HfApi()
    # To list model tags

    >>> api.get_model_tags()
    # To list dataset tags

    >>> api.get_dataset_tags()
    >>> new_filter = ModelFilter(tags="benchmark:raft")

    >>> # Related to the dataset
    >>> new_filter = ModelFilter(trained_dataset="common_voice")
    ```
    """

    author: str = None
    library: Union[str, List[str]] = None
    language: Union[str, List[str]] = None
    model_name: str = None
    task: Union[str, List[str]] = None
    trained_dataset: Union[str, List[str]] = None
    tags: Union[str, List[str]] = None


class AttributeDictionary(dict):
    """
    `dict` subclass that also provides access to keys as attributes

    If a key starts with a number, it will exist in the dictionary but not as an
    attribute

    Example usage:

    ```python
    >>> d = AttributeDictionary()
    >>> d["test"] = "a"
    >>> print(d.test)  # prints "a"
    ```

    """

    def __getattr__(self, k):
        if k in self:
            return self[k]
        else:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        (self.__setitem__, super().__setattr__)[k[0] == "_"](k, v)

    def __delattr__(self, k):
        if k in self:
            del self[k]
        else:
            raise AttributeError(k)

    def __dir__(self):
        keys = sorted(self.keys())
        keys = [key for key in keys if key.replace("_", "").isalpha()]
        return super().__dir__() + keys

    def __repr__(self):
        repr_str = "Available Attributes or Keys:\n"
        for key in sorted(self.keys()):
            repr_str += f" * {key}"
            if not key.replace("_", "").isalpha():
                repr_str += " (Key only)"
            repr_str += "\n"
        return repr_str


class GeneralTags(AttributeDictionary):
    """
    A namespace object holding all tags, filtered by `keys` If a tag starts with
    a number, it will only exist in the dictionary

    Example usage:
    ```python
    >>> a.b["1a"]  # will work
    >>> a["b"]["1a"]  # will work
    >>> # a.b.1a # will not work
    ```

    Args:
        tag_dictionary (`dict`):
            A dictionary of tags returned from the /api/***-tags-by-type api
            endpoint
        keys (`list`):
            A list of keys to unpack the `tag_dictionary` with, such as
            `["library","language"]`
    """

    def __init__(self, tag_dictionary: dict, keys: list = None):
        self._tag_dictionary = tag_dictionary
        if keys is None:
            keys = list(self._tag_dictionary.keys())
        for key in keys:
            self._unpack_and_assign_dictionary(key)

    def _unpack_and_assign_dictionary(self, key: str):
        "Assignes nested attributes to `self.key` containing information as an `AttributeDictionary`"
        setattr(self, key, AttributeDictionary())
        for item in self._tag_dictionary[key]:
            ref = getattr(self, key)
            item["label"] = (
                item["label"].replace(" ", "").replace("-", "_").replace(".", "_")
            )
            setattr(ref, item["label"], item["id"])


class ModelTags(GeneralTags):
    """
    A namespace object holding all available model tags If a tag starts with a
    number, it will only exist in the dictionary

    Example usage:

    ```python
    >>> a.dataset["1_5BArabicCorpus"]  # will work
    >>> a["dataset"]["1_5BArabicCorpus"]  # will work
    >>> # o.dataset.1_5BArabicCorpus # will not work
    ```

    Args:
        model_tag_dictionary (`dict`):
            A dictionary of valid model tags, returned from the
            /api/models-tags-by-type api endpoint
    """

    def __init__(self, model_tag_dictionary: dict):
        keys = ["library", "language", "license", "dataset", "pipeline_tag"]
        super().__init__(model_tag_dictionary, keys)


class DatasetTags(GeneralTags):
    """
    A namespace object holding all available dataset tags If a tag starts with a
    number, it will only exist in the dictionary

    Example

    ```python
    >>> a.size_categories["100K<n<1M"]  # will work
    >>> a["size_categories"]["100K<n<1M"]  # will work
    >>> # o.size_categories.100K<n<1M # will not work
    ```

    Args:
        dataset_tag_dictionary (`dict`):
            A dictionary of valid dataset tags, returned from the
            /api/datasets-tags-by-type api endpoint
    """

    def __init__(self, dataset_tag_dictionary: dict):
        keys = [
            "languages",
            "multilinguality",
            "language_creators",
            "task_categories",
            "size_categories",
            "benchmark",
            "task_ids",
            "licenses",
        ]
        super().__init__(dataset_tag_dictionary, keys)
