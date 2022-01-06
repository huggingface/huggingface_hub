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

from dataclasses import dataclass
from typing import List, Union


@dataclass
class DatasetFilter:
    """A class that converts human-readable dataset search parameters into ones compatible with
    the REST API. For all parameters capitalization does not matter.

    Args:
        author (:obj:`str`, `optional`):
            A string or list of strings that can be used to identify datasets on the Hub
            by the original uploader (author or organization), such as `facebook` or `huggingface`
            Example usage:

                >>> from huggingface_hub import DatasetFilter
                >>> new_filter = DatasetFilter(author="facebook")

       benchmark (:obj:`str` or :class:`List`, `optional`):
            A string or list of strings that can be used to identify datasets on the Hub by their official benchmark
            Example usage:

                >>> from huggingface_hub import DatasetFilter
                >>> new_filter = DatasetFilter(benchmark="raft")

        dataset_name (:obj:`str`, `optional`):
            A string or list of strings that can be used to identify datasets on the Hub by its name,
            such as `SQAC` or `wikineural`
            Example usage:

                >>> from huggingface_hub import DatasetFilter
                >>> new_filter = DatasetFilter(dataset_name="wikineural")

        language_creators (:obj:`str` or :class:`List`, `optional`):
            A string or list of strings that can be used to identify datasets on the Hub
            with how the data was curated, such as `crowdsourced` or `machine_generated`
            Example usage:

                >>> from huggingface_hub import DatasetFilter
                >>> new_filter = DatasetFilter(language_creator="crowdsourced")

        languages (:obj:`str` or :class:`List`, `optional`):
            A string or list of strings representing a two-character language to filter datasets by on the Hub
            Example usage:

                >>> from huggingface_hub import DatasetFilter
                >>> new_filter = DatasetFilter(language="en")

        multilinguality (:obj:`str` or :class:`List`, `optional`):
            A string or list of strings representing a filter for datasets that contain multiple languages
            Example usage:

                >>> from huggingface_hub import DatasetFilter
                >>> new_filter = DatasetFilter(multilinguality="yes")

        size_categories (:obj:`str` or :class:`List`, `optional`):
            A string or list of strings that can be used to identify datasets on the Hub
            by the size of the dataset such as `100K<n<1M` or `1M<n<10M`
            Example usage:

                >>> from huggingface_hub import DatasetFilter
                >>> new_filter = DatasetFilter(size_categories="100K<n<1M")

        task_categories (:obj:`str` or :class:`List`, `optional`):
            A string or list of strings that can be used to identify datasets on the Hub
            by the designed task, such as `audio_classification` or `named_entity_recognition`
            Example usage:

                >>> from huggingface_hub import DatasetFilter
                >>> new_filter = DatasetFilter(author="facebook")

        task_ids (:obj:`str` or :class:`List`, `optional`):
            A string or list of strings that can be used to identify datasets on the Hub
            by the specific task such as `speech_emotion_recognition` or `paraphrase`
            Example usage:

                >>> from huggingface_hub import DatasetFilter
                >>> new_filter = DatasetFilter(task_ids="paraphrase")
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
    """A class that converts human-readable model search parameters into ones compatible with
    the REST API. For all parameters capitalization does not matter.

    Args:
        author (:obj:`str`, `optional`):
            A string that can be used to identify models on the Hub
            by the original uploader (author or organization), such as `facebook` or `huggingface`
            Example usage:

                >>> from huggingface_hub import Filter
                >>> new_filter = ModelFilter(author_or_organization="facebook")

         library (:obj:`str` or :class:`List`, `optional`):
            A string or list of strings of foundational libraries models were originally trained from,
            such as pytorch, tensorflow, or allennlp
            Example usage:

                >>> new_filter = ModelFilter(library="pytorch")

         language (:obj:`str` or :class:`List`, `optional`):
            A string or list of strings of languages, both by name
            and country code, such as "en" or "English"
            Example usage:

                >>> new_filter = ModelFilter(language="french")

         model_name (:obj:`str`, `optional`):
            A string that contain complete or partial names for models on the Hub,
            such as "bert" or "bert-base-cased"
            Example usage:

                >>> new_filter = ModelFilter(model_name="bert")


         task (:obj:`str` or :class:`List`, `optional`):
            A string or list of strings of tasks models were designed for,
            such as: "fill-mask" or "automatic-speech-recognition"
            Example usage:

                >>> new_filter = ModelFilter(task="text-classification")

         tags (:obj:`str` or :class:`List`, `optional`):
            A string tag or a list of tags to filter models on the Hub by,
            such as `text-generation` or `spacy`. For a full list of tags do:
                >>> from huggingface_hub import HfApi
                >>> api = HfApi()
                # To list model tags
                >>> api.get_model_tags()
                # To list dataset tags
                >>> api.get_dataset_tags()

            Example usage:
                >>> new_filter = ModelFilter(tags="benchmark:raft")

        trained_dataset (:obj:`str` or :class:`List`, `optional`):
            A string tag or a list of string tags of the trained dataset for a model on the Hub.
            Example usage:
                >>> new_filter = ModelFilter(trained_dataset="common_voice")

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

    If a key starts with a number, it will exist in the dictionary
    but not as an attribute

    Example usage:

        >>> d = AttributeDictionary()
        >>> d["test"] = "a"
        >>> print(d.test) # prints "a"

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
    A namespace object holding all tags, filtered by `keys`
    If a tag starts with a number, it will only exist in the dictionary

    Example
        >>> a.b.1a # will not work
        >>> a.b["1a"] # will work
        >>> a["b"]["1a"] # will work

    Args:
        tag_dictionary (``dict``):
            A dictionary of tags returned from the /api/***-tags-by-type api endpoint
        keys (``list``):
            A list of keys to unpack the `tag_dictionary` with, such as `["library","language"]`
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
    A namespace object holding all available model tags
    If a tag starts with a number, it will only exist in the dictionary

    Example
        >>> o.dataset.1_5BArabicCorpus # will not work
        >>> a.dataset["1_5BArabicCorpus"] # will work
        >>> a["dataset"]["1_5BArabicCorpus"] # will work

    Args:
        model_tag_dictionary (``dict``):
            A dictionary of valid model tags, returned from the /api/models-tags-by-type api endpoint
    """

    def __init__(self, model_tag_dictionary: dict):
        keys = ["library", "language", "license", "dataset", "pipeline_tag"]
        super().__init__(model_tag_dictionary, keys)


class DatasetTags(GeneralTags):
    """
    A namespace object holding all available dataset tags
    If a tag starts with a number, it will only exist in the dictionary

    Example
        >>> o.size_categories.100K<n<1M # will not work
        >>> a.size_categories["100K<n<1M"] # will work
        >>> a["size_categories"]["100K<n<1M"] # will work

    Args:
        dataset_tag_dictionary (``dict``):
            A dictionary of valid dataset tags, returned from the /api/datasets-tags-by-type api endpoint
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
