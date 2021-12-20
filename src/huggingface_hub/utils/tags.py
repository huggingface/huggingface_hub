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
