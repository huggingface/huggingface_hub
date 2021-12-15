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
""" Tagging utilities. """


class AttributeDictionary(dict):
    "`dict` subclass that also provides access to keys as attrs"

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
        return super().__dir__() + list(self.keys())

    def __repr__(self):
        _ignore = [str(o) for o in dir(AttributeDictionary())]
        repr_str = "Available Attributes:\n"
        for o in dir(self):
            if (o not in _ignore) and not (o.startswith("_")):
                repr_str += f" * {o}\n"
        return repr_str


class GeneralTags(AttributeDictionary):
    "A namespace object holding all model tags, filtered by `keys`"

    def __init__(self, tag_dictionary: dict, keys: list = None):
        self._tag_dictionary = tag_dictionary
        if keys is None:
            keys = list(self._tag_dictionary.keys())
        for key in keys:
            self._unpack_and_assign_dictionary(key)

    def _unpack_and_assign_dictionary(self, key: str):
        "Assignes nested attr to `self.key` containing information as an `AttrDict`"
        setattr(self, key, AttributeDictionary())
        for item in self._tag_dictionary[key]:
            ref = getattr(self, key)
            item["label"] = item["label"].replace(" ", "").replace("-", "_")
            setattr(ref, item["label"], item["id"])


class ModelTags(GeneralTags):
    "A namespace object holding all available model tags"

    def __init__(self, model_tag_dictionary: dict):
        keys = ["library", "language", "licence", "dataset", "pipeline_tag"]
        super().__init__(model_tag_dictionary, keys)


class DatasetTags(GeneralTags):
    "A namespace object holding all available dataset tags"

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
