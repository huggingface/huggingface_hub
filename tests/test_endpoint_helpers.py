# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import unittest

import requests

from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils.endpoint_helpers import (
    AttributeDictionary,
    DatasetTags,
    GeneralTags,
    ModelTags,
)

from .testing_utils import with_production_testing


class AttributeDictionaryCommonTest(unittest.TestCase):
    _attrdict = AttributeDictionary()


class AttributeDictionaryTest(AttributeDictionaryCommonTest):
    def setUp(self):
        self._attrdict.clear()

    def test_adding_item(self):
        self._attrdict["itemA"] = 2
        self.assertEqual(self._attrdict.itemA, 2)
        self.assertEqual(self._attrdict["itemA"], 2)
        # We should be able to both set a property and a key
        self._attrdict.itemB = 3
        self.assertEqual(self._attrdict.itemB, 3)
        self.assertEqual(self._attrdict["itemB"], 3)

    def test_removing_item(self):
        self._attrdict["itemA"] = 2
        delattr(self._attrdict, "itemA")
        with self.assertRaises(KeyError):
            _ = self._attrdict["itemA"]

        self._attrdict.itemB = 3
        del self._attrdict["itemB"]
        with self.assertRaises(AttributeError):
            _ = self._attrdict.itemB

    def test_dir(self):
        # Since we subclass dict, dir should have everything
        # from dict and the attributes
        _dict_keys = dir(dict) + [
            "__dict__",
            "__getattr__",
            "__module__",
            "__weakref__",
        ]
        self._attrdict["itemA"] = 2
        self._attrdict.itemB = 3
        _dict_keys += ["itemA", "itemB"]
        _dict_keys.sort()

        full_dir = dir(self._attrdict)
        full_dir.sort()
        self.assertEqual(full_dir, _dict_keys)

    def test_dir_with_numbers(self):
        self._attrdict["1a"] = 4
        self.assertFalse("1a" in dir(self._attrdict))
        self.assertTrue("1a" in list(self._attrdict.keys()))

    def test_dir_with_special_characters(self):
        self._attrdict["1<2"] = 3
        self.assertFalse("1<2" in dir(self._attrdict))
        self.assertTrue("1<2" in list(self._attrdict.keys()))

        self._attrdict["?abc"] = 4
        self.assertFalse("?abc" in dir(self._attrdict))
        self.assertTrue("?abc" in list(self._attrdict.keys()))

    def test_repr(self):
        self._attrdict["itemA"] = 2
        self._attrdict.itemB = 3
        self._attrdict["1a"] = 2
        self._attrdict["itemA?"] = 4
        repr_string = "Available Attributes or Keys:\n * 1a (Key only)\n * itemA\n * itemA? (Key only)\n * itemB\n"
        self.assertEqual(repr_string, repr(self._attrdict))


class GeneralTagsCommonTest(unittest.TestCase):
    # Similar to the output from /api/***-tags-by-type
    # id = how we can search hfapi, such as `'id': 'language:en'`
    # label = A human readable version assigned to everything, such as `"label":"en"`
    _tag_dictionary = {
        "language": [
            {"id": "itemA", "label": "Item A"},
            {"id": "itemB", "label": "1Item-B"},
        ],
        "license": [
            {"id": "itemC", "label": "Item C"},
            {"id": "itemD", "label": "Item.D"},
        ],
    }


class GeneralTagsTest(GeneralTagsCommonTest):
    def test_init(self):
        _tags = GeneralTags(self._tag_dictionary)
        languages = _tags.language
        licenses = _tags.license
        # Ensure they have the right bits

        self.assertEqual(
            languages,
            AttributeDictionary({"ItemA": "itemA", "1Item_B": "itemB"}),
        )

        self.assertTrue("1Item_B" not in dir(languages))

        self.assertEqual(licenses, AttributeDictionary({"ItemC": "itemC", "Item_D": "itemD"}))

    def test_filter(self):
        _tags = GeneralTags(self._tag_dictionary, keys=["license"])
        self.assertTrue(hasattr(_tags, "license"))
        self.assertFalse(hasattr(_tags, "languages"))
        self.assertEqual(_tags.license, AttributeDictionary({"ItemC": "itemC", "Item_D": "itemD"}))


class ModelTagsTest(unittest.TestCase):
    @with_production_testing
    def test_tags(self):
        # ModelTags instantiation must not fail!
        res = requests.get(f"{HfApi().endpoint}/api/models-tags-by-type")
        res.raise_for_status()
        tags = ModelTags(res.json())

        # Check existing keys to get notified about server-side changes
        for existing_key in [
            "dataset",
            "language",
            "library",
            "license",
            "pipeline_tag",
        ]:
            self.assertGreater(len(getattr(tags, existing_key).keys()), 0)


class DatasetTagsTest(unittest.TestCase):
    @with_production_testing
    def test_tags(self):
        # DatasetTags instantiation must not fail!
        res = requests.get(f"{HfApi().endpoint}/api/datasets-tags-by-type")
        res.raise_for_status()
        tags = DatasetTags(res.json())

        # Some keys existed before but have been removed server-side
        for missing_key in (
            "language_creators",
            "multilinguality",
        ):
            self.assertEqual(len(getattr(tags, missing_key).keys()), 0)

        # Check existing keys to get notified about server-side changes
        for existing_key in [
            "benchmark",
            "language",
            "license",
            "size_categories",
            "task_categories",
            "task_ids",
        ]:
            self.assertGreater(len(getattr(tags, existing_key).keys()), 0)
