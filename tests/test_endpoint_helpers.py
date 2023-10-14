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


