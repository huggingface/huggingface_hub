# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import shutil
import unittest
from pathlib import Path

from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.repocard import metadata_load, metadata_save

from .testing_utils import set_write_permission_and_retry


DUMMY_MODELCARD = """

Hi

---
license: mit
datasets:
- foo
- bar
---

Hello
"""

DUMMY_MODELCARD_TARGET = """

Hi

---
meaning_of_life: 42
---

Hello
"""

DUMMY_MODELCARD_TARGET_NO_YAML = """---
meaning_of_life: 42
---
Hello
"""

DUMMY_MODELCARD_TARGET_NO_TAGS = """
Hello
"""

REPOCARD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures/repocard"
)


class RepocardTest(unittest.TestCase):
    def setUp(self):
        os.makedirs(REPOCARD_DIR, exist_ok=True)

    def tearDown(self) -> None:
        try:
            shutil.rmtree(REPOCARD_DIR, onerror=set_write_permission_and_retry)
        except FileNotFoundError:
            pass

    def test_metadata_load(self):
        filepath = Path(REPOCARD_DIR) / REPOCARD_NAME
        filepath.write_text(DUMMY_MODELCARD)
        data = metadata_load(filepath)
        self.assertDictEqual(data, {"license": "mit", "datasets": ["foo", "bar"]})

    def test_metadata_save(self):
        filename = "dummy_target.md"
        filepath = Path(REPOCARD_DIR) / filename
        filepath.write_text(DUMMY_MODELCARD)
        metadata_save(filepath, {"meaning_of_life": 42})
        content = filepath.read_text()
        self.assertEqual(content, DUMMY_MODELCARD_TARGET)

    def test_metadata_save_from_file_no_yaml(self):
        filename = "dummy_target_2.md"
        filepath = Path(REPOCARD_DIR) / filename
        filepath.write_text("Hello\n")
        metadata_save(filepath, {"meaning_of_life": 42})
        content = filepath.read_text()
        self.assertEqual(content, DUMMY_MODELCARD_TARGET_NO_YAML)

    def test_no_metadata_returns_none(self):
        filename = "dummy_target_3.md"
        filepath = Path(REPOCARD_DIR) / filename
        filepath.write_text(DUMMY_MODELCARD_TARGET_NO_TAGS)
        data = metadata_load(filepath)
        self.assertEqual(data, None)
