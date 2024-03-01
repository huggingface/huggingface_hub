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
"""Contain tests for `HfFolder` utility."""

import os
import unittest
from uuid import uuid4

from huggingface_hub.utils import HfFolder


def _generate_token() -> str:
    return f"token-{uuid4()}"


class HfFolderTest(unittest.TestCase):
    def test_token_workflow(self):
        """
        Test the whole token save/get/delete workflow,
        with the desired behavior with respect to non-existent tokens.
        """
        token = _generate_token()
        HfFolder.save_token(token)
        self.assertEqual(HfFolder.get_token(), token)
        HfFolder.delete_token()
        HfFolder.delete_token()
        # ^^ not an error, we test that the
        # second call does not fail.
        self.assertEqual(HfFolder.get_token(), None)
        # test TOKEN in env
        self.assertEqual(HfFolder.get_token(), None)
        with unittest.mock.patch.dict(os.environ, {"HF_TOKEN": token}):
            self.assertEqual(HfFolder.get_token(), token)

    def test_token_strip(self):
        """
        Test the workflow when the token is mistakenly finishing with new-line or space character.
        """
        token = _generate_token()
        HfFolder.save_token(" " + token + "\n")
        self.assertEqual(HfFolder.get_token(), token)
        HfFolder.delete_token()
