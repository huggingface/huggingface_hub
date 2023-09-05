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
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from huggingface_hub.utils import HfFolder, SoftTemporaryDirectory


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
        with unittest.mock.patch.dict(os.environ, {"HUGGING_FACE_HUB_TOKEN": token}):
            self.assertEqual(HfFolder.get_token(), token)

    def test_token_in_old_path(self):
        token = _generate_token()
        token2 = _generate_token()
        with SoftTemporaryDirectory() as tmpdir:
            path_token = Path(tmpdir) / "new_token_path"
            old_path_token = Path(tmpdir) / "old_path_token"

            # Use dummy paths
            new_patcher = patch.object(HfFolder, "path_token", path_token)
            old_patcher = patch.object(HfFolder, "_old_path_token", old_path_token)
            new_patcher.start()
            old_patcher.start()

            # Reads from old path -> works but warn
            old_path_token.write_text(token)
            with self.assertWarns(UserWarning):
                self.assertEqual(HfFolder.get_token(), token)
            # Old path still exists
            self.assertEqual(old_path_token.read_text(), token)
            # New path is created
            self.assertEqual(path_token.read_text(), token)

            # Delete -> works, doesn't warn, delete both paths
            HfFolder.delete_token()
            self.assertFalse(old_path_token.exists())
            self.assertFalse(path_token.exists())

            # Write -> only to new path
            HfFolder.save_token(token)
            self.assertFalse(old_path_token.exists())
            self.assertEqual(path_token.read_text(), token)

            # Read -> new path has priority. No warning message.
            old_path_token.write_text(token2)
            self.assertEqual(HfFolder.get_token(), token)

            # Un-patch
            new_patcher.stop()
            old_patcher.stop()

    def test_token_strip(self):
        """
        Test the workflow when the token is mistakenly finishing with new-line or space character.
        """
        token = _generate_token()
        HfFolder.save_token(" " + token + "\n")
        self.assertEqual(HfFolder.get_token(), token)
        HfFolder.delete_token()

