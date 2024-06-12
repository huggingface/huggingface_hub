"""Contains tests that are specific to windows machines."""

import os
import unittest

from huggingface_hub.file_download import are_symlinks_supported


def require_windows(test_case):
    if os.name != "nt":
        return unittest.skip("test of git lfs workflow")(test_case)
    else:
        return test_case


@require_windows
class WindowsTests(unittest.TestCase):
    def test_are_symlink_supported(self) -> None:
        self.assertFalse(are_symlinks_supported())
