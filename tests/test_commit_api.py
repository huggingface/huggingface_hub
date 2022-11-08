import unittest

from huggingface_hub._commit_api import CommitOperationDelete


class TestCommitOperationDelete(unittest.TestCase):
    def test_implicit_file(self):
        self.assertFalse(CommitOperationDelete(path_in_repo="path/to/file").is_folder)
        self.assertFalse(
            CommitOperationDelete(path_in_repo="path/to/file.md").is_folder
        )

    def test_implicit_folder(self):
        self.assertTrue(CommitOperationDelete(path_in_repo="path/to/folder/").is_folder)
        self.assertTrue(
            CommitOperationDelete(path_in_repo="path/to/folder.md/").is_folder
        )

    def test_explicit_file(self):
        # Weird case: if user explicitly set as file (`is_folder`=False) but path has a
        # trailing "/" => user input has priority
        self.assertFalse(
            CommitOperationDelete(
                path_in_repo="path/to/folder/", is_folder=False
            ).is_folder
        )
        self.assertFalse(
            CommitOperationDelete(
                path_in_repo="path/to/folder.md/", is_folder=False
            ).is_folder
        )

    def test_explicit_folder(self):
        # No need for the trailing "/" is `is_folder` explicitly passed
        self.assertTrue(
            CommitOperationDelete(
                path_in_repo="path/to/folder", is_folder=True
            ).is_folder
        )
        self.assertTrue(
            CommitOperationDelete(
                path_in_repo="path/to/folder.md", is_folder=True
            ).is_folder
        )

    def test_is_folder_wrong_value(self):
        with self.assertRaises(ValueError):
            CommitOperationDelete(path_in_repo="path/to/folder", is_folder="any value")
