import unittest

from huggingface_hub._commit_api import (
    CommitOperationAdd,
    CommitOperationDelete,
    warn_on_overwriting_operations,
)


class TestCommitOperationDelete(unittest.TestCase):
    def test_implicit_file(self):
        self.assertFalse(CommitOperationDelete(path_in_repo="path/to/file").is_folder)
        self.assertFalse(CommitOperationDelete(path_in_repo="path/to/file.md").is_folder)

    def test_implicit_folder(self):
        self.assertTrue(CommitOperationDelete(path_in_repo="path/to/folder/").is_folder)
        self.assertTrue(CommitOperationDelete(path_in_repo="path/to/folder.md/").is_folder)

    def test_explicit_file(self):
        # Weird case: if user explicitly set as file (`is_folder`=False) but path has a
        # trailing "/" => user input has priority
        self.assertFalse(CommitOperationDelete(path_in_repo="path/to/folder/", is_folder=False).is_folder)
        self.assertFalse(CommitOperationDelete(path_in_repo="path/to/folder.md/", is_folder=False).is_folder)

    def test_explicit_folder(self):
        # No need for the trailing "/" is `is_folder` explicitly passed
        self.assertTrue(CommitOperationDelete(path_in_repo="path/to/folder", is_folder=True).is_folder)
        self.assertTrue(CommitOperationDelete(path_in_repo="path/to/folder.md", is_folder=True).is_folder)

    def test_is_folder_wrong_value(self):
        with self.assertRaises(ValueError):
            CommitOperationDelete(path_in_repo="path/to/folder", is_folder="any value")


class TestCommitOperationAdd(unittest.TestCase):
    def test_path_in_repo_normal_path(self):
        self.assertEqual(
            CommitOperationAdd(path_in_repo="file.txt", path_or_fileobj=b"").path_in_repo,
            "file.txt",
        )

    def test_path_in_repo_prefix_stripped(self):
        self.assertEqual(
            CommitOperationAdd(path_in_repo="./file.txt", path_or_fileobj=b"").path_in_repo,
            "file.txt",
        )

    def test_path_in_repo_not_stripped_on_hidden_file(self):
        self.assertEqual(
            CommitOperationAdd(path_in_repo=".file.txt", path_or_fileobj=b"").path_in_repo,
            ".file.txt",
        )

    def test_path_in_repo_invalid_single_dot(self):
        with self.assertRaises(ValueError):
            CommitOperationAdd(path_in_repo=".", path_or_fileobj=b"")

    def test_path_in_repo_invalid_double_dot(self):
        with self.assertRaises(ValueError):
            CommitOperationAdd(path_in_repo="..", path_or_fileobj=b"")

    def test_path_in_repo_invalid_double_dot_slash_prefix(self):
        with self.assertRaises(ValueError):
            CommitOperationAdd(path_in_repo="../file.txt", path_or_fileobj=b"")


class TestWarnOnOverwritingOperations(unittest.TestCase):
    add_file_ab = CommitOperationAdd(path_in_repo="a/b.txt", path_or_fileobj=b"data")
    add_file_abc = CommitOperationAdd(path_in_repo="a/b/c.md", path_or_fileobj=b"data")
    add_file_abd = CommitOperationAdd(path_in_repo="a/b/d.md", path_or_fileobj=b"data")
    update_file_abc = CommitOperationAdd(path_in_repo="a/b/c.md", path_or_fileobj=b"updated data")
    delete_file_abc = CommitOperationDelete(path_in_repo="a/b/c.md")
    delete_folder_a = CommitOperationDelete(path_in_repo="a/")
    delete_folder_e = CommitOperationDelete(path_in_repo="e/")

    def test_no_overwrite(self) -> None:
        warn_on_overwriting_operations(
            [
                self.add_file_ab,
                self.add_file_abc,
                self.add_file_abd,
                self.delete_folder_e,
            ]
        )

    def test_add_then_update_file(self) -> None:
        with self.assertWarns(UserWarning):
            warn_on_overwriting_operations([self.add_file_abc, self.update_file_abc])

    def test_add_then_delete_file(self) -> None:
        with self.assertWarns(UserWarning):
            warn_on_overwriting_operations([self.add_file_abc, self.delete_file_abc])

    def test_add_then_delete_folder(self) -> None:
        with self.assertWarns(UserWarning):
            warn_on_overwriting_operations([self.add_file_abc, self.delete_folder_a])

        with self.assertWarns(UserWarning):
            warn_on_overwriting_operations([self.add_file_ab, self.delete_folder_a])

    def test_delete_file_then_add(self) -> None:
        warn_on_overwriting_operations([self.delete_file_abc, self.add_file_abc])

    def test_delete_folder_then_add(self) -> None:
        warn_on_overwriting_operations([self.delete_folder_a, self.add_file_ab, self.add_file_abc])
