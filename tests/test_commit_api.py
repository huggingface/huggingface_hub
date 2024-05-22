import unittest

from huggingface_hub._commit_api import (
    CommitOperationAdd,
    CommitOperationDelete,
    _warn_on_overwriting_operations,
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


class TestCommitOperationPathInRepo(unittest.TestCase):
    valid_values = {  # key is input, value is expected validated output
        "file.txt": "file.txt",
        ".file.txt": ".file.txt",
        "/file.txt": "file.txt",
        "./file.txt": "file.txt",
    }
    invalid_values = [".", "..", "../file.txt"]

    def test_path_in_repo_valid(self) -> None:
        for input, expected in self.valid_values.items():
            with self.subTest(f"Testing with valid input: '{input}'"):
                self.assertEqual(CommitOperationAdd(path_in_repo=input, path_or_fileobj=b"").path_in_repo, expected)
                self.assertEqual(CommitOperationDelete(path_in_repo=input).path_in_repo, expected)

    def test_path_in_repo_invalid(self) -> None:
        for input in self.invalid_values:
            with self.subTest(f"Testing with invalid input: '{input}'"):
                with self.assertRaises(ValueError):
                    CommitOperationAdd(path_in_repo=input, path_or_fileobj=b"")
                with self.assertRaises(ValueError):
                    CommitOperationDelete(path_in_repo=input)


class TestCommitOperationForbiddenPathInRepo(unittest.TestCase):
    """Commit operations must throw an error on files in the .git/ or .cache/huggingface/ folders.

    Server would error anyway so it's best to prevent early.
    """

    INVALID_PATHS_IN_REPO = {
        ".git",
        ".git/path/to/file",
        "./.git/path/to/file",
        "subfolder/path/.git/to/file",
        "./subfolder/path/.git/to/file",
        ".cache/huggingface",
        "./.cache/huggingface/path/to/file",
        "./subfolder/path/.cache/huggingface/to/file",
    }

    VALID_PATHS_IN_REPO = {
        ".gitignore",
        "path/to/.gitignore",
        "path/to/something.git",
        "path/to/something.git/more",
        "path/to/something.huggingface/more",
        "huggingface",
        ".huggingface",
        "./.huggingface/path/to/file",
        "./subfolder/path/huggingface/to/file",
        "./subfolder/path/.huggingface/to/file",
    }

    def test_cannot_update_file_in_git_folder(self):
        for path in self.INVALID_PATHS_IN_REPO:
            with self.subTest(msg=f"Add: '{path}'"):
                with self.assertRaises(ValueError):
                    CommitOperationAdd(path_in_repo=path, path_or_fileobj=b"content")
            with self.subTest(msg=f"Delete: '{path}'"):
                with self.assertRaises(ValueError):
                    CommitOperationDelete(path_in_repo=path)

    def test_valid_path_in_repo_containing_git(self):
        for path in self.VALID_PATHS_IN_REPO:
            with self.subTest(msg=f"Add: '{path}'"):
                CommitOperationAdd(path_in_repo=path, path_or_fileobj=b"content")
            with self.subTest(msg=f"Delete: '{path}'"):
                CommitOperationDelete(path_in_repo=path)


class TestWarnOnOverwritingOperations(unittest.TestCase):
    add_file_ab = CommitOperationAdd(path_in_repo="a/b.txt", path_or_fileobj=b"data")
    add_file_abc = CommitOperationAdd(path_in_repo="a/b/c.md", path_or_fileobj=b"data")
    add_file_abd = CommitOperationAdd(path_in_repo="a/b/d.md", path_or_fileobj=b"data")
    update_file_abc = CommitOperationAdd(path_in_repo="a/b/c.md", path_or_fileobj=b"updated data")
    delete_file_abc = CommitOperationDelete(path_in_repo="a/b/c.md")
    delete_folder_a = CommitOperationDelete(path_in_repo="a/")
    delete_folder_e = CommitOperationDelete(path_in_repo="e/")

    def test_no_overwrite(self) -> None:
        _warn_on_overwriting_operations(
            [
                self.add_file_ab,
                self.add_file_abc,
                self.add_file_abd,
                self.delete_folder_e,
            ]
        )

    def test_add_then_update_file(self) -> None:
        with self.assertWarns(UserWarning):
            _warn_on_overwriting_operations([self.add_file_abc, self.update_file_abc])

    def test_add_then_delete_file(self) -> None:
        with self.assertWarns(UserWarning):
            _warn_on_overwriting_operations([self.add_file_abc, self.delete_file_abc])

    def test_add_then_delete_folder(self) -> None:
        with self.assertWarns(UserWarning):
            _warn_on_overwriting_operations([self.add_file_abc, self.delete_folder_a])

        with self.assertWarns(UserWarning):
            _warn_on_overwriting_operations([self.add_file_ab, self.delete_folder_a])

    def test_delete_file_then_add(self) -> None:
        _warn_on_overwriting_operations([self.delete_file_abc, self.add_file_abc])

    def test_delete_folder_then_add(self) -> None:
        _warn_on_overwriting_operations([self.delete_folder_a, self.add_file_ab, self.add_file_abc])
