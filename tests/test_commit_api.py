import unittest

import pytest

from huggingface_hub._commit_api import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    _warn_on_overwriting_operations,
)
from huggingface_hub.hf_api import _resolve_copy_target_path


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


class TestCommitOperationCopy:
    def test_cross_repo_copy_missing_repo_id_or_type(self):
        with pytest.raises(ValueError, match="`src_repo_type` is required when `src_repo_id` is set"):
            CommitOperationCopy(src_path_in_repo="src.bin", path_in_repo="dst.bin", src_repo_id="user/source")

        with pytest.raises(ValueError, match="`src_repo_id` is required when `src_repo_type` is set"):
            CommitOperationCopy(src_path_in_repo="src.bin", path_in_repo="dst.bin", src_repo_type="model")

    def test_path_normalization(self):
        op = CommitOperationCopy(src_path_in_repo="./src.bin", path_in_repo="/dst.bin")
        assert op.src_path_in_repo == "src.bin"
        assert op.path_in_repo == "dst.bin"


_RESOLVE_DEFAULTS = {
    "src_file_path": "file.txt",
    "src_root_path": None,
    "is_single_file": True,
    "destination_path": "",
    "destination_is_directory": False,
    "destination_exists_as_directory": False,
    "merge_contents": False,
}


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        # Single file cases
        ({"src_file_path": "file.txt", "destination_path": ""}, "file.txt"),
        ({"src_file_path": "file.txt", "destination_path": "renamed.txt"}, "renamed.txt"),
        ({"src_file_path": "file.txt", "destination_path": "dir", "destination_is_directory": True}, "dir/file.txt"),
        # Folder to nonexistent destination (rename semantics)
        (
            {
                "src_file_path": "folder/a.txt",
                "src_root_path": "folder",
                "is_single_file": False,
                "destination_path": "target",
            },
            "target/a.txt",
        ),
        # Folder to existing directory (cp -r nesting)
        (
            {
                "src_file_path": "folder/a.txt",
                "src_root_path": "folder",
                "is_single_file": False,
                "destination_path": "target",
                "destination_exists_as_directory": True,
            },
            "target/folder/a.txt",
        ),
        # Trailing slash on source (rsync semantics, no nesting)
        (
            {
                "src_file_path": "folder/a.txt",
                "src_root_path": "folder",
                "is_single_file": False,
                "destination_path": "target",
                "destination_exists_as_directory": True,
                "merge_contents": True,
            },
            "target/a.txt",
        ),
        # Folder to root (existing dir)
        (
            {
                "src_file_path": "folder/sub/a.txt",
                "src_root_path": "folder",
                "is_single_file": False,
                "destination_path": "",
                "destination_exists_as_directory": True,
            },
            "folder/sub/a.txt",
        ),
        # Folder contents to root
        (
            {
                "src_file_path": "folder/sub/a.txt",
                "src_root_path": "folder",
                "is_single_file": False,
                "destination_path": "",
                "destination_exists_as_directory": True,
                "merge_contents": True,
            },
            "sub/a.txt",
        ),
        # Nested subfolder
        (
            {
                "src_file_path": "data/train/a.csv",
                "src_root_path": "data",
                "is_single_file": False,
                "destination_path": "backup",
            },
            "backup/train/a.csv",
        ),
    ],
)
def test_resolve_copy_target_path(kwargs, expected):
    assert _resolve_copy_target_path(**{**_RESOLVE_DEFAULTS, **kwargs}) == expected
