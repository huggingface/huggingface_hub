import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from huggingface_hub.utils import IGNORE_GIT_FOLDER_PATTERNS, filter_repo_objects


@dataclass
class DummyObject:
    path: Path


DUMMY_FILES = ["not_hidden.pdf", "profile.jpg", ".hidden.pdf", ".hidden_picture.png"]
DUMMY_PATHS = [Path(path) for path in DUMMY_FILES]
DUMMY_OBJECTS = [DummyObject(path=path) for path in DUMMY_FILES]


class TestPathsUtils(unittest.TestCase):
    def test_get_all_pdfs(self) -> None:
        """Get all PDFs even hidden ones."""
        self._check(
            items=DUMMY_FILES,
            expected_items=["not_hidden.pdf", ".hidden.pdf"],
            allow_patterns=["*.pdf"],
        )

    def test_get_all_pdfs_except_hidden(self) -> None:
        """Get all PDFs except hidden ones."""
        self._check(
            items=DUMMY_FILES,
            expected_items=["not_hidden.pdf"],
            allow_patterns=["*.pdf"],
            ignore_patterns=[".*"],
        )

    def test_get_all_pdfs_except_hidden_using_single_pattern(self) -> None:
        """Get all PDFs except hidden ones, using single pattern."""
        self._check(
            items=DUMMY_FILES,
            expected_items=["not_hidden.pdf"],
            allow_patterns="*.pdf",  # not a list
            ignore_patterns=".*",  # not a list
        )

    def test_get_all_images(self) -> None:
        """Get all images."""
        self._check(
            items=DUMMY_FILES,
            expected_items=["profile.jpg", ".hidden_picture.png"],
            allow_patterns=["*.png", "*.jpg"],
        )

    def test_get_all_images_except_hidden_from_paths(self) -> None:
        """Get all images except hidden ones, from Path list."""
        self._check(
            items=DUMMY_PATHS,
            expected_items=[Path("profile.jpg")],
            allow_patterns=["*.png", "*.jpg"],
            ignore_patterns=".*",
        )

    def test_get_all_images_except_hidden_from_objects(self) -> None:
        """Get all images except hidden ones, from object list."""
        self._check(
            items=DUMMY_OBJECTS,
            expected_items=[DummyObject(path="profile.jpg")],
            allow_patterns=["*.png", "*.jpg"],
            ignore_patterns=".*",
            key=lambda x: x.path,
        )

    def test_filter_objects_key_not_provided(self) -> None:
        """Test ValueError is raised if filtering non-string objects."""
        with self.assertRaisesRegex(ValueError, "Please provide `key` argument"):
            list(
                filter_repo_objects(
                    items=DUMMY_OBJECTS,
                    allow_patterns=["*.png", "*.jpg"],
                    ignore_patterns=".*",
                )
            )

    def _check(
        self,
        items: List[Any],
        expected_items: List[Any],
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        key: Optional[Callable[[Any], str]] = None,
    ) -> None:
        """Run `filter_repo_objects` and check output against expected result."""
        self.assertListEqual(
            list(
                filter_repo_objects(
                    items=items,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    key=key,
                )
            ),
            expected_items,
        )


class TestGitFolderExclusion(unittest.TestCase):
    GIT_FOLDER_PATHS = [
        ".git",
        ".git/file.txt",
        ".git/folder/file.txt",
        "path/to/folder/.git",
        "path/to/folder/.git/file.txt",
        "path/to/.git/folder/file.txt",
    ]

    NOT_GIT_FOLDER_PATHS = [
        ".gitignore",
        "path/foo.git/file.txt",
        "path/.git_bar/file.txt",
        "path/to/file.git",
    ]

    def test_exclude_git_folder(self):
        filtered_paths = filter_repo_objects(
            items=self.GIT_FOLDER_PATHS + self.NOT_GIT_FOLDER_PATHS, ignore_patterns=IGNORE_GIT_FOLDER_PATTERNS
        )
        self.assertListEqual(list(filtered_paths), self.NOT_GIT_FOLDER_PATHS)
