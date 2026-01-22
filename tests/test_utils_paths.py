import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

from huggingface_hub.utils import DEFAULT_IGNORE_PATTERNS, filter_repo_objects
from huggingface_hub.utils._paths import _fnmatch_path


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

    def test_filter_object_with_folder(self) -> None:
        self._check(
            items=[
                "file.txt",
                "lfs.bin",
                "path/to/file.txt",
                "path/to/lfs.bin",
                "nested/path/to/file.txt",
                "nested/path/to/lfs.bin",
            ],
            expected_items=["path/to/file.txt", "path/to/lfs.bin"],
            allow_patterns=["path/to/"],
        )

    def _check(
        self,
        items: list[Any],
        expected_items: list[Any],
        allow_patterns: Optional[Union[list[str], str]] = None,
        ignore_patterns: Optional[Union[list[str], str]] = None,
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


class TestDefaultIgnorePatterns(unittest.TestCase):
    PATHS_TO_IGNORE = [
        ".git",
        ".git/file.txt",
        ".git/folder/file.txt",
        "path/to/folder/.git",
        "path/to/folder/.git/file.txt",
        "path/to/.git/folder/file.txt",
        ".cache/huggingface",
        ".cache/huggingface/file.txt",
        ".cache/huggingface/folder/file.txt",
        "path/to/.cache/huggingface",
        "path/to/.cache/huggingface/file.txt",
    ]

    VALID_PATHS = [
        ".gitignore",
        "path/foo.git/file.txt",
        "path/.git_bar/file.txt",
        "path/to/file.git",
        "file.huggingface",
        "path/file.huggingface",
        ".cache/huggingface_folder",
        ".cache/huggingface_folder/file.txt",
    ]

    def test_exclude_git_folder(self):
        filtered_paths = filter_repo_objects(
            items=self.PATHS_TO_IGNORE + self.VALID_PATHS, ignore_patterns=DEFAULT_IGNORE_PATTERNS
        )
        self.assertListEqual(list(filtered_paths), self.VALID_PATHS)


class TestWildcardDirectoryBoundaries(unittest.TestCase):
    """Test that wildcard patterns respect directory boundaries (Issue #3709)."""

    def test_wildcard_does_not_match_subdirectories(self):
        """Test that `data/*.json` only matches files directly under `data/`, not in subdirectories."""
        items = [
            "data/adam.json",
            "data/lion.json",
            "data/garbage.json",
            "data/another_garbage.json",
            "data/test_adafactor.json",
            "data/H100/adam.json",
            "data/H100/lion.json",
            "data/T4/adam.json",
            "data/T4/lion.json",
        ]
        expected_items = [
            "data/adam.json",
            "data/lion.json",
            "data/garbage.json",
            "data/another_garbage.json",
            "data/test_adafactor.json",
        ]
        self._check(
            items=items,
            expected_items=expected_items,
            allow_patterns=["data/*.json"],
        )

    def test_recursive_wildcard_matches_subdirectories(self):
        """Test that `data/**/*.json` matches files in subdirectories."""
        items = [
            "data/adam.json",
            "data/garbage.json",
            "data/H100/adam.json",
            "data/H100/lion.json",
            "data/T4/adam.json",
            "data/T4/lion.json",
        ]
        expected_items = items
        self._check(
            items=items,
            expected_items=expected_items,
            allow_patterns=["data/**/*.json"],
        )

    def test_single_level_wildcard_in_middle(self):
        """Test that `path/*/file.txt` matches one level but not nested."""
        items = [
            "path/to/file.txt",
            "path/from/file.txt",
            "path/to/nested/file.txt",
        ]
        expected_items = [
            "path/to/file.txt",
            "path/from/file.txt",
        ]
        self._check(
            items=items,
            expected_items=expected_items,
            allow_patterns=["path/*/file.txt"],
        )

    def _check(
        self,
        items: list[str],
        expected_items: list[str],
        allow_patterns: Optional[Union[list[str], str]] = None,
        ignore_patterns: Optional[Union[list[str], str]] = None,
    ) -> None:
        """Run `filter_repo_objects` and check output against expected result."""
        self.assertListEqual(
            sorted(list(filter_repo_objects(items=items, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns))),
            sorted(expected_items),
        )


class TestFnmatchPath(unittest.TestCase):
    """Test the `_fnmatch_path` function directly."""

    def test_wildcard_does_not_match_directory_separator(self):
        """Test that `*` does not match `/` character."""
        self.assertTrue(_fnmatch_path("data/adam.json", "data/*.json"))
        self.assertFalse(_fnmatch_path("data/H100/adam.json", "data/*.json"))

    def test_recursive_wildcard_matches_nested_paths(self):
        """Test that `**` matches nested directory structures."""
        self.assertTrue(_fnmatch_path("data/file.json", "data/**/file.json"))
        self.assertTrue(_fnmatch_path("data/sub/file.json", "data/**/file.json"))
        self.assertTrue(_fnmatch_path("data/H100/adam.json", "data/**/*.json"))
