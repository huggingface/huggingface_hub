from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pytest

from huggingface_hub.utils import DEFAULT_IGNORE_PATTERNS, filter_repo_objects


@dataclass
class DummyObject:
    path: Path


DUMMY_FILES = ["not_hidden.pdf", "profile.jpg", ".hidden.pdf", ".hidden_picture.png"]
DUMMY_PATHS = [Path(path) for path in DUMMY_FILES]
DUMMY_OBJECTS = [DummyObject(path=path) for path in DUMMY_FILES]


class TestPathsUtils:
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
        with pytest.raises(ValueError, match="Please provide `key` argument"):
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

    def test_filter_object_with_empty_pattern(self) -> None:
        # An empty pattern must not raise: `_add_wildcard_to_directories("")` indexed `pattern[-1]`.
        self._check(
            items=["file.txt", "lfs.bin"],
            expected_items=[],
            allow_patterns=[""],
        )
        self._check(
            items=["file.txt", "lfs.bin"],
            expected_items=["file.txt", "lfs.bin"],
            ignore_patterns=[""],
        )

    def test_filter_is_case_sensitive(self) -> None:
        """Pattern matching is case-sensitive.
        Regression test for https://github.com/huggingface/huggingface_hub/issues/4434.
        """
        # Uppercase pattern matches only the uppercase path, not the lowercase one.
        self._check(
            items=["README.md", "notes.MD"],
            expected_items=["notes.MD"],
            allow_patterns=["*.MD"],
        )
        # Lowercase pattern matches only the lowercase path, not the uppercase one.
        self._check(
            items=["README.md", "notes.MD"],
            expected_items=["README.md"],
            allow_patterns=["*.md"],
        )
        # Case-sensitivity also applies to the ignore list.
        self._check(
            items=["keep.txt", "drop.LOG", "keep.log"],
            expected_items=["keep.txt", "keep.log"],
            ignore_patterns=["*.LOG"],
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
        assert (
            list(
                filter_repo_objects(
                    items=items,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    key=key,
                )
            )
            == expected_items
        )


class TestDefaultIgnorePatterns:
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
        assert list(filtered_paths) == self.VALID_PATHS
