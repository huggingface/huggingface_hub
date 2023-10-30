import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from huggingface_hub import cached_assets_path


@pytest.mark.usefixtures("fx_cache_dir")
class CacheAssetsTest(unittest.TestCase):
    cache_dir: Path

    def test_cached_assets_path_with_namespace_and_subfolder(self) -> None:
        expected_path = self.cache_dir / "datasets" / "SQuAD" / "download"
        self.assertFalse(expected_path.is_dir())

        path = cached_assets_path(
            library_name="datasets",
            namespace="SQuAD",
            subfolder="download",
            assets_dir=self.cache_dir,
        )

        self.assertEqual(path, expected_path)  # Path is generated
        self.assertTrue(path.is_dir())  # And dir is created

    def test_cached_assets_path_without_subfolder(self) -> None:
        path = cached_assets_path(library_name="datasets", namespace="SQuAD", assets_dir=self.cache_dir)
        self.assertEqual(path, self.cache_dir / "datasets" / "SQuAD" / "default")
        self.assertTrue(path.is_dir())

    def test_cached_assets_path_without_namespace(self) -> None:
        path = cached_assets_path(library_name="datasets", subfolder="download", assets_dir=self.cache_dir)
        self.assertEqual(path, self.cache_dir / "datasets" / "default" / "download")
        self.assertTrue(path.is_dir())

    def test_cached_assets_path_without_namespace_and_subfolder(self) -> None:
        path = cached_assets_path(library_name="datasets", assets_dir=self.cache_dir)
        self.assertEqual(path, self.cache_dir / "datasets" / "default" / "default")
        self.assertTrue(path.is_dir())

    def test_cached_assets_path_forbidden_symbols(self) -> None:
        path = cached_assets_path(
            library_name="ReAlLy dumb",
            namespace="user/repo_name",
            subfolder="this is/not\\clever",
            assets_dir=self.cache_dir,
        )
        self.assertEqual(
            path,
            self.cache_dir / "ReAlLy--dumb" / "user--repo_name" / "this--is--not--clever",
        )
        self.assertTrue(path.is_dir())

    def test_cached_assets_path_default_assets_dir(self) -> None:
        with patch(
            "huggingface_hub.utils._cache_assets.HF_ASSETS_CACHE",
            self.cache_dir,
        ):  # Uses environment variable from HF_ASSETS_CACHE
            self.assertEqual(
                cached_assets_path(library_name="datasets"),
                self.cache_dir / "datasets" / "default" / "default",
            )

    def test_cached_assets_path_is_a_file(self) -> None:
        expected_path = self.cache_dir / "datasets" / "default" / "default"
        expected_path.parent.mkdir(parents=True)
        expected_path.touch()  # this should be the generated folder but is a file !

        with self.assertRaises(ValueError):
            cached_assets_path(library_name="datasets", assets_dir=self.cache_dir)

    def test_cached_assets_path_parent_is_a_file(self) -> None:
        expected_path = self.cache_dir / "datasets" / "default" / "default"
        expected_path.parent.parent.mkdir(parents=True)
        expected_path.parent.touch()  # cannot create folder as a parent is a file !

        with self.assertRaises(ValueError):
            cached_assets_path(library_name="datasets", assets_dir=self.cache_dir)
