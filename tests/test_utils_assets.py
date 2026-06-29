from pathlib import Path
from unittest.mock import patch

import pytest

from huggingface_hub import cached_assets_path


class TestCacheAssets:
    def test_cached_assets_path_with_namespace_and_subfolder(self, tmp_path: Path) -> None:
        expected_path = tmp_path / "datasets" / "SQuAD" / "download"
        assert not expected_path.is_dir()

        path = cached_assets_path(
            library_name="datasets",
            namespace="SQuAD",
            subfolder="download",
            assets_dir=tmp_path,
        )

        assert path == expected_path  # Path is generated
        assert path.is_dir()  # And dir is created

    def test_cached_assets_path_without_subfolder(self, tmp_path: Path) -> None:
        path = cached_assets_path(library_name="datasets", namespace="SQuAD", assets_dir=tmp_path)
        assert path == tmp_path / "datasets" / "SQuAD" / "default"
        assert path.is_dir()

    def test_cached_assets_path_without_namespace(self, tmp_path: Path) -> None:
        path = cached_assets_path(library_name="datasets", subfolder="download", assets_dir=tmp_path)
        assert path == tmp_path / "datasets" / "default" / "download"
        assert path.is_dir()

    def test_cached_assets_path_without_namespace_and_subfolder(self, tmp_path: Path) -> None:
        path = cached_assets_path(library_name="datasets", assets_dir=tmp_path)
        assert path == tmp_path / "datasets" / "default" / "default"
        assert path.is_dir()

    def test_cached_assets_path_forbidden_symbols(self, tmp_path: Path) -> None:
        path = cached_assets_path(
            library_name="ReAlLy dumb",
            namespace="user/repo_name",
            subfolder="this is/not\\clever",
            assets_dir=tmp_path,
        )
        assert path == tmp_path / "ReAlLy--dumb" / "user--repo_name" / "this--is--not--clever"
        assert path.is_dir()

    def test_cached_assets_path_default_assets_dir(self, tmp_path: Path) -> None:
        with patch(
            "huggingface_hub.utils._cache_assets.HF_ASSETS_CACHE",
            tmp_path,
        ):  # Uses environment variable from HF_ASSETS_CACHE
            assert cached_assets_path(library_name="datasets") == tmp_path / "datasets" / "default" / "default"

    def test_cached_assets_path_is_a_file(self, tmp_path: Path) -> None:
        expected_path = tmp_path / "datasets" / "default" / "default"
        expected_path.parent.mkdir(parents=True)
        expected_path.touch()  # this should be the generated folder but is a file !

        with pytest.raises(ValueError):
            cached_assets_path(library_name="datasets", assets_dir=tmp_path)

    def test_cached_assets_path_parent_is_a_file(self, tmp_path: Path) -> None:
        expected_path = tmp_path / "datasets" / "default" / "default"
        expected_path.parent.parent.mkdir(parents=True)
        expected_path.parent.touch()  # cannot create folder as a parent is a file !

        with pytest.raises(ValueError):
            cached_assets_path(library_name="datasets", assets_dir=tmp_path)
