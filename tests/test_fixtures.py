import unittest

import pytest

from huggingface_hub.utils import RepositoryNotFoundError


@pytest.mark.usefixtures("fx_cache_dir")
class TestCacheDirFixture(unittest.TestCase):
    """Test cache_dir fixture."""

    def test_cache_dir_fixture(self):
        self.assertTrue(self.cache_dir.exists())
        self.assertTrue(self.cache_dir.is_dir())
        self.assertIn("cache_dir_fixture", str(self.cache_dir))
        self.assertEqual(str(self.cache_dir), self.cache_dir_str)


@pytest.mark.usefixtures("fx_create_tmp_repo")
class TestCreateTmpRepoFixture(unittest.TestCase):
    """Test tmp repo fixture."""

    def test_create_tmp_repo_fixture(self):
        with self.create_tmp_repo(repo_type="dataset") as repo_id:
            dataset_info = self._api.repo_info(
                repo_id=repo_id,
                repo_type="dataset",
            )
            self.assertEqual(dataset_info.id, repo_id)

        with self.assertRaises(RepositoryNotFoundError):
            self._api.repo_info(
                repo_id=repo_id,
                repo_type="dataset",
            )
