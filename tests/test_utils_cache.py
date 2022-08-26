import os
import shutil
import sys
import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator
from unittest.mock import Mock

import pytest

from _pytest.fixtures import SubRequest
from huggingface_hub._snapshot_download import snapshot_download
from huggingface_hub.commands.cache import ScanCacheCommand
from huggingface_hub.utils import scan_cache_dir

from .testing_constants import TOKEN


VALID_MODEL_ID = "valid_org/test_scan_repo_a"
VALID_DATASET_ID = "valid_org/test_scan_dataset_b"

REPO_A_MAIN_HASH = "401874e6a9c254a8baae85edd8a073921ecbd7f5"
REPO_A_PR_1_HASH = "fc674b0d440d3ea6f94bc4012e33ebd1dfc11b5b"
REPO_A_OTHER_HASH = "1da18ebd9185d146bcf84e308de53715d97d67d1"
REPO_A_MAIN_README_BLOB_HASH = "4baf04727c45b660add228b2934001991bd34b29"


@pytest.fixture
def fx_cache_dir(request: SubRequest) -> Generator[None, None, None]:
    """Add a `cache_dir` attribute pointing to a temporary directory."""
    with TemporaryDirectory() as cache_dir:
        request.cls.cache_dir = Path(cache_dir).resolve()
        yield


@pytest.mark.usefixtures("fx_cache_dir")
class TestValidCacheUtils(unittest.TestCase):
    cache_dir: Path

    def setUp(self) -> None:
        """Setup a clean cache for tests that will remain valid in all tests."""
        # Download latest main
        snapshot_download(
            repo_id=VALID_MODEL_ID,
            repo_type="model",
            cache_dir=self.cache_dir,
            use_auth_token=TOKEN,
        )

        # Download latest commit which is same as `main`
        snapshot_download(
            repo_id=VALID_MODEL_ID,
            revision=REPO_A_MAIN_HASH,
            repo_type="model",
            cache_dir=self.cache_dir,
            use_auth_token=TOKEN,
        )

        # Download the first commit
        snapshot_download(
            repo_id=VALID_MODEL_ID,
            revision=REPO_A_OTHER_HASH,
            repo_type="model",
            cache_dir=self.cache_dir,
            use_auth_token=TOKEN,
        )

        # Download from a PR
        snapshot_download(
            repo_id=VALID_MODEL_ID,
            revision="refs/pr/1",
            repo_type="model",
            cache_dir=self.cache_dir,
            use_auth_token=TOKEN,
        )

        # Download a Dataset repo from "main"
        snapshot_download(
            repo_id=VALID_DATASET_ID,
            revision="main",
            repo_type="dataset",
            cache_dir=self.cache_dir,
            use_auth_token=TOKEN,
        )

    def test_scan_cache_on_valid_cache(self) -> None:
        """Scan the cache dir without errors."""
        report = scan_cache_dir(self.cache_dir)

        # Check general information about downloaded snapshots
        self.assertEquals(report.size_on_disk, 3547)
        self.assertEquals(len(report.repos), 2)  # Model and dataset
        self.assertEquals(len(report.errors), 0)  # Repos are valid

        repo_a = [repo for repo in report.repos if repo.repo_id == VALID_MODEL_ID][0]

        # Check repo A general information
        repo_a_path = self.cache_dir / "models--valid_org--test_scan_repo_a"
        self.assertEquals(repo_a.repo_id, VALID_MODEL_ID)
        self.assertEquals(repo_a.repo_type, "model")
        self.assertEquals(repo_a.repo_path, repo_a_path)

        # 4 downloads but 3 revisions because "main" and REPO_A_MAIN_HASH are the same
        self.assertEquals(len(repo_a.revisions), 3)
        self.assertEquals(
            {rev.commit_hash for rev in repo_a.revisions},
            {REPO_A_MAIN_HASH, REPO_A_PR_1_HASH, REPO_A_OTHER_HASH},
        )

        # Repo size on disk is less than sum of revisions !
        self.assertEquals(repo_a.size_on_disk, 1391)
        self.assertEquals(sum(rev.size_on_disk for rev in repo_a.revisions), 4102)

        # Repo nb files is less than sum of revisions !
        self.assertEquals(repo_a.nb_files, 4)
        self.assertEquals(sum(rev.nb_files for rev in repo_a.revisions), 8)

        # 2 REFS in the repo: "main" and "refs/pr/1"
        # We could have add a tag as well
        self.assertEquals(set(repo_a.refs.keys()), {"main", "refs/pr/1"})
        self.assertEquals(repo_a.refs["main"].commit_hash, REPO_A_MAIN_HASH)
        self.assertEquals(repo_a.refs["refs/pr/1"].commit_hash, REPO_A_PR_1_HASH)

        # Check "main" revision information
        main_revision = repo_a.refs["main"]
        main_revision_path = repo_a_path / "snapshots" / REPO_A_MAIN_HASH

        self.assertEquals(main_revision.commit_hash, REPO_A_MAIN_HASH)
        self.assertEquals(main_revision.snapshot_path, main_revision_path)
        self.assertEquals(main_revision.refs, {"main"})

        # Same nb of files and size on disk that the sum
        self.assertEquals(main_revision.nb_files, len(main_revision.files))
        self.assertEquals(
            main_revision.size_on_disk,
            sum(file.size_on_disk for file in main_revision.files),
        )

        # Check readme file from "main" revision
        main_readme_file = [
            file for file in main_revision.files if file.file_name == "README.md"
        ][0]
        main_readme_file_path = main_revision_path / "README.md"
        main_readme_blob_path = repo_a_path / "blobs" / REPO_A_MAIN_README_BLOB_HASH

        self.assertEquals(main_readme_file.file_name, "README.md")
        self.assertEquals(main_readme_file.file_path, main_readme_file_path)
        self.assertEquals(main_readme_file.blob_path, main_readme_blob_path)

        # Check readme file from "refs/pr/1" revision
        pr_1_revision = repo_a.refs["refs/pr/1"]
        pr_1_revision_path = repo_a_path / "snapshots" / REPO_A_PR_1_HASH
        pr_1_readme_file = [
            file for file in pr_1_revision.files if file.file_name == "README.md"
        ][0]
        pr_1_readme_file_path = pr_1_revision_path / "README.md"

        # file_path in "refs/pr/1" revision is different than "main" but same blob path
        self.assertEquals(
            pr_1_readme_file.file_path, pr_1_readme_file_path
        )  # different
        self.assertEquals(pr_1_readme_file.blob_path, main_readme_blob_path)  # same

    def test_cli_scan_cache_quiet(self) -> None:
        """Test output from CLI scan cache with non verbose output.

        End-to-end test just to see if output is in expected format.
        """
        output = StringIO()
        args = Mock()
        args.verbose = 0
        args.dir = self.cache_dir

        # Taken from https://stackoverflow.com/a/34738440
        previous_output = sys.stdout
        sys.stdout = output
        ScanCacheCommand(args).run()
        sys.stdout = previous_output

        expected_output = f"""
        REPO ID                       REPO TYPE SIZE ON DISK NB FILES REFS            LOCAL PATH
        ----------------------------- --------- ------------ -------- --------------- -------------------------------------------------------------------------------------------------------------
        valid_org/test_scan_dataset_b dataset           2.2K        2 main            {self.cache_dir}/datasets--valid_org--test_scan_dataset_b
        valid_org/test_scan_repo_a    model             1.4K        4 main, refs/pr/1 {self.cache_dir}/models--valid_org--test_scan_repo_a

        Done in 0.0s. Scanned 2 repo(s) for a total of \x1b[1m\x1b[31m3.5K\x1b[0m.
        """

        self.assertListEqual(
            output.getvalue().replace("-", "").split(),
            expected_output.replace("-", "").split(),
        )

    def test_cli_scan_cache_verbose(self) -> None:
        """Test output from CLI scan cache with verbose output.

        End-to-end test just to see if output is in expected format.
        """
        output = StringIO()
        args = Mock()
        args.verbose = 1
        args.dir = self.cache_dir

        # Taken from https://stackoverflow.com/a/34738440
        previous_output = sys.stdout
        sys.stdout = output
        ScanCacheCommand(args).run()
        sys.stdout = previous_output

        expected_output = f"""
        REPO ID                       REPO TYPE REVISION                                 SIZE ON DISK NB FILES REFS      LOCAL PATH
        ----------------------------- --------- ---------------------------------------- ------------ -------- --------- ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        valid_org/test_scan_dataset_b dataset   1ac47c6f707cbc4825c2aa431ad5ab8cf09e60ed         2.2K        2 main      {self.cache_dir}/datasets--valid_org--test_scan_dataset_b/snapshots/1ac47c6f707cbc4825c2aa431ad5ab8cf09e60ed
        valid_org/test_scan_repo_a    model     1da18ebd9185d146bcf84e308de53715d97d67d1         1.3K        1           {self.cache_dir}/models--valid_org--test_scan_repo_a/snapshots/1da18ebd9185d146bcf84e308de53715d97d67d1
        valid_org/test_scan_repo_a    model     401874e6a9c254a8baae85edd8a073921ecbd7f5         1.4K        3 main      {self.cache_dir}/models--valid_org--test_scan_repo_a/snapshots/401874e6a9c254a8baae85edd8a073921ecbd7f5
        valid_org/test_scan_repo_a    model     fc674b0d440d3ea6f94bc4012e33ebd1dfc11b5b         1.4K        4 refs/pr/1 {self.cache_dir}/models--valid_org--test_scan_repo_a/snapshots/fc674b0d440d3ea6f94bc4012e33ebd1dfc11b5b

        Done in 0.0s. Scanned 2 repo(s) for a total of \x1b[1m\x1b[31m3.5K\x1b[0m.
        """

        self.assertListEqual(
            output.getvalue().replace("-", "").split(),
            expected_output.replace("-", "").split(),
        )


@pytest.mark.usefixtures("fx_cache_dir")
class TestCorruptedCacheUtils(unittest.TestCase):
    cache_dir: Path
    repo_path: Path

    def setUp(self) -> None:
        """Setup a clean cache for tests that will get corrupted in tests."""
        # Download latest main
        snapshot_download(
            repo_id=VALID_MODEL_ID,
            repo_type="model",
            cache_dir=self.cache_dir,
            use_auth_token=TOKEN,
        )

        self.repo_path = self.cache_dir / "models--valid_org--test_scan_repo_a"

    def test_repo_path_not_valid_dir(self) -> None:
        """Test if found a not valid path in cache dir."""
        # Case 1: a file
        repo_path = self.cache_dir / "a_file_that_should_not_be_there.txt"
        repo_path.touch()

        report = scan_cache_dir(self.cache_dir)
        self.assertEquals(len(report.repos), 1)  # Scan still worked !

        self.assertEqual(len(report.errors), 1)
        self.assertEqual(
            str(report.errors[0]), f"Repo path is not a directory: {repo_path}"
        )

        # Case 2: a folder with wrong naming
        os.remove(repo_path)
        repo_path = self.cache_dir / "a_folder_that_should_not_be_there"
        repo_path.mkdir()

        report = scan_cache_dir(self.cache_dir)
        self.assertEquals(len(report.repos), 1)  # Scan still worked !

        self.assertEqual(len(report.errors), 1)
        self.assertEqual(
            str(report.errors[0]),
            f"Repo path is not a valid HuggingFace cache directory: {repo_path}",
        )

        # Case 3: good naming but not a dataset/model/space
        shutil.rmtree(repo_path)
        repo_path = self.cache_dir / "not-models--t5-small"
        repo_path.mkdir()

        report = scan_cache_dir(self.cache_dir)
        self.assertEquals(len(report.repos), 1)  # Scan still worked !

        self.assertEqual(len(report.errors), 1)
        self.assertEqual(
            str(report.errors[0]),
            "Repo type must be `dataset`, `model` or `space`, found `not-model`"
            f" ({repo_path}).",
        )

    def test_snapshots_path_not_found(self) -> None:
        """Test if snapshots directory is missing in cached repo."""
        snapshots_path = self.repo_path / "snapshots"
        shutil.rmtree(snapshots_path)

        report = scan_cache_dir(self.cache_dir)
        self.assertEquals(len(report.repos), 0)  # Failed

        self.assertEqual(len(report.errors), 1)
        self.assertEqual(
            str(report.errors[0]),
            f"Snapshots dir doesn't exist in cached repo: {snapshots_path}",
        )

    def test_file_in_snapshots_dir(self) -> None:
        """Test if snapshots directory contains a file."""
        wrong_file_path = self.repo_path / "snapshots" / "should_not_be_there"
        wrong_file_path.touch()

        report = scan_cache_dir(self.cache_dir)
        self.assertEquals(len(report.repos), 0)  # Failed

        self.assertEqual(len(report.errors), 1)
        self.assertEqual(
            str(report.errors[0]),
            f"Snapshots folder corrupted. Found a file: {wrong_file_path}",
        )

    def test_ref_to_missing_revision(self) -> None:
        """Test if a `refs` points to a missing revision."""
        new_ref = self.repo_path / "refs" / "not_main"
        with new_ref.open("w") as f:
            f.write("revision_hash_that_does_not_exist")

        report = scan_cache_dir(self.cache_dir)
        self.assertEquals(len(report.repos), 0)  # Failed

        self.assertEqual(len(report.errors), 1)
        self.assertEqual(
            str(report.errors[0]),
            "Reference(s) refer to missing commit hashes:"
            " {'revision_hash_that_does_not_exist': {'not_main'}} "
            + f"({self.repo_path }).",
        )
