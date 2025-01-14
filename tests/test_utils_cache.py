import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

import pytest

from huggingface_hub._snapshot_download import snapshot_download
from huggingface_hub.commands.scan_cache import ScanCacheCommand
from huggingface_hub.utils import DeleteCacheStrategy, HFCacheInfo, capture_output, scan_cache_dir
from huggingface_hub.utils._cache_manager import (
    CacheNotFound,
    _format_size,
    _format_timesince,
    _try_delete_path,
)

from .testing_constants import TOKEN
from .testing_utils import (
    rmtree_with_retry,
    with_production_testing,
    xfail_on_windows,
)


# On production server to avoid recreating them all the time
MODEL_ID = "hf-internal-testing/hfh_ci_scan_repo_a"
MODEL_PATH = "models--hf-internal-testing--hfh_ci_scan_repo_a"

DATASET_ID = "hf-internal-testing/hfh_ci_scan_dataset_b"
DATASET_PATH = "datasets--hf-internal-testing--hfh_ci_scan_dataset_b"

REPO_A_MAIN_HASH = "c0d57e03d9f128062eadb6665618982db612b2e3"
REPO_A_PR_1_HASH = "1a665a9d28a66b1d0f8edd9359fc824aacc63234"
REPO_A_OTHER_HASH = "f95875cd910793299a545417cc4b3c9055202883"
REPO_A_MAIN_README_BLOB_HASH = "fffc22b462ba2368b09b4d38527760051c9090a9"
REPO_B_MAIN_HASH = "f1cdcd4641b3ea2dfa8d4333dba1ea3b532735e1"

REF_1_NAME = "refs/pr/1"


@pytest.mark.usefixtures("fx_cache_dir")
class TestMissingCacheUtils(unittest.TestCase):
    cache_dir: Path

    def test_cache_dir_is_missing(self) -> None:
        """Directory to scan does not exist raises CacheNotFound."""
        self.assertRaises(CacheNotFound, scan_cache_dir, self.cache_dir / "does_not_exist")

    def test_cache_dir_is_a_file(self) -> None:
        """Directory to scan is a file raises ValueError."""
        file_path = self.cache_dir / "file.txt"
        file_path.touch()
        self.assertRaises(ValueError, scan_cache_dir, file_path)


@pytest.mark.usefixtures("fx_cache_dir")
class TestValidCacheUtils(unittest.TestCase):
    cache_dir: Path

    @with_production_testing
    def setUp(self) -> None:
        """Setup a clean cache for tests that will remain valid in all tests."""
        # Download latest main
        snapshot_download(
            repo_id=MODEL_ID,
            repo_type="model",
            cache_dir=self.cache_dir,
            use_auth_token=TOKEN,
        )

        # Download latest commit which is same as `main`
        snapshot_download(
            repo_id=MODEL_ID,
            revision=REPO_A_MAIN_HASH,
            repo_type="model",
            cache_dir=self.cache_dir,
            use_auth_token=TOKEN,
        )

        # Download the first commit
        snapshot_download(
            repo_id=MODEL_ID,
            revision=REPO_A_OTHER_HASH,
            repo_type="model",
            cache_dir=self.cache_dir,
            use_auth_token=TOKEN,
        )

        # Download from a PR
        snapshot_download(
            repo_id=MODEL_ID,
            revision="refs/pr/1",
            repo_type="model",
            cache_dir=self.cache_dir,
            use_auth_token=TOKEN,
        )

        # Download a Dataset repo from "main"
        snapshot_download(
            repo_id=DATASET_ID,
            revision="main",
            repo_type="dataset",
            cache_dir=self.cache_dir,
            use_auth_token=TOKEN,
        )

    @unittest.skipIf(os.name == "nt", "Windows cache is tested separately")
    def test_scan_cache_on_valid_cache_unix(self) -> None:
        """Scan the cache dir without warnings (on unix-based platform).

        This test is duplicated and adapted for Windows in `test_scan_cache_on_valid_cache_windows`.
        Note: Please make sure to updated both if any change is made.
        """
        report = scan_cache_dir(self.cache_dir)

        # Check general information about downloaded snapshots
        self.assertEqual(report.size_on_disk, 3766)
        self.assertEqual(len(report.repos), 2)  # Model and dataset
        self.assertEqual(len(report.warnings), 0)  # Repos are valid

        repo_a = [repo for repo in report.repos if repo.repo_id == MODEL_ID][0]

        # Check repo A general information
        repo_a_path = self.cache_dir / MODEL_PATH
        self.assertEqual(repo_a.repo_id, MODEL_ID)
        self.assertEqual(repo_a.repo_type, "model")
        self.assertEqual(repo_a.repo_path, repo_a_path)

        # 4 downloads but 3 revisions because "main" and REPO_A_MAIN_HASH are the same
        self.assertEqual(len(repo_a.revisions), 3)
        self.assertEqual(
            {rev.commit_hash for rev in repo_a.revisions},
            {REPO_A_MAIN_HASH, REPO_A_PR_1_HASH, REPO_A_OTHER_HASH},
        )

        # Repo size on disk is less than sum of revisions !
        self.assertEqual(repo_a.size_on_disk, 1501)
        self.assertEqual(sum(rev.size_on_disk for rev in repo_a.revisions), 4463)

        # Repo nb files is less than sum of revisions !
        self.assertEqual(repo_a.nb_files, 3)
        self.assertEqual(sum(rev.nb_files for rev in repo_a.revisions), 6)

        # 2 REFS in the repo: "main" and "refs/pr/1"
        # We could have add a tag as well
        self.assertEqual(set(repo_a.refs.keys()), {"main", REF_1_NAME})
        self.assertEqual(repo_a.refs["main"].commit_hash, REPO_A_MAIN_HASH)
        self.assertEqual(repo_a.refs[REF_1_NAME].commit_hash, REPO_A_PR_1_HASH)

        # Check "main" revision information
        main_revision = repo_a.refs["main"]
        main_revision_path = repo_a_path / "snapshots" / REPO_A_MAIN_HASH

        self.assertEqual(main_revision.commit_hash, REPO_A_MAIN_HASH)
        self.assertEqual(main_revision.snapshot_path, main_revision_path)
        self.assertEqual(main_revision.refs, {"main"})

        # Same nb of files and size on disk that the sum
        self.assertEqual(main_revision.nb_files, len(main_revision.files))
        self.assertEqual(
            main_revision.size_on_disk,
            sum(file.size_on_disk for file in main_revision.files),
        )

        # Check readme file from "main" revision
        main_readme_file = [file for file in main_revision.files if file.file_name == "README.md"][0]
        main_readme_file_path = main_revision_path / "README.md"
        main_readme_blob_path = repo_a_path / "blobs" / REPO_A_MAIN_README_BLOB_HASH

        self.assertEqual(main_readme_file.file_name, "README.md")
        self.assertEqual(main_readme_file.file_path, main_readme_file_path)
        self.assertEqual(main_readme_file.blob_path, main_readme_blob_path)

        # Check readme file from "refs/pr/1" revision
        pr_1_revision = repo_a.refs[REF_1_NAME]
        pr_1_revision_path = repo_a_path / "snapshots" / REPO_A_PR_1_HASH
        pr_1_readme_file = [file for file in pr_1_revision.files if file.file_name == "README.md"][0]
        pr_1_readme_file_path = pr_1_revision_path / "README.md"

        # file_path in "refs/pr/1" revision is different than "main" but same blob path
        self.assertEqual(pr_1_readme_file.file_path, pr_1_readme_file_path)  # different
        self.assertEqual(pr_1_readme_file.blob_path, main_readme_blob_path)  # same

    @unittest.skipIf(os.name != "nt", "Windows cache is tested separately")
    def test_scan_cache_on_valid_cache_windows(self) -> None:
        """Scan the cache dir without warnings (on Windows).

        Windows tests do not use symlinks which leads to duplication in the cache.
        This test is duplicated from `test_scan_cache_on_valid_cache_unix` with a few
        tweaks specific to windows.
        Note: Please make sure to updated both if any change is made.
        """
        report = scan_cache_dir(self.cache_dir)

        # Check general information about downloaded snapshots
        self.assertEqual(report.size_on_disk, 6728)
        self.assertEqual(len(report.repos), 2)  # Model and dataset
        self.assertEqual(len(report.warnings), 0)  # Repos are valid

        repo_a = [repo for repo in report.repos if repo.repo_id == MODEL_ID][0]

        # Check repo A general information
        repo_a_path = self.cache_dir / MODEL_PATH
        self.assertEqual(repo_a.repo_id, MODEL_ID)
        self.assertEqual(repo_a.repo_type, "model")
        self.assertEqual(repo_a.repo_path, repo_a_path)

        # 4 downloads but 3 revisions because "main" and REPO_A_MAIN_HASH are the same
        self.assertEqual(len(repo_a.revisions), 3)
        self.assertEqual(
            {rev.commit_hash for rev in repo_a.revisions},
            {REPO_A_MAIN_HASH, REPO_A_PR_1_HASH, REPO_A_OTHER_HASH},
        )

        # Repo size on disk is equal to the sum of revisions (no symlinks)
        self.assertEqual(repo_a.size_on_disk, 4463)  # Windows-specific
        self.assertEqual(sum(rev.size_on_disk for rev in repo_a.revisions), 4463)

        # Repo nb files is equal to the sum of revisions !
        self.assertEqual(repo_a.nb_files, 6)  # Windows-specific
        self.assertEqual(sum(rev.nb_files for rev in repo_a.revisions), 6)

        # 2 REFS in the repo: "main" and "refs/pr/1"
        # We could have add a tag as well
        REF_1_NAME = "refs\\pr\\1"  # Windows-specific
        self.assertEqual(set(repo_a.refs.keys()), {"main", REF_1_NAME})
        self.assertEqual(repo_a.refs["main"].commit_hash, REPO_A_MAIN_HASH)
        self.assertEqual(repo_a.refs[REF_1_NAME].commit_hash, REPO_A_PR_1_HASH)

        # Check "main" revision information
        main_revision = repo_a.refs["main"]
        main_revision_path = repo_a_path / "snapshots" / REPO_A_MAIN_HASH

        self.assertEqual(main_revision.commit_hash, REPO_A_MAIN_HASH)
        self.assertEqual(main_revision.snapshot_path, main_revision_path)
        self.assertEqual(main_revision.refs, {"main"})

        # Same nb of files and size on disk that the sum
        self.assertEqual(main_revision.nb_files, len(main_revision.files))
        self.assertEqual(
            main_revision.size_on_disk,
            sum(file.size_on_disk for file in main_revision.files),
        )

        # Check readme file from "main" revision
        main_readme_file = [file for file in main_revision.files if file.file_name == "README.md"][0]
        main_readme_file_path = main_revision_path / "README.md"
        main_readme_blob_path = repo_a_path / "blobs" / REPO_A_MAIN_README_BLOB_HASH

        self.assertEqual(main_readme_file.file_name, "README.md")
        self.assertEqual(main_readme_file.file_path, main_readme_file_path)
        self.assertEqual(main_readme_file.blob_path, main_readme_file_path)  # Windows-specific: no blob file
        self.assertFalse(main_readme_blob_path.exists())  # Windows-specific

        # Check readme file from "refs/pr/1" revision
        pr_1_revision = repo_a.refs[REF_1_NAME]
        pr_1_revision_path = repo_a_path / "snapshots" / REPO_A_PR_1_HASH
        pr_1_readme_file = [file for file in pr_1_revision.files if file.file_name == "README.md"][0]
        pr_1_readme_file_path = pr_1_revision_path / "README.md"

        # file_path in "refs/pr/1" revision is different than "main"
        # Windows-specific: even blob path is different
        self.assertEqual(pr_1_readme_file.file_path, pr_1_readme_file_path)
        self.assertNotEqual(  # Windows-specific: different as well
            pr_1_readme_file.blob_path, main_readme_file.blob_path
        )

    @xfail_on_windows("Size on disk and paths differ on Windows. Not useful to test.")
    def test_cli_scan_cache_quiet(self) -> None:
        """Test output from CLI scan cache with non verbose output.

        End-to-end test just to see if output is in expected format.
        """
        args = Mock()
        args.verbose = 0
        args.dir = self.cache_dir

        with capture_output() as output:
            ScanCacheCommand(args).run()

        expected_output = f"""
        REPO ID                       REPO TYPE SIZE ON DISK NB FILES LAST_ACCESSED     LAST_MODIFIED     REFS            LOCAL PATH
        ----------------------------- --------- ------------ -------- ----------------- ----------------- --------------- ---------------------------------------------------------
        {DATASET_ID} dataset           2.3K        1 a few seconds ago a few seconds ago main            {self.cache_dir}/{DATASET_PATH}
        {MODEL_ID}   model             1.5K        3 a few seconds ago a few seconds ago main, refs/pr/1 {self.cache_dir}/{MODEL_PATH}

        Done in 0.0s. Scanned 2 repo(s) for a total of \x1b[1m\x1b[31m3.8K\x1b[0m.
        """

        self.assertListEqual(
            output.getvalue().replace("-", "").split(),
            expected_output.replace("-", "").split(),
        )

    @xfail_on_windows("Size on disk and paths differ on Windows. Not useful to test.")
    def test_cli_scan_cache_verbose(self) -> None:
        """Test output from CLI scan cache with verbose output.

        End-to-end test just to see if output is in expected format.
        """
        args = Mock()
        args.verbose = 1
        args.dir = self.cache_dir

        with capture_output() as output:
            ScanCacheCommand(args).run()

        expected_output = f"""
        REPO ID                       REPO TYPE REVISION                                 SIZE ON DISK NB FILES LAST_MODIFIED     REFS      LOCAL PATH
        ----------------------------- --------- ---------------------------------------- ------------ -------- ----------------- --------- ------------------------------------------------------------------------------------------------------------
        {DATASET_ID} dataset   {REPO_B_MAIN_HASH}          2.3K        1 a few seconds ago main      {self.cache_dir}/{DATASET_PATH}/snapshots/{REPO_B_MAIN_HASH}
        {MODEL_ID}   model     {REPO_A_PR_1_HASH}          1.5K        3 a few seconds ago refs/pr/1 {self.cache_dir}/{MODEL_PATH}/snapshots/{REPO_A_PR_1_HASH}
        {MODEL_ID}   model     {REPO_A_MAIN_HASH}          1.5K        2 a few seconds ago main      {self.cache_dir}/{MODEL_PATH}/snapshots/{REPO_A_MAIN_HASH}
        {MODEL_ID}   model     {REPO_A_OTHER_HASH}         1.5K        1 a few seconds ago           {self.cache_dir}/{MODEL_PATH}/snapshots/{REPO_A_OTHER_HASH}

        Done in 0.0s. Scanned 2 repo(s) for a total of \x1b[1m\x1b[31m3.8K\x1b[0m.
        """

        self.assertListEqual(
            output.getvalue().replace("-", "").split(),
            expected_output.replace("-", "").split(),
        )

    def test_cli_scan_missing_cache(self) -> None:
        """Test output from CLI scan cache when cache does not exist.

        End-to-end test just to see if output is in expected format.
        """
        tmp_dir = tempfile.mkdtemp()
        os.rmdir(tmp_dir)

        args = Mock()
        args.verbose = 0
        args.dir = tmp_dir

        with capture_output() as output:
            ScanCacheCommand(args).run()

        expected_output = f"""
        Cache directory not found: {Path(tmp_dir).resolve()}
        """

        self.assertListEqual(output.getvalue().split(), expected_output.split())


@pytest.mark.usefixtures("fx_cache_dir")
class TestCorruptedCacheUtils(unittest.TestCase):
    cache_dir: Path
    repo_path: Path
    refs_path: Path
    snapshots_path: Path

    @with_production_testing
    def setUp(self) -> None:
        """Setup a clean cache for tests that will get corrupted/modified in tests."""
        # Download latest main
        snapshot_download(
            repo_id=MODEL_ID,
            repo_type="model",
            cache_dir=self.cache_dir,
            use_auth_token=TOKEN,
        )

        self.repo_path = self.cache_dir / MODEL_PATH
        self.refs_path = self.repo_path / "refs"
        self.snapshots_path = self.repo_path / "snapshots"

    def test_repo_path_not_valid_dir(self) -> None:
        """Test if found a not valid path in cache dir."""
        # Case 1: a file
        repo_path = self.cache_dir / "a_file_that_should_not_be_there.txt"
        repo_path.touch()

        report = scan_cache_dir(self.cache_dir)
        self.assertEqual(len(report.repos), 1)  # Scan still worked !

        self.assertEqual(len(report.warnings), 1)
        self.assertEqual(str(report.warnings[0]), f"Repo path is not a directory: {repo_path}")

        # Case 2: a folder with wrong naming
        os.remove(repo_path)
        repo_path = self.cache_dir / "a_folder_that_should_not_be_there"
        repo_path.mkdir()

        report = scan_cache_dir(self.cache_dir)
        self.assertEqual(len(report.repos), 1)  # Scan still worked !

        self.assertEqual(len(report.warnings), 1)
        self.assertEqual(
            str(report.warnings[0]),
            f"Repo path is not a valid HuggingFace cache directory: {repo_path}",
        )

        # Case 3: good naming but not a dataset/model/space
        rmtree_with_retry(repo_path)
        repo_path = self.cache_dir / "not-models--t5-small"
        repo_path.mkdir()

        report = scan_cache_dir(self.cache_dir)
        self.assertEqual(len(report.repos), 1)  # Scan still worked !

        self.assertEqual(len(report.warnings), 1)
        self.assertEqual(
            str(report.warnings[0]),
            f"Repo type must be `dataset`, `model` or `space`, found `not-model` ({repo_path}).",
        )

    def test_snapshots_path_not_found(self) -> None:
        """Test if snapshots directory is missing in cached repo."""
        rmtree_with_retry(self.snapshots_path)

        report = scan_cache_dir(self.cache_dir)
        self.assertEqual(len(report.repos), 0)  # Failed

        self.assertEqual(len(report.warnings), 1)
        self.assertEqual(
            str(report.warnings[0]),
            f"Snapshots dir doesn't exist in cached repo: {self.snapshots_path}",
        )

    def test_file_in_snapshots_dir(self) -> None:
        """Test if snapshots directory contains a file."""
        wrong_file_path = self.snapshots_path / "should_not_be_there"
        wrong_file_path.touch()

        report = scan_cache_dir(self.cache_dir)
        self.assertEqual(len(report.repos), 0)  # Failed

        self.assertEqual(len(report.warnings), 1)
        self.assertEqual(
            str(report.warnings[0]),
            f"Snapshots folder corrupted. Found a file: {wrong_file_path}",
        )

    def test_snapshot_with_no_blob_files(self) -> None:
        """Test if a snapshot directory (e.g. a cached revision) is empty."""
        for revision_path in self.snapshots_path.glob("*"):
            # Delete content of the revision
            rmtree_with_retry(revision_path)
            revision_path.mkdir()

        # Scan
        report = scan_cache_dir(self.cache_dir)

        # Get single repo
        self.assertEqual(len(report.warnings), 0)  # Did not fail
        self.assertEqual(len(report.repos), 1)
        repo_report = list(report.repos)[0]

        # Repo report is empty
        self.assertEqual(repo_report.size_on_disk, 0)
        self.assertEqual(len(repo_report.revisions), 1)
        revision_report = list(repo_report.revisions)[0]

        # No files in revision so last_modified is the one from the revision folder
        self.assertEqual(revision_report.nb_files, 0)
        self.assertEqual(revision_report.last_modified, revision_path.stat().st_mtime)

    def test_repo_with_no_snapshots(self) -> None:
        """Test if the snapshot directory exists but is empty."""
        rmtree_with_retry(self.refs_path)
        rmtree_with_retry(self.snapshots_path)
        self.snapshots_path.mkdir()

        # Scan
        report = scan_cache_dir(self.cache_dir)

        # Get single repo
        self.assertEqual(len(report.warnings), 0)  # Did not fail
        self.assertEqual(len(report.repos), 1)
        repo_report = list(report.repos)[0]

        # No revisions in repos so last_modified is the one from the repo folder
        self.assertEqual(repo_report.size_on_disk, 0)
        self.assertEqual(len(repo_report.revisions), 0)
        self.assertEqual(repo_report.last_modified, self.repo_path.stat().st_mtime)
        self.assertEqual(repo_report.last_accessed, self.repo_path.stat().st_atime)

    def test_ref_to_missing_revision(self) -> None:
        """Test if a `refs` points to a missing revision."""
        new_ref = self.repo_path / "refs" / "not_main"
        with new_ref.open("w") as f:
            f.write("revision_hash_that_does_not_exist")

        report = scan_cache_dir(self.cache_dir)
        self.assertEqual(len(report.repos), 0)  # Failed

        self.assertEqual(len(report.warnings), 1)
        self.assertEqual(
            str(report.warnings[0]),
            "Reference(s) refer to missing commit hashes: {'revision_hash_that_does_not_exist': {'not_main'}} "
            + f"({self.repo_path}).",
        )

    @xfail_on_windows("Last modified/last accessed work a bit differently on Windows.")
    def test_scan_cache_last_modified_and_last_accessed(self) -> None:
        """Scan the last_modified and last_accessed properties when scanning."""
        TIME_GAP = 0.1

        # Make a first scan
        report_1 = scan_cache_dir(self.cache_dir)

        # Values from first report
        repo_1 = list(report_1.repos)[0]
        revision_1 = list(repo_1.revisions)[0]
        readme_file_1 = [file for file in revision_1.files if file.file_name == "README.md"][0]
        another_file_1 = [file for file in revision_1.files if file.file_name == ".gitattributes"][0]

        # Comparison of last_accessed/last_modified between file and repo
        self.assertLessEqual(readme_file_1.blob_last_accessed, repo_1.last_accessed)
        self.assertLessEqual(readme_file_1.blob_last_modified, repo_1.last_modified)
        self.assertEqual(revision_1.last_modified, repo_1.last_modified)

        # Sleep and write new readme
        time.sleep(TIME_GAP)
        readme_file_1.file_path.write_text("modified readme")

        # Sleep and read content from readme
        time.sleep(TIME_GAP)
        with readme_file_1.file_path.open("r") as f:
            _ = f.read()

        # Sleep and re-scan
        time.sleep(TIME_GAP)
        report_2 = scan_cache_dir(self.cache_dir)

        # Values from second report
        repo_2 = list(report_2.repos)[0]
        revision_2 = list(repo_2.revisions)[0]
        readme_file_2 = [file for file in revision_2.files if file.file_name == "README.md"][0]
        another_file_2 = [file for file in revision_1.files if file.file_name == ".gitattributes"][0]

        # Report 1 is not updated when cache changes
        self.assertLess(repo_1.last_accessed, repo_2.last_accessed)
        self.assertLess(repo_1.last_modified, repo_2.last_modified)

        # "Another_file.md" did not change
        self.assertEqual(another_file_1, another_file_2)

        # Readme.md has been modified and then accessed more recently
        self.assertGreaterEqual(
            readme_file_2.blob_last_modified - readme_file_1.blob_last_modified,
            TIME_GAP * 0.9,  # 0.9 factor because not exactly precise
        )
        self.assertGreaterEqual(
            readme_file_2.blob_last_accessed - readme_file_1.blob_last_accessed,
            2 * TIME_GAP * 0.9,  # 0.9 factor because not exactly precise
        )
        self.assertGreaterEqual(
            readme_file_2.blob_last_accessed - readme_file_2.blob_last_modified,
            TIME_GAP * 0.9,  # 0.9 factor because not exactly precise
        )

        # Comparison of last_accessed/last_modified between file and repo
        self.assertEqual(readme_file_2.blob_last_accessed, repo_2.last_accessed)
        self.assertEqual(readme_file_2.blob_last_modified, repo_2.last_modified)
        self.assertEqual(revision_2.last_modified, repo_2.last_modified)


class TestDeleteRevisionsDryRun(unittest.TestCase):
    cache_info: Mock  # Mocked HFCacheInfo

    def setUp(self) -> None:
        """Set up fake cache scan report."""
        repo_A_path = Path("repo_A")
        blobs_path = repo_A_path / "blobs"
        snapshots_path = repo_A_path / "snapshots_path"

        # Define blob files
        main_only_file = Mock()
        main_only_file.blob_path = blobs_path / "main_only_hash"
        main_only_file.size_on_disk = 1

        detached_only_file = Mock()
        detached_only_file.blob_path = blobs_path / "detached_only_hash"
        detached_only_file.size_on_disk = 10

        pr_1_only_file = Mock()
        pr_1_only_file.blob_path = blobs_path / "pr_1_only_hash"
        pr_1_only_file.size_on_disk = 100

        detached_and_pr_1_only_file = Mock()
        detached_and_pr_1_only_file.blob_path = blobs_path / "detached_and_pr_1_only_hash"
        detached_and_pr_1_only_file.size_on_disk = 1000

        shared_file = Mock()
        shared_file.blob_path = blobs_path / "shared_file_hash"
        shared_file.size_on_disk = 10000

        # Define revisions
        repo_A_rev_main = Mock()
        repo_A_rev_main.commit_hash = "repo_A_rev_main"
        repo_A_rev_main.snapshot_path = snapshots_path / "repo_A_rev_main"
        repo_A_rev_main.files = {main_only_file, shared_file}
        repo_A_rev_main.refs = {"main"}

        repo_A_rev_detached = Mock()
        repo_A_rev_detached.commit_hash = "repo_A_rev_detached"
        repo_A_rev_detached.snapshot_path = snapshots_path / "repo_A_rev_detached"
        repo_A_rev_detached.files = {
            detached_only_file,
            detached_and_pr_1_only_file,
            shared_file,
        }
        repo_A_rev_detached.refs = {}

        repo_A_rev_pr_1 = Mock()
        repo_A_rev_pr_1.commit_hash = "repo_A_rev_pr_1"
        repo_A_rev_pr_1.snapshot_path = snapshots_path / "repo_A_rev_pr_1"
        repo_A_rev_pr_1.files = {
            pr_1_only_file,
            detached_and_pr_1_only_file,
            shared_file,
        }
        repo_A_rev_pr_1.refs = {"refs/pr/1"}

        # Define repo
        repo_A = Mock()
        repo_A.repo_path = Path("repo_A")
        repo_A.size_on_disk = 4444
        repo_A.revisions = {repo_A_rev_main, repo_A_rev_detached, repo_A_rev_pr_1}

        # Define cache
        cache_info = Mock()
        cache_info.repos = [repo_A]
        self.cache_info = cache_info

    def test_delete_detached_revision(self) -> None:
        strategy = HFCacheInfo.delete_revisions(self.cache_info, "repo_A_rev_detached")
        expected = DeleteCacheStrategy(
            expected_freed_size=10,
            blobs={
                # "shared_file_hash" and "detached_and_pr_1_only_hash" are not deleted
                Path("repo_A/blobs/detached_only_hash"),
            },
            refs=set(),  # No ref deleted since detached
            repos=set(),  # No repo deleted as other revisions exist
            snapshots={Path("repo_A/snapshots_path/repo_A_rev_detached")},
        )
        self.assertEqual(strategy, expected)

    def test_delete_pr_1_revision(self) -> None:
        strategy = HFCacheInfo.delete_revisions(self.cache_info, "repo_A_rev_pr_1")
        expected = DeleteCacheStrategy(
            expected_freed_size=100,
            blobs={
                # "shared_file_hash" and "detached_and_pr_1_only_hash" are not deleted
                Path("repo_A/blobs/pr_1_only_hash")
            },
            refs={Path("repo_A/refs/refs/pr/1")},  # Ref is deleted !
            repos=set(),  # No repo deleted as other revisions exist
            snapshots={Path("repo_A/snapshots_path/repo_A_rev_pr_1")},
        )
        self.assertEqual(strategy, expected)

    def test_delete_pr_1_and_detached(self) -> None:
        strategy = HFCacheInfo.delete_revisions(self.cache_info, "repo_A_rev_detached", "repo_A_rev_pr_1")
        expected = DeleteCacheStrategy(
            expected_freed_size=1110,
            blobs={
                Path("repo_A/blobs/detached_only_hash"),
                Path("repo_A/blobs/pr_1_only_hash"),
                # blob shared in both revisions and only those two
                Path("repo_A/blobs/detached_and_pr_1_only_hash"),
            },
            refs={Path("repo_A/refs/refs/pr/1")},
            repos=set(),
            snapshots={
                Path("repo_A/snapshots_path/repo_A_rev_detached"),
                Path("repo_A/snapshots_path/repo_A_rev_pr_1"),
            },
        )
        self.assertEqual(strategy, expected)

    def test_delete_all_revisions(self) -> None:
        strategy = HFCacheInfo.delete_revisions(
            self.cache_info, "repo_A_rev_detached", "repo_A_rev_pr_1", "repo_A_rev_main"
        )
        expected = DeleteCacheStrategy(
            expected_freed_size=4444,
            blobs=set(),
            refs=set(),
            repos={Path("repo_A")},  # No remaining revisions: full repo is deleted
            snapshots=set(),
        )
        self.assertEqual(strategy, expected)

    def test_delete_unknown_revision(self) -> None:
        with self.assertLogs() as captured:
            strategy = HFCacheInfo.delete_revisions(self.cache_info, "repo_A_rev_detached", "abcdef123456789")

        # Expected is same strategy as without "abcdef123456789"
        expected = HFCacheInfo.delete_revisions(self.cache_info, "repo_A_rev_detached")
        self.assertEqual(strategy, expected)

        # Expect a warning message
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].levelname, "WARNING")
        self.assertEqual(
            captured.records[0].message,
            "Revision(s) not found - cannot delete them: abcdef123456789",
        )


@pytest.mark.usefixtures("fx_cache_dir")
class TestDeleteStrategyExecute(unittest.TestCase):
    cache_dir: Path

    def test_execute(self) -> None:
        # Repo folders
        repo_A_path = self.cache_dir / "repo_A"
        repo_A_path.mkdir()
        repo_B_path = self.cache_dir / "repo_B"
        repo_B_path.mkdir()

        # Refs files in repo_B
        refs_main_path = repo_B_path / "refs" / "main"
        refs_main_path.parent.mkdir(parents=True)
        refs_main_path.touch()
        refs_pr_1_path = repo_B_path / "refs" / "refs" / "pr" / "1"
        refs_pr_1_path.parent.mkdir(parents=True)
        refs_pr_1_path.touch()

        # Blobs files in repo_B
        (repo_B_path / "blobs").mkdir()
        blob_1 = repo_B_path / "blobs" / "blob_1"
        blob_2 = repo_B_path / "blobs" / "blob_2"
        blob_3 = repo_B_path / "blobs" / "blob_3"
        blob_1.touch()
        blob_2.touch()
        blob_3.touch()

        # Snapshot folders in repo_B
        snapshot_1 = repo_B_path / "snapshots" / "snapshot_1"
        snapshot_2 = repo_B_path / "snapshots" / "snapshot_2"

        snapshot_1.mkdir(parents=True)
        snapshot_2.mkdir()

        # Execute deletion
        # Delete repo_A + keep only blob_1, main ref and snapshot_1 in repo_B.
        DeleteCacheStrategy(
            expected_freed_size=123456,
            blobs={blob_2, blob_3},
            refs={refs_pr_1_path},
            repos={repo_A_path},
            snapshots={snapshot_2},
        ).execute()

        # Repo A deleted
        self.assertFalse(repo_A_path.exists())
        self.assertTrue(repo_B_path.exists())

        # Only `blob` 1 remains
        self.assertTrue(blob_1.exists())
        self.assertFalse(blob_2.exists())
        self.assertFalse(blob_3.exists())

        # Only ref `main` remains
        self.assertTrue(refs_main_path.exists())
        self.assertFalse(refs_pr_1_path.exists())

        # Only `snapshot_1` remains
        self.assertTrue(snapshot_1.exists())
        self.assertFalse(snapshot_2.exists())


@pytest.mark.usefixtures("fx_cache_dir")
class TestTryDeletePath(unittest.TestCase):
    cache_dir: Path

    def test_delete_path_on_file_success(self) -> None:
        """Successfully delete a local file."""
        file_path = self.cache_dir / "file.txt"
        file_path.touch()
        _try_delete_path(file_path, path_type="TYPE")
        self.assertFalse(file_path.exists())

    def test_delete_path_on_folder_success(self) -> None:
        """Successfully delete a local folder."""
        dir_path = self.cache_dir / "something"
        subdir_path = dir_path / "bar"
        subdir_path.mkdir(parents=True)  # subfolder

        file_path_1 = dir_path / "file.txt"  # file at root
        file_path_1.touch()

        file_path_2 = subdir_path / "config.json"  # file in subfolder
        file_path_2.touch()

        _try_delete_path(dir_path, path_type="TYPE")

        self.assertFalse(dir_path.exists())
        self.assertFalse(subdir_path.exists())
        self.assertFalse(file_path_1.exists())
        self.assertFalse(file_path_2.exists())

    def test_delete_path_on_missing_file(self) -> None:
        """Try delete a missing file."""
        file_path = self.cache_dir / "file.txt"

        with self.assertLogs() as captured:
            _try_delete_path(file_path, path_type="TYPE")

        # Assert warning message with traceback for debug purposes
        self.assertEqual(len(captured.output), 1)
        self.assertTrue(
            captured.output[0].startswith(
                "WARNING:huggingface_hub.utils._cache_manager:Couldn't delete TYPE:"
                f" file not found ({file_path})\nTraceback (most recent call last):"
            )
        )

    def test_delete_path_on_missing_folder(self) -> None:
        """Try delete a missing folder."""
        dir_path = self.cache_dir / "folder"

        with self.assertLogs() as captured:
            _try_delete_path(dir_path, path_type="TYPE")

        # Assert warning message with traceback for debug purposes
        self.assertEqual(len(captured.output), 1)
        self.assertTrue(
            captured.output[0].startswith(
                "WARNING:huggingface_hub.utils._cache_manager:Couldn't delete TYPE:"
                f" file not found ({dir_path})\nTraceback (most recent call last):"
            )
        )

    @xfail_on_windows(reason="Permissions are handled differently on Windows.")
    def test_delete_path_on_local_folder_with_wrong_permission(self) -> None:
        """Try delete a local folder that is protected."""
        dir_path = self.cache_dir / "something"
        dir_path.mkdir()
        file_path_1 = dir_path / "file.txt"  # file at root
        file_path_1.touch()
        dir_path.chmod(444)  # Read-only folder

        with self.assertLogs() as captured:
            _try_delete_path(dir_path, path_type="TYPE")

        # Folder still exists (couldn't be deleted)
        self.assertTrue(dir_path.is_dir())

        # Assert warning message with traceback for debug purposes
        self.assertEqual(len(captured.output), 1)
        self.assertTrue(
            captured.output[0].startswith(
                "WARNING:huggingface_hub.utils._cache_manager:Couldn't delete TYPE:"
                f" permission denied ({dir_path})\nTraceback (most recent call last):"
            )
        )

        # For proper cleanup
        dir_path.chmod(509)


class TestStringFormatters(unittest.TestCase):
    SIZES = {
        16.0: "16.0",
        1000.0: "1.0K",
        1024 * 1024 * 1024: "1.1G",  # not 1.0GiB
    }

    SINCE = {
        1: "a few seconds ago",
        15: "a few seconds ago",
        25: "25 seconds ago",
        80: "1 minute ago",
        1000: "17 minutes ago",
        4000: "1 hour ago",
        8000: "2 hours ago",
        3600 * 24 * 13: "2 weeks ago",
        3600 * 24 * 30 * 8.2: "8 months ago",
        3600 * 24 * 365: "1 year ago",
        3600 * 24 * 365 * 9.6: "10 years ago",
    }

    def test_format_size(self) -> None:
        """Test `_format_size` formatter."""
        for size, expected in self.SIZES.items():
            self.assertEqual(
                _format_size(size),
                expected,
                msg=f"Wrong formatting for {size} == '{expected}'",
            )

    def test_format_timesince(self) -> None:
        """Test `_format_timesince` formatter."""
        for ts, expected in self.SINCE.items():
            self.assertEqual(
                _format_timesince(time.time() - ts),
                expected,
                msg=f"Wrong formatting for {ts} == '{expected}'",
            )
