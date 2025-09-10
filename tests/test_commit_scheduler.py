import time
import unittest
from io import SEEK_END
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub._commit_scheduler import CommitScheduler, PartialFileIO

from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import repo_name


@pytest.mark.usefixtures("fx_cache_dir")
class TestCommitScheduler(unittest.TestCase):
    cache_dir: Path

    def setUp(self) -> None:
        self.api = HfApi(token=TOKEN, endpoint=ENDPOINT_STAGING)
        self.repo_name = repo_name()

    def tearDown(self) -> None:
        try:  # try stopping scheduler (if exists)
            self.scheduler.stop()
        except AttributeError:
            pass

        try:  # try delete temporary repo
            self.api.delete_repo(self.repo_name)
        except Exception:
            pass

    @patch("huggingface_hub._commit_scheduler.CommitScheduler.push_to_hub")
    def test_mocked_push_to_hub(self, push_to_hub_mock: MagicMock) -> None:
        self.scheduler = CommitScheduler(
            folder_path=self.cache_dir,
            repo_id=self.repo_name,
            every=1 / 60 / 10,  # every 0.1s
            hf_api=self.api,
        )
        time.sleep(0.5)

        # Triggered at least twice (at 0.0s and then 0.1s, 0.2s,...)
        self.assertGreater(len(push_to_hub_mock.call_args_list), 2)

        # Can get the last upload result
        self.assertEqual(self.scheduler.last_future.result(), push_to_hub_mock.return_value)

    def test_invalid_folder_path_is_a_file(self) -> None:
        """Test cannot scheduler upload of a single file."""
        file_path = self.cache_dir / "file.txt"
        file_path.write_text("something")

        with self.assertRaises(ValueError):
            CommitScheduler(folder_path=file_path, repo_id=self.repo_name, hf_api=self.api)

    def test_missing_folder_is_created(self) -> None:
        folder_path = self.cache_dir / "folder" / "subfolder"
        self.scheduler = CommitScheduler(folder_path=folder_path, repo_id=self.repo_name, hf_api=self.api)
        self.assertTrue(folder_path.is_dir())

    def test_sync_local_folder(self) -> None:
        """Test sync local folder to remote repo."""
        watched_folder = self.cache_dir / "watched_folder"
        hub_cache = self.cache_dir / "hub"  # to download hub files

        file_path = watched_folder / "file.txt"
        lfs_path = watched_folder / "lfs.bin"

        self.scheduler = CommitScheduler(
            folder_path=watched_folder,
            repo_id=self.repo_name,
            every=1 / 60,  # every 1s
            hf_api=self.api,
        )

        # 1 push to hub triggered (empty commit not pushed)
        time.sleep(0.5)

        # write content to files
        with file_path.open("a") as f:
            f.write("first line\n")
        with lfs_path.open("a") as f:
            f.write("binary content")

        # 2 push to hub triggered (1 commit + 1 ignored)
        time.sleep(2)
        self.scheduler.last_future.result()

        # new content in file
        with file_path.open("a") as f:
            f.write("second line\n")

        # 1 push to hub triggered (1 commit)
        time.sleep(1)
        self.scheduler.last_future.result()

        with lfs_path.open("a") as f:
            f.write(" updated")

        # 5 push to hub triggered (1 commit)
        time.sleep(5)  # wait for every threads/uploads to complete
        self.scheduler.stop()
        self.scheduler.last_future.result()

        # 4 commits expected (initial commit + 3 push to hub)
        repo_id = self.scheduler.repo_id
        commits = self.api.list_repo_commits(repo_id)
        self.assertEqual(len(commits), 4)
        push_1 = commits[2].commit_id  # sorted by last first
        push_2 = commits[1].commit_id
        push_3 = commits[0].commit_id

        def _download(filename: str, revision: str) -> Path:
            return Path(hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=hub_cache, revision=revision))

        # Check file.txt consistency
        file_push1 = _download(filename="file.txt", revision=push_1)
        file_push2 = _download(filename="file.txt", revision=push_2)
        file_push3 = _download(filename="file.txt", revision=push_3)

        self.assertEqual(file_push1.read_text(), "first line\n")
        self.assertEqual(file_push2.read_text(), "first line\nsecond line\n")
        self.assertEqual(file_push3.read_text(), "first line\nsecond line\n")

        # Check lfs.bin consistency
        lfs_push1 = _download(filename="lfs.bin", revision=push_1)
        lfs_push2 = _download(filename="lfs.bin", revision=push_2)
        lfs_push3 = _download(filename="lfs.bin", revision=push_3)

        self.assertEqual(lfs_push1.read_text(), "binary content")
        self.assertEqual(lfs_push2.read_text(), "binary content")
        self.assertEqual(lfs_push3.read_text(), "binary content updated")

    def test_sync_and_squash_history(self) -> None:
        """Test squash history when pushing to the Hub."""
        watched_folder = self.cache_dir / "watched_folder"
        watched_folder.mkdir(exist_ok=True, parents=True)
        file_path = watched_folder / "file.txt"
        with file_path.open("a") as f:
            f.write("first line\n")

        self.scheduler = CommitScheduler(
            folder_path=watched_folder,
            repo_id=self.repo_name,
            every=1 / 60 / 10,  # every 0.1s
            hf_api=self.api,
            squash_history=True,
        )

        # At least 1 push to hub triggered
        time.sleep(0.5)
        self.scheduler.stop()
        self.scheduler.last_future.result()

        # Branch history has been squashed
        commits = self.api.list_repo_commits(repo_id=self.scheduler.repo_id)
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0].title, "Super-squash branch 'main' using huggingface_hub")

    def test_context_manager(self) -> None:
        watched_folder = self.cache_dir / "watched_folder"
        watched_folder.mkdir(exist_ok=True, parents=True)
        file_path = watched_folder / "file.txt"

        with CommitScheduler(
            folder_path=watched_folder,
            repo_id=self.repo_name,
            every=5,  # every 5min
            hf_api=self.api,
        ) as scheduler:
            with file_path.open("w") as f:
                f.write("first line\n")

        assert "file.txt" in self.api.list_repo_files(scheduler.repo_id)
        assert scheduler._CommitScheduler__stopped  # means the scheduler has been stopped when exiting the context


@pytest.mark.usefixtures("fx_cache_dir")
class TestPartialFileIO(unittest.TestCase):
    """Test PartialFileIO object."""

    cache_dir: Path

    def setUp(self) -> None:
        """Set up a test file."""
        self.file_path = self.cache_dir / "file.txt"
        self.file_path.write_text("123456789")  # file size: 9 bytes

    def test_read_partial_file_twice(self) -> None:
        file = PartialFileIO(self.file_path, size_limit=5)
        self.assertEqual(file.read(), b"12345")
        self.assertEqual(file.read(), b"")  # End of file

    def test_read_partial_file_by_chunks(self) -> None:
        file = PartialFileIO(self.file_path, size_limit=5)
        self.assertEqual(file.read(2), b"12")
        self.assertEqual(file.read(2), b"34")
        self.assertEqual(file.read(2), b"5")
        self.assertEqual(file.read(2), b"")

    def test_read_partial_file_too_much(self) -> None:
        file = PartialFileIO(self.file_path, size_limit=5)
        self.assertEqual(file.read(20), b"12345")

    def test_partial_file_len(self) -> None:
        """Useful for httpx internally."""
        file = PartialFileIO(self.file_path, size_limit=5)
        self.assertEqual(len(file), 5)

        file = PartialFileIO(self.file_path, size_limit=50)
        self.assertEqual(len(file), 9)

    def test_partial_file_fileno(self) -> None:
        """We explicitly do not implement fileno() to avoid misuse.

        httpx tries to use it to check file size which we don't want for PartialFileIO.
        """
        file = PartialFileIO(self.file_path, size_limit=5)
        with self.assertRaises(AttributeError):
            file.fileno()

    def test_partial_file_seek_and_tell(self) -> None:
        file = PartialFileIO(self.file_path, size_limit=5)

        self.assertEqual(file.tell(), 0)

        file.read(2)
        self.assertEqual(file.tell(), 2)

        file.seek(0)
        self.assertEqual(file.tell(), 0)

        file.seek(2)
        self.assertEqual(file.tell(), 2)

        file.seek(50)
        self.assertEqual(file.tell(), 5)

        file.seek(-3, SEEK_END)
        self.assertEqual(file.tell(), 2)  # 5-3

    def test_methods_not_implemented(self) -> None:
        """Test `PartialFileIO` only implements a subset of the `io` interface. This is on-purpose to avoid misuse."""
        file = PartialFileIO(self.file_path, size_limit=5)

        with self.assertRaises(NotImplementedError):
            file.readline()

        with self.assertRaises(NotImplementedError):
            file.write(b"123")

    def test_append_to_file_then_read(self) -> None:
        file = PartialFileIO(self.file_path, size_limit=9)

        with self.file_path.open("ab") as f:
            f.write(b"abcdef")

        # Output is truncated even if new content appended to the wrapped file
        self.assertEqual(file.read(), b"123456789")

    def test_high_size_limit(self) -> None:
        file = PartialFileIO(self.file_path, size_limit=20)
        with self.file_path.open("ab") as f:
            f.write(b"abcdef")

        # File size limit is truncated to the actual file size at instance creation (not on the fly)
        self.assertEqual(len(file), 9)
        self.assertEqual(file._size_limit, 9)

    def test_with_commit_operation_add(self) -> None:
        # Truncated file
        op_truncated = CommitOperationAdd(
            path_or_fileobj=PartialFileIO(self.file_path, size_limit=5), path_in_repo="file.txt"
        )
        self.assertEqual(op_truncated.upload_info.size, 5)
        self.assertEqual(op_truncated.upload_info.sample, b"12345")

        with op_truncated.as_file() as f:
            self.assertEqual(f.read(), b"12345")

        # Full file
        op_full = CommitOperationAdd(
            path_or_fileobj=PartialFileIO(self.file_path, size_limit=9), path_in_repo="file.txt"
        )
        self.assertEqual(op_full.upload_info.size, 9)
        self.assertEqual(op_full.upload_info.sample, b"123456789")

        with op_full.as_file() as f:
            self.assertEqual(f.read(), b"123456789")

        # Truncated file has a different hash than the full file
        self.assertNotEqual(op_truncated.upload_info.sha256, op_full.upload_info.sha256)
