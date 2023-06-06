import unittest
from io import SEEK_END
from pathlib import Path

import pytest

from huggingface_hub._commit_scheduler import PartialFileIO
from huggingface_hub import CommitOperationAdd


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
        """Useful for `requests` internally."""
        file = PartialFileIO(self.file_path, size_limit=5)
        self.assertEqual(len(file), 5)

        file = PartialFileIO(self.file_path, size_limit=50)
        self.assertEqual(len(file), 9)

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
