import os
import unittest
from hashlib import sha256
from io import BytesIO
from unittest.mock import MagicMock, patch

from huggingface_hub.lfs import UploadInfo, post_lfs_batch_info
from huggingface_hub.utils import SoftTemporaryDirectory
from huggingface_hub.utils._lfs import SliceFileObj


class TestUploadInfo(unittest.TestCase):
    def setUp(self) -> None:
        self.content = b"RandOm ConTEnT" * 1024
        self.size = len(self.content)
        self.sha = sha256(self.content).digest()
        self.sample = self.content[:512]

    def test_upload_info_from_path(self):
        with SoftTemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "file.bin")
            with open(filepath, "wb+") as file:
                file.write(self.content)
            upload_info = UploadInfo.from_path(filepath)

        self.assertEqual(upload_info.sample, self.sample)
        self.assertEqual(upload_info.size, self.size)
        self.assertEqual(upload_info.sha256, self.sha)

    def test_upload_info_from_bytes(self):
        upload_info = UploadInfo.from_bytes(self.content)

        self.assertEqual(upload_info.sample, self.sample)
        self.assertEqual(upload_info.size, self.size)
        self.assertEqual(upload_info.sha256, self.sha)

    def test_upload_info_from_bytes_io(self):
        upload_info = UploadInfo.from_fileobj(BytesIO(self.content))

        self.assertEqual(upload_info.sample, self.sample)
        self.assertEqual(upload_info.size, self.size)
        self.assertEqual(upload_info.sha256, self.sha)


class TestSliceFileObj(unittest.TestCase):
    def setUp(self) -> None:
        self.content = b"RANDOM self.content uauabciabeubahveb" * 1024

    def test_slice_fileobj_BytesIO(self):
        fileobj = BytesIO(self.content)
        prev_pos = fileobj.tell()

        # Test read
        with SliceFileObj(fileobj, seek_from=24, read_limit=18) as fileobj_slice:
            self.assertEqual(fileobj_slice.tell(), 0)
            self.assertEqual(fileobj_slice.read(), self.content[24:42])
            self.assertEqual(fileobj_slice.tell(), 18)
            self.assertEqual(fileobj_slice.read(), b"")
            self.assertEqual(fileobj_slice.tell(), 18)

        self.assertEqual(fileobj.tell(), prev_pos)

        with SliceFileObj(fileobj, seek_from=0, read_limit=990) as fileobj_slice:
            self.assertEqual(fileobj_slice.tell(), 0)
            self.assertEqual(fileobj_slice.read(200), self.content[0:200])
            self.assertEqual(fileobj_slice.read(500), self.content[200:700])
            self.assertEqual(fileobj_slice.read(200), self.content[700:900])
            self.assertEqual(fileobj_slice.read(200), self.content[900:990])
            self.assertEqual(fileobj_slice.read(200), b"")

        # Test seek with whence = os.SEEK_SET
        with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
            self.assertEqual(fileobj_slice.tell(), 0)
            fileobj_slice.seek(2, os.SEEK_SET)
            self.assertEqual(fileobj_slice.tell(), 2)
            self.assertEqual(fileobj_slice.fileobj.tell(), 102)
            fileobj_slice.seek(-4, os.SEEK_SET)
            self.assertEqual(fileobj_slice.tell(), 0)
            self.assertEqual(fileobj_slice.fileobj.tell(), 100)
            fileobj_slice.seek(100 + 4, os.SEEK_SET)
            self.assertEqual(fileobj_slice.tell(), 100)
            self.assertEqual(fileobj_slice.fileobj.tell(), 200)

        # Test seek with whence = os.SEEK_CUR
        with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
            self.assertEqual(fileobj_slice.tell(), 0)
            fileobj_slice.seek(-5, os.SEEK_CUR)
            self.assertEqual(fileobj_slice.tell(), 0)
            self.assertEqual(fileobj_slice.fileobj.tell(), 100)
            fileobj_slice.seek(50, os.SEEK_CUR)
            self.assertEqual(fileobj_slice.tell(), 50)
            self.assertEqual(fileobj_slice.fileobj.tell(), 150)
            fileobj_slice.seek(100, os.SEEK_CUR)
            self.assertEqual(fileobj_slice.tell(), 100)
            self.assertEqual(fileobj_slice.fileobj.tell(), 200)
            fileobj_slice.seek(-300, os.SEEK_CUR)
            self.assertEqual(fileobj_slice.tell(), 0)
            self.assertEqual(fileobj_slice.fileobj.tell(), 100)

        # Test seek with whence = os.SEEK_END
        with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
            self.assertEqual(fileobj_slice.tell(), 0)
            fileobj_slice.seek(-5, os.SEEK_END)
            self.assertEqual(fileobj_slice.tell(), 95)
            self.assertEqual(fileobj_slice.fileobj.tell(), 195)
            fileobj_slice.seek(50, os.SEEK_END)
            self.assertEqual(fileobj_slice.tell(), 100)
            self.assertEqual(fileobj_slice.fileobj.tell(), 200)
            fileobj_slice.seek(-200, os.SEEK_END)
            self.assertEqual(fileobj_slice.tell(), 0)
            self.assertEqual(fileobj_slice.fileobj.tell(), 100)

    def test_slice_fileobj_file(self):
        self.content = b"RANDOM self.content uauabciabeubahveb" * 1024

        with SoftTemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "file.bin")
            with open(filepath, "wb+") as f:
                f.write(self.content)
            with open(filepath, "rb") as fileobj:
                prev_pos = fileobj.tell()
                # Test read
                with SliceFileObj(fileobj, seek_from=24, read_limit=18) as fileobj_slice:
                    self.assertEqual(fileobj_slice.tell(), 0)
                    self.assertEqual(fileobj_slice.read(), self.content[24:42])
                    self.assertEqual(fileobj_slice.tell(), 18)
                    self.assertEqual(fileobj_slice.read(), b"")
                    self.assertEqual(fileobj_slice.tell(), 18)

                self.assertEqual(fileobj.tell(), prev_pos)

                with SliceFileObj(fileobj, seek_from=0, read_limit=990) as fileobj_slice:
                    self.assertEqual(fileobj_slice.tell(), 0)
                    self.assertEqual(fileobj_slice.read(200), self.content[0:200])
                    self.assertEqual(fileobj_slice.read(500), self.content[200:700])
                    self.assertEqual(fileobj_slice.read(200), self.content[700:900])
                    self.assertEqual(fileobj_slice.read(200), self.content[900:990])
                    self.assertEqual(fileobj_slice.read(200), b"")

                # Test seek with whence = os.SEEK_SET
                with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
                    self.assertEqual(fileobj_slice.tell(), 0)
                    fileobj_slice.seek(2, os.SEEK_SET)
                    self.assertEqual(fileobj_slice.tell(), 2)
                    self.assertEqual(fileobj_slice.fileobj.tell(), 102)
                    fileobj_slice.seek(-4, os.SEEK_SET)
                    self.assertEqual(fileobj_slice.tell(), 0)
                    self.assertEqual(fileobj_slice.fileobj.tell(), 100)
                    fileobj_slice.seek(100 + 4, os.SEEK_SET)
                    self.assertEqual(fileobj_slice.tell(), 100)
                    self.assertEqual(fileobj_slice.fileobj.tell(), 200)

                # Test seek with whence = os.SEEK_CUR
                with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
                    self.assertEqual(fileobj_slice.tell(), 0)
                    fileobj_slice.seek(-5, os.SEEK_CUR)
                    self.assertEqual(fileobj_slice.tell(), 0)
                    self.assertEqual(fileobj_slice.fileobj.tell(), 100)
                    fileobj_slice.seek(50, os.SEEK_CUR)
                    self.assertEqual(fileobj_slice.tell(), 50)
                    self.assertEqual(fileobj_slice.fileobj.tell(), 150)
                    fileobj_slice.seek(100, os.SEEK_CUR)
                    self.assertEqual(fileobj_slice.tell(), 100)
                    self.assertEqual(fileobj_slice.fileobj.tell(), 200)
                    fileobj_slice.seek(-300, os.SEEK_CUR)
                    self.assertEqual(fileobj_slice.tell(), 0)
                    self.assertEqual(fileobj_slice.fileobj.tell(), 100)

                # Test seek with whence = os.SEEK_END
                with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
                    self.assertEqual(fileobj_slice.tell(), 0)
                    fileobj_slice.seek(-5, os.SEEK_END)
                    self.assertEqual(fileobj_slice.tell(), 95)
                    self.assertEqual(fileobj_slice.fileobj.tell(), 195)
                    fileobj_slice.seek(50, os.SEEK_END)
                    self.assertEqual(fileobj_slice.tell(), 100)
                    self.assertEqual(fileobj_slice.fileobj.tell(), 200)
                    fileobj_slice.seek(-200, os.SEEK_END)
                    self.assertEqual(fileobj_slice.tell(), 0)
                    self.assertEqual(fileobj_slice.fileobj.tell(), 100)


@patch("huggingface_hub.lfs.hf_raise_for_status")
@patch("huggingface_hub.lfs.http_backoff")
def test_post_lfs_batch_info_uses_http_backoff(mock_http_backoff, mock_raise_for_status):
    """post_lfs_batch_info uses http_backoff for retry on transient failures."""
    mock_http_backoff.return_value = MagicMock(json=lambda: {"objects": []})

    post_lfs_batch_info(
        upload_infos=[UploadInfo(sha256=b"\x00" * 32, size=100, sample=b"test")],
        token="test_token",
        repo_type="model",
        repo_id="test/repo",
    )

    mock_http_backoff.assert_called_once()
    assert mock_http_backoff.call_args[0][0] == "POST"
    assert "/info/lfs/objects/batch" in mock_http_backoff.call_args[0][1]
