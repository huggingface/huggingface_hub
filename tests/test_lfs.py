import os
from hashlib import sha256
from io import BytesIO
from tempfile import TemporaryDirectory

from huggingface_hub.lfs import SliceFileObj, UploadInfo


def test_upload_info_from_path():
    content = b"RandOm ConTEnT" * 1024
    size = len(content)
    sha = sha256(content).digest()
    sample = content[:512]

    with TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "file.bin")
        with open(filepath, "wb+") as file:
            file.write(content)

        upload_info = UploadInfo.from_path(filepath)

    assert upload_info.sample == sample
    assert upload_info.size == size
    assert upload_info.sha256 == sha


def test_upload_info_from_bytes():
    content = b"RandOm ConTEnT" * 1024
    size = len(content)
    sha = sha256(content).digest()
    sample = content[:512]

    upload_info = UploadInfo.from_bytes(content)

    assert upload_info.sample == sample
    assert upload_info.size == size
    assert upload_info.sha256 == sha


def test_upload_info_from_fileobj():
    content = b"RandOm ConTEnT" * 1024
    size = len(content)
    sha = sha256(content).digest()
    sample = content[:512]

    upload_info = UploadInfo.from_fileobj(BytesIO(content))

    assert upload_info.sample == sample
    assert upload_info.size == size
    assert upload_info.sha256 == sha


def test_slice_fileobj_BytesIO():
    content = b"RANDOM content uauabciabeubahveb" * 1024

    fileobj = BytesIO(content)
    prev_pos = fileobj.tell()

    # Test read
    with SliceFileObj(fileobj, seek_from=24, read_limit=18) as fileobj_slice:
        assert fileobj_slice.tell() == 0
        assert fileobj_slice.read() == content[24:42]
        assert fileobj_slice.tell() == 18
        assert fileobj_slice.read() == b""
        assert fileobj_slice.tell() == 18

    assert fileobj.tell() == prev_pos

    with SliceFileObj(fileobj, seek_from=0, read_limit=990) as fileobj_slice:
        assert fileobj_slice.tell() == 0
        assert fileobj_slice.read(200) == content[0:200]
        assert fileobj_slice.read(500) == content[200:700]
        assert fileobj_slice.read(200) == content[700:900]
        assert fileobj_slice.read(200) == content[900:990]
        assert fileobj_slice.read(200) == b""

    # Test seek with whence = os.SEEK_SET
    with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
        assert fileobj_slice.tell() == 0
        fileobj_slice.seek(2, os.SEEK_SET)
        assert fileobj_slice.tell() == 2
        assert fileobj_slice.fileobj.tell() == 102
        fileobj_slice.seek(-4, os.SEEK_SET)
        assert fileobj_slice.tell() == 0
        assert fileobj_slice.fileobj.tell() == 100
        fileobj_slice.seek(100 + 4, os.SEEK_SET)
        assert fileobj_slice.tell() == 100
        assert fileobj_slice.fileobj.tell() == 200

    # Test seek with whence = os.SEEK_CUR
    with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
        assert fileobj_slice.tell() == 0
        fileobj_slice.seek(-5, os.SEEK_CUR)
        assert fileobj_slice.tell() == 0
        assert fileobj_slice.fileobj.tell() == 100
        fileobj_slice.seek(50, os.SEEK_CUR)
        assert fileobj_slice.tell() == 50
        assert fileobj_slice.fileobj.tell() == 150
        fileobj_slice.seek(100, os.SEEK_CUR)
        assert fileobj_slice.tell() == 100
        assert fileobj_slice.fileobj.tell() == 200
        fileobj_slice.seek(-300, os.SEEK_CUR)
        assert fileobj_slice.tell() == 0
        assert fileobj_slice.fileobj.tell() == 100

    # Test seek with whence = os.SEEK_END
    with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
        assert fileobj_slice.tell() == 0
        fileobj_slice.seek(-5, os.SEEK_END)
        assert fileobj_slice.tell() == 95
        assert fileobj_slice.fileobj.tell() == 195
        fileobj_slice.seek(50, os.SEEK_END)
        assert fileobj_slice.tell() == 100
        assert fileobj_slice.fileobj.tell() == 200
        fileobj_slice.seek(-200, os.SEEK_END)
        assert fileobj_slice.tell() == 0
        assert fileobj_slice.fileobj.tell() == 100


def test_slice_fileobj_file():
    content = b"RANDOM content uauabciabeubahveb" * 1024

    with TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "file.bin")
        with open(filepath, "wb+") as f:
            f.write(content)
        with open(filepath, "rb") as fileobj:
            prev_pos = fileobj.tell()
            # Test read
            with SliceFileObj(fileobj, seek_from=24, read_limit=18) as fileobj_slice:
                assert fileobj_slice.tell() == 0
                assert fileobj_slice.read() == content[24:42]
                assert fileobj_slice.tell() == 18
                assert fileobj_slice.read() == b""
                assert fileobj_slice.tell() == 18

            assert fileobj.tell() == prev_pos

            with SliceFileObj(fileobj, seek_from=0, read_limit=990) as fileobj_slice:
                assert fileobj_slice.tell() == 0
                assert fileobj_slice.read(200) == content[0:200]
                assert fileobj_slice.read(500) == content[200:700]
                assert fileobj_slice.read(200) == content[700:900]
                assert fileobj_slice.read(200) == content[900:990]
                assert fileobj_slice.read(200) == b""

            # Test seek with whence = os.SEEK_SET
            with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
                assert fileobj_slice.tell() == 0
                fileobj_slice.seek(2, os.SEEK_SET)
                assert fileobj_slice.tell() == 2
                assert fileobj_slice.fileobj.tell() == 102
                fileobj_slice.seek(-4, os.SEEK_SET)
                assert fileobj_slice.tell() == 0
                assert fileobj_slice.fileobj.tell() == 100
                fileobj_slice.seek(100 + 4, os.SEEK_SET)
                assert fileobj_slice.tell() == 100
                assert fileobj_slice.fileobj.tell() == 200

            # Test seek with whence = os.SEEK_CUR
            with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
                assert fileobj_slice.tell() == 0
                fileobj_slice.seek(-5, os.SEEK_CUR)
                assert fileobj_slice.tell() == 0
                assert fileobj_slice.fileobj.tell() == 100
                fileobj_slice.seek(50, os.SEEK_CUR)
                assert fileobj_slice.tell() == 50
                assert fileobj_slice.fileobj.tell() == 150
                fileobj_slice.seek(100, os.SEEK_CUR)
                assert fileobj_slice.tell() == 100
                assert fileobj_slice.fileobj.tell() == 200
                fileobj_slice.seek(-300, os.SEEK_CUR)
                assert fileobj_slice.tell() == 0
                assert fileobj_slice.fileobj.tell() == 100

            # Test seek with whence = os.SEEK_END
            with SliceFileObj(fileobj, seek_from=100, read_limit=100) as fileobj_slice:
                assert fileobj_slice.tell() == 0
                fileobj_slice.seek(-5, os.SEEK_END)
                assert fileobj_slice.tell() == 95
                assert fileobj_slice.fileobj.tell() == 195
                fileobj_slice.seek(50, os.SEEK_END)
                assert fileobj_slice.tell() == 100
                assert fileobj_slice.fileobj.tell() == 200
                fileobj_slice.seek(-200, os.SEEK_END)
                assert fileobj_slice.tell() == 0
                assert fileobj_slice.fileobj.tell() == 100
