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
    with SliceFileObj(fileobj, seek_from=24, read_limit=18) as fileobj_slice:
        assert fileobj_slice.read() == content[24:42]
        assert fileobj_slice.read() == b""
    assert fileobj.tell() == prev_pos

    with SliceFileObj(fileobj, seek_from=0, read_limit=990) as fileobj_slice:
        assert fileobj_slice.read(200) == content[0:200]
        assert fileobj_slice.read(500) == content[200:700]
        assert fileobj_slice.read(200) == content[700:900]
        assert fileobj_slice.read(200) == content[900:990]
        assert fileobj_slice.read(200) == b""


def test_slice_fileobj_file():
    content = b"RANDOM content uauabciabeubahveb" * 1024

    with TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "file.bin")
        with open(filepath, "wb+") as f:
            f.write(content)
        with open(filepath, "rb") as fileobj:
            prev_pos = fileobj.tell()
            with SliceFileObj(fileobj, seek_from=24, read_limit=18) as fileobj_slice:
                assert fileobj_slice.read() == content[24:42]
                assert fileobj_slice.read() == b""
            assert fileobj.tell() == prev_pos

            with SliceFileObj(fileobj, seek_from=0, read_limit=990) as fileobj_slice:
                assert fileobj_slice.read(200) == content[0:200]
                assert fileobj_slice.read(500) == content[200:700]
                assert fileobj_slice.read(200) == content[700:900]
                assert fileobj_slice.read(200) == content[900:990]
                assert fileobj_slice.read(200) == b""
            assert fileobj.tell() == 0
