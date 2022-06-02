from hashlib import sha256
from io import BytesIO
import os
from tempfile import TemporaryDirectory

from huggingface_hub.lfs import UploadInfo


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
