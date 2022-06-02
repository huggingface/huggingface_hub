from io import BytesIO
import os
from huggingface_hub.utils.sha import sha_fileobj
from hashlib import sha256
import pytest
from tempfile import TemporaryDirectory


def test_sha_fileobj():
    with TemporaryDirectory() as tmpdir:
        content = b"Random content" * 1000
        sha = sha256(content).digest()

        # Test with file object
        filepath = os.path.join(tmpdir, "file.bin")
        with open(filepath, "wb+") as file:
            file.write(content)

        with open(filepath, "rb") as fileobj:
            assert sha_fileobj(fileobj, None) == sha
        with open(filepath, "rb") as fileobj:
            assert sha_fileobj(fileobj, 50) == sha
        with open(filepath, "rb") as fileobj:
            assert sha_fileobj(fileobj, 50_000) == sha

        # Test with in-memory file object
        assert sha_fileobj(BytesIO(content), None) == sha
        assert sha_fileobj(BytesIO(content), 50) == sha
        assert sha_fileobj(BytesIO(content), 50_000) == sha
