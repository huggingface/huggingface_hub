import os
import subprocess
from hashlib import sha256
from io import BytesIO

from huggingface_hub.utils import SoftTemporaryDirectory
from huggingface_hub.utils.sha import git_hash, sha_fileobj


def test_sha_fileobj():
    with SoftTemporaryDirectory() as tmpdir:
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


def test_git_hash(tmpdir):
    """Test the `git_hash` output is the same as `git hash-object` command."""
    path = os.path.join(tmpdir, "file.txt")
    with open(path, "wb") as file:
        file.write(b"Hello, World!")

    output = subprocess.run(f"git hash-object -t blob {path}", shell=True, capture_output=True, text=True)
    assert output.stdout.strip() == git_hash(b"Hello, World!")
