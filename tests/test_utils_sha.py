import os
import unittest
from hashlib import sha256
from io import BytesIO

from huggingface_hub.utils import SoftTemporaryDirectory
from huggingface_hub.utils.sha import sha_fileobj


class TestShaUtils(unittest.TestCase):
    def test_sha_fileobj(self):
        with SoftTemporaryDirectory() as tmpdir:
            content = b"Random content" * 1000
            sha = sha256(content).digest()

            # Test with file object
            filepath = os.path.join(tmpdir, "file.bin")
            with open(filepath, "wb+") as file:
                file.write(content)

            with open(filepath, "rb") as fileobj:
                self.assertEqual(sha_fileobj(fileobj, None), sha)
            with open(filepath, "rb") as fileobj:
                self.assertEqual(sha_fileobj(fileobj, 50), sha)
            with open(filepath, "rb") as fileobj:
                self.assertEqual(sha_fileobj(fileobj, 50_000), sha)

            # Test with in-memory file object
            self.assertEqual(sha_fileobj(BytesIO(content), None), sha)
            self.assertEqual(sha_fileobj(BytesIO(content), 50), sha)
            self.assertEqual(sha_fileobj(BytesIO(content), 50_000), sha)
