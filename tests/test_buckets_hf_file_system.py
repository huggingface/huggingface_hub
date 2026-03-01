import unittest

from huggingface_hub.hf_file_system import HfFileSystem

from .test_hf_file_system import _HfFileSystemBaseROTests, _HfFileSystemBaseRWTests, _HfFileSystemBucketChecks
from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import repo_name, requires


@requires("hf_xet")
class HfFileSystemBucketROTests(_HfFileSystemBucketChecks, _HfFileSystemBaseROTests):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        super(_HfFileSystemBaseROTests, cls).setUpClass()

        # Create dummy bucket
        repo_url = cls.api.create_bucket(repo_name())
        cls.bucket_id = repo_url.bucket_id
        cls.hf_path = f"buckets/{cls.bucket_id}"

        # Upload files
        cls.api.batch_bucket_files(
            cls.bucket_id,
            add=[
                ("dummy text data".encode("utf-8"), "data/text_data.txt"),
                (b"dummy binary data", "data/binary_data.bin"),
                ("# Dataset card".encode("utf-8"), "README.md"),
            ],
        )

        cls.readme_file_path = "README.md"
        cls.readme_file = cls.hf_path + "/" + cls.readme_file_path
        cls.text_file_path = "data/text_data.txt"
        cls.text_file = cls.hf_path + "/" + cls.text_file_path

    @classmethod
    def tearDownClass(cls):
        super(_HfFileSystemBaseROTests, cls).tearDownClass()
        cls.api.delete_bucket(cls.bucket_id)

    def setUp(self):
        self.hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)


@requires("hf_xet")
class HfFileSystemBucketRWTests(_HfFileSystemBucketChecks, _HfFileSystemBaseRWTests):
    __test__ = True

    def setUp(self):
        self.hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)

        # Create dummy bucket
        repo_url = self.api.create_bucket(repo_name())
        self.bucket_id = repo_url.bucket_id
        self.hf_path = f"buckets/{self.bucket_id}"

        # Upload files
        self.api.batch_bucket_files(
            self.bucket_id,
            add=[
                ("dummy text data".encode("utf-8"), "data/text_data.txt"),
                (b"dummy binary data", "data/binary_data.bin"),
                ("# Dataset card".encode("utf-8"), "README.md"),
            ],
        )

        self.readme_file_path = "README.md"
        self.readme_file = self.hf_path + "/" + self.readme_file_path
        self.text_file_path = "data/text_data.txt"
        self.text_file = self.hf_path + "/" + self.text_file_path

    def tearDown(self):
        self.api.delete_bucket(self.bucket_id)

    @unittest.skip("Not implemented yet")
    def test_copy_file(self):
        pass
