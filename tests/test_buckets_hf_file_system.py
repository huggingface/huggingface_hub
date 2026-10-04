import pytest

from huggingface_hub.hf_file_system import HfFileSystem

from .test_hf_file_system import _HfFileSystemBaseROTests, _HfFileSystemBaseRWTests, _HfFileSystemBucketChecks
from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import repo_name


pytestmark = pytest.mark.xet


class TestHfFileSystemBucketRO(_HfFileSystemBucketChecks, _HfFileSystemBaseROTests):
    __test__ = True

    @pytest.fixture(scope="class", autouse=True)
    def _shared_bucket(self, request):
        api = type(self).api

        # Create dummy bucket
        repo_url = api.create_bucket(repo_name())
        bucket_id = repo_url.bucket_id
        hf_path = f"buckets/{bucket_id}"
        request.cls.bucket_id = bucket_id
        request.cls.hf_path = hf_path

        # Upload files
        api.batch_bucket_files(
            bucket_id,
            add=[
                ("dummy text data".encode("utf-8"), "data/text_data.txt"),
                (b"dummy binary data", "data/binary_data.bin"),
                ("# Dataset card".encode("utf-8"), "README.md"),
            ],
        )

        request.cls.readme_file_path = "README.md"
        request.cls.readme_file = hf_path + "/" + "README.md"
        request.cls.text_file_path = "data/text_data.txt"
        request.cls.text_file = hf_path + "/" + "data/text_data.txt"
        yield
        api.delete_bucket(bucket_id)

    @pytest.fixture(autouse=True)
    def _new_hffs(self):
        self.hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)

    def test_prefix_collision(self):
        colliding_path = f"{self.hf_path}/dat"

        assert not self.hffs.exists(f"{colliding_path}/new.txt")
        assert self.hffs.glob(f"{colliding_path}/*") == []
        with pytest.raises(FileNotFoundError):
            self.hffs.ls(colliding_path)


class TestHfFileSystemBucketRW(_HfFileSystemBucketChecks, _HfFileSystemBaseRWTests):
    __test__ = True

    @pytest.fixture(autouse=True)
    def _bucket(self):
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
        yield
        self.api.delete_bucket(self.bucket_id)

    def test_copy_file(self):
        self.hffs.cp_file(self.text_file, self.hf_path + "/data/text_data_copy.txt")
        with self.hffs.open(self.hf_path + "/data/text_data_copy.txt", "r") as f:
            assert f.read() == "dummy text data"

    def test_move_and_rename_file(self):
        copied_file = self.hf_path + "/data/text_data_copy.txt"
        moved_file = self.hf_path + "/data/text_data_moved.txt"
        renamed_file = self.hf_path + "/data/text_data_renamed.txt"

        self.hffs.cp_file(self.text_file, copied_file)
        self.hffs.mv(copied_file, moved_file)
        assert not self.hffs.exists(copied_file)
        assert self.hffs.exists(moved_file)

        self.hffs.rename(moved_file, renamed_file)
        assert not self.hffs.exists(moved_file)
        with self.hffs.open(renamed_file, "r") as f:
            assert f.read() == "dummy text data"

    def test_copy_between_buckets_is_not_supported(self):
        other_bucket_id = self.api.create_bucket(repo_name()).bucket_id
        try:
            with pytest.raises(NotImplementedError, match="between buckets"):
                self.hffs.cp_file(self.text_file, f"buckets/{other_bucket_id}/copy.txt")
        finally:
            self.api.delete_bucket(other_bucket_id)
