import pytest

from huggingface_hub.hf_file_system import HfFileSystem

from .test_hf_file_system import _HfFileSystemBaseROTests, _HfFileSystemBaseRWTests, _HfFileSystemBucketChecks
from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import repo_name


pytestmark = pytest.mark.xet

try:
    from hf_xet import XetSession

    HAS_XET_RANGE_UPLOAD = hasattr(XetSession, "new_range_upload")
except ImportError:
    HAS_XET_RANGE_UPLOAD = False

# Append and edit modes rely on xet range uploads, only available in recent versions of `hf_xet`.
requires_xet_range_upload = pytest.mark.skipif(
    not HAS_XET_RANGE_UPLOAD, reason="Append/edit modes require a version of hf_xet supporting range uploads."
)


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

    @requires_xet_range_upload
    def test_append_file(self):
        with self.hffs.open(self.text_file, "ab") as f:
            f.write(b" appended text")

        with self.hffs.open(self.text_file, "r") as f:
            assert f.read() == "dummy text data appended text"

    @requires_xet_range_upload
    def test_edit_file(self):
        with self.hffs.open(self.text_file, "eb") as f:
            f.insert(0, b"this is ")
            f.edit((8, 13), b"a fantastic")
            f.delete(24, 5)
            f.append(b"!")

        with self.hffs.open(self.text_file, "r") as f:
            assert f.read() == "this is a fantastic text!"

    @requires_xet_range_upload
    def test_edit_file_truncate_and_seek(self):
        with self.hffs.open(self.text_file, "eb") as f:
            f.seek(6)
            f.truncate()  # truncate at current location

        with self.hffs.open(self.text_file, "r") as f:
            assert f.read() == "dummy "

        with pytest.raises(ValueError):
            with self.hffs.open(self.text_file, "eb") as f:
                f.seek(1000)  # past end of file

    def test_edit_modes_are_binary_only(self):
        with pytest.raises(NotImplementedError, match="Only binary modes"):
            self.hffs.open(self.text_file, "a")

        with pytest.raises(NotImplementedError, match="Only binary modes"):
            self.hffs.open(self.text_file, "e")
