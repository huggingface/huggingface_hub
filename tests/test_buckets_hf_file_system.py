import time

import pytest

from huggingface_hub.hf_api import HfApi
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

    @pytest.mark.skip("Not implemented yet")
    def test_copy_file(self):
        pass


class TestHfFileSystemBucketLiveFollow:
    """
    Live-following of bucket file changes through `HfFileSystem`.

    The Hub event feed is not served on every deployment (it is disabled on the CI one). Tests therefore only
    assert that changes propagate when the feed is actually followed, and check the graceful fallback
    otherwise.
    """

    api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)

    @pytest.fixture
    def bucket(self):
        bucket_id = self.api.create_bucket(repo_name()).bucket_id
        self.api.batch_bucket_files(bucket_id, add=[(b"data", "data/a.txt")])
        yield bucket_id, f"buckets/{bucket_id}"
        self.api.delete_bucket(bucket_id)

    @pytest.fixture
    def hffs(self):
        fs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True, live_follow=True)
        yield fs
        for follower in fs._bucket_followers.values():
            follower.stop()

    def _followed(self, hffs, bucket_id, timeout=15):
        """Whether the feed of a bucket is actually being followed (the `ls` calls start the follower)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            follower = hffs._bucket_followers.get(bucket_id)
            if follower is not None:
                if follower.subscribed:
                    return True
                follower.join(timeout=0.2)
                if not follower.is_alive():
                    return False  # the follower gave up: this deployment does not serve the feed
            else:
                time.sleep(0.2)
        raise AssertionError("timed out waiting for the live-follow feed of the bucket")

    def test_no_follower_started_without_live_follow(self, bucket):
        bucket_id, hf_path = bucket
        hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN, skip_instance_cache=True)
        assert hffs.ls(hf_path, detail=False) == [f"{hf_path}/data"]
        assert hffs._bucket_followers == {}

    def test_cached_listing_is_refreshed_on_remote_changes(self, hffs, bucket):
        bucket_id, hf_path = bucket
        data_path = f"{hf_path}/data"

        assert hffs.ls(data_path, detail=False) == [f"{data_path}/a.txt"]  # listing is now cached

        if not self._followed(hffs, bucket_id):
            pytest.skip(f"the change feed is not served on {ENDPOINT_STAGING} (its fallback is covered offline)")

        self.api.batch_bucket_files(bucket_id, add=[(b"data", "data/b.txt"), (b"data", "c.txt")])
        self._until(lambda: len(hffs.ls(data_path, detail=False)) == 2)
        assert sorted(hffs.ls(data_path, detail=False)) == [f"{data_path}/a.txt", f"{data_path}/b.txt"]
        assert hffs.exists(f"{hf_path}/c.txt")

        self.api.batch_bucket_files(bucket_id, delete=["data/b.txt", "c.txt"])
        self._until(lambda: len(hffs.ls(data_path, detail=False)) == 1)
        assert hffs.ls(data_path, detail=False) == [f"{data_path}/a.txt"]
        assert not hffs.exists(f"{hf_path}/c.txt")

    def _until(self, predicate, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline and not predicate():
            time.sleep(0.2)
        assert predicate(), "timed out waiting for the live-follow feed to refresh the listing"
