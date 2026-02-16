# coding=utf-8
# Copyright 2026-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

from huggingface_hub import BucketInfo, BucketUrl, HfApi
from huggingface_hub.errors import BucketNotFoundError, HfHubHTTPError

from .testing_constants import ENDPOINT_STAGING, ENTERPRISE_ORG, ENTERPRISE_TOKEN, OTHER_TOKEN, TOKEN, USER
from .testing_utils import repo_name


def bucket_name() -> str:
    return repo_name(prefix="bucket")


@pytest.fixture(scope="module")
def api():
    return HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)


@pytest.fixture(scope="module")
def api_other():
    return HfApi(endpoint=ENDPOINT_STAGING, token=OTHER_TOKEN)


@pytest.fixture(scope="module")
def api_enterprise():
    return HfApi(endpoint=ENDPOINT_STAGING, token=ENTERPRISE_TOKEN)


@pytest.fixture(scope="module")
def api_unauth():
    return HfApi(endpoint=ENDPOINT_STAGING, token=False)


def _init_bucket(api: HfApi, bucket_id: str, private: bool = False) -> str:
    bucket = api.create_bucket(bucket_id, private=private)
    api.batch_bucket_files(
        bucket.bucket_id,
        add=[
            (b"content", "file.txt"),
            (b"content", "sub/file.txt"),
            (b"binary", "binary.bin"),
            (b"binary", "sub/binary.bin"),
        ],
    )
    return bucket.bucket_id


@pytest.fixture(scope="module")
def bucket_read(api: HfApi) -> str:
    """Bucket for read-only tests."""
    return _init_bucket(api, bucket_name())


@pytest.fixture(scope="module")
def bucket_read_private(api: HfApi) -> str:
    """Private bucket for read-only tests."""
    return _init_bucket(api, bucket_name(), private=True)


@pytest.fixture(scope="module")
def bucket_read_other(api_other: HfApi) -> str:
    """Bucket for read-only tests with other user."""
    bucket = api_other.create_bucket(bucket_name())
    return bucket.bucket_id


@pytest.fixture(scope="module")
def bucket_read_private_other(api_other: HfApi) -> str:
    """Private bucket for read-only tests with other user."""
    bucket = api_other.create_bucket(bucket_name(), private=True)
    return bucket.bucket_id


@pytest.fixture(scope="function")
def bucket_write(api: HfApi) -> str:
    """Bucket for read-write tests (rebuilt every test)."""
    bucket = api.create_bucket(bucket_name())
    return bucket.bucket_id


def test_create_bucket(api: HfApi):
    bucket_id = f"{USER}/{bucket_name()}"
    bucket_url = api.create_bucket(bucket_id)
    assert bucket_url.bucket_id == bucket_id

    # Cannot create a bucket with the same name
    with pytest.raises(HfHubHTTPError) as exc_info:
        api.create_bucket(bucket_id)
    assert exc_info.value.response.status_code == 409

    # Use exists_ok
    bucket_url_2 = api.create_bucket(bucket_id, exist_ok=True)
    assert bucket_url == bucket_url_2


def test_create_bucket_enterprise_org(api_enterprise: HfApi, api_other: HfApi):
    bucket_id = f"{ENTERPRISE_ORG}/{bucket_name()}"
    bucket_url = api_enterprise.create_bucket(bucket_id)
    assert bucket_url.bucket_id == bucket_id

    # Bucket is private by default in this enterprise org
    bucket = api_enterprise.bucket_info(bucket_id)
    assert bucket.private

    # Cannot access it from other user
    with pytest.raises(HfHubHTTPError):
        api_other.bucket_info(bucket_id)


def test_create_bucket_implicit_namespace(api: HfApi):
    name = bucket_name()
    bucket_url = api.create_bucket(name)
    assert bucket_url.bucket_id == f"{USER}/{name}"


def test_bucket_info(api: HfApi, api_other: HfApi, api_unauth: HfApi, bucket_read: str):
    # Can access bucket
    info = api.bucket_info(bucket_read)
    assert isinstance(info, BucketInfo)
    assert info.id == bucket_read
    assert info.private is False

    # Accessible to other users
    info_other = api_other.bucket_info(bucket_read)
    assert info_other.id == bucket_read
    assert info_other.private is False

    # Accessible to unauthenticated users
    info_unauth = api_unauth.bucket_info(bucket_read)
    assert info_unauth.id == bucket_read
    assert info_unauth.private is False


def test_cannot_bucket_info_with_implicit_namespace(api: HfApi, bucket_read: str):
    with pytest.raises(HfHubHTTPError) as exc_info:
        api.bucket_info(bucket_read.split("/")[1])
    assert exc_info.value.response.status_code == 404


def test_bucket_info_private(api: HfApi, api_other: HfApi, api_unauth: HfApi, bucket_read_private: str):
    info = api.bucket_info(bucket_read_private)
    assert info.id == bucket_read_private
    assert info.private is True

    with pytest.raises(HfHubHTTPError):
        api_other.bucket_info(bucket_read_private)

    with pytest.raises(HfHubHTTPError):
        api_unauth.bucket_info(bucket_read_private)


def test_list_buckets_return_type(api: HfApi, bucket_read: str):
    bucket_ids = set()
    for bucket in api.list_buckets():
        assert isinstance(bucket, BucketInfo)
        bucket_ids.add(bucket.id)
    assert bucket_read in bucket_ids


def test_list_buckets_with_private(
    api: HfApi, api_other: HfApi, api_unauth: HfApi, bucket_read: str, bucket_read_private: str
):
    # List buckets with main user (defaults to "me" namespace)
    bucket_ids = {bucket.id for bucket in api.list_buckets()}
    assert bucket_read in bucket_ids
    assert bucket_read_private in bucket_ids

    # Other user lists their own buckets by default => doesn't see main user's buckets
    bucket_ids_other = {bucket.id for bucket in api_other.list_buckets()}
    assert bucket_read not in bucket_ids_other
    assert bucket_read_private not in bucket_ids_other

    # Other user can list main user's public buckets by passing namespace
    bucket_ids_other_ns = {bucket.id for bucket in api_other.list_buckets(namespace=USER)}
    assert bucket_read in bucket_ids_other_ns
    assert bucket_read_private not in bucket_ids_other_ns

    # Unauthenticated user can list main user's public buckets by passing namespace
    bucket_ids_unauth = {bucket.id for bucket in api_unauth.list_buckets(namespace=USER)}
    assert bucket_read in bucket_ids_unauth
    assert bucket_read_private not in bucket_ids_unauth


def test_delete_bucket(api: HfApi, bucket_write: str):
    api.delete_bucket(bucket_write)

    with pytest.raises(BucketNotFoundError):
        api.bucket_info(bucket_write)


def test_delete_bucket_missing_ok(api: HfApi):
    # Deleting a non-existing bucket should raise 404
    with pytest.raises(BucketNotFoundError):
        api.delete_bucket(f"{USER}/{bucket_name()}")

    # Deleting a non-existing bucket with missing_ok=True should not raise an error
    api.delete_bucket(f"{USER}/{bucket_name()}", missing_ok=True)


def test_delete_bucket_cannot_do_implicit_namespace(api: HfApi):
    with pytest.raises(HfHubHTTPError) as exc_info:
        api.delete_bucket(bucket_name())
    assert exc_info.value.response.status_code == 404


def test_list_bucket_tree_on_public_bucket(api: HfApi, bucket_read: str):
    tree = list(api.list_bucket_tree(bucket_read))
    assert len(tree) == 4

    for entry in tree:
        assert entry.type == "file"
        assert entry.size > 0
        assert entry.xet_hash is not None
        assert entry.mtime is not None

    assert {entry.path for entry in tree} == {"file.txt", "sub/file.txt", "binary.bin", "sub/binary.bin"}


def test_list_bucket_tree_on_private_bucket(api: HfApi, api_other: HfApi, api_unauth: HfApi, bucket_read_private: str):
    assert len(list(api.list_bucket_tree(bucket_read_private))) == 4

    with pytest.raises(BucketNotFoundError):
        list(api_other.list_bucket_tree(bucket_read_private))

    with pytest.raises(HfHubHTTPError) as exc_info:
        list(api_unauth.list_bucket_tree(bucket_read_private))
    assert exc_info.value.response.status_code == 401
