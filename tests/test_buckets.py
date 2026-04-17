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
import warnings

import pytest

from huggingface_hub import HfApi
from huggingface_hub._buckets import BucketInfo
from huggingface_hub.errors import BucketNotFoundError, EntryNotFoundError, HfHubHTTPError

from .testing_constants import ENDPOINT_STAGING, ENTERPRISE_ORG, ENTERPRISE_TOKEN, OTHER_TOKEN, TOKEN, USER
from .testing_utils import repo_name, requires


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


@pytest.fixture(scope="function")
def bucket_write_2(api: HfApi) -> str:
    """Second bucket for read-write tests (rebuilt every test)."""
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


def test_move_bucket_rename(api: HfApi, bucket_write: str):
    """Test renaming a bucket within the same namespace."""
    new_bucket_id = f"{USER}/{bucket_name()}"
    api.move_bucket(from_id=bucket_write, to_id=new_bucket_id)

    # New bucket should exist
    info = api.bucket_info(new_bucket_id)
    assert info.id == new_bucket_id

    # Clean up - delete the renamed bucket
    api.delete_bucket(new_bucket_id)


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


@requires("hf_xet")
def test_download_bucket_files_skips_missing_first_file(api: HfApi, bucket_read: str, tmp_path):
    """Test that download_bucket_files works when the first file in the list is missing.

    This is a regression test for a bug where the code used files[0][0] to fetch
    Xet connection metadata, which would fail if that file was missing (and skipped).
    """
    # Request a non-existent file first, followed by a valid file
    files = [
        ("non_existent_file.txt", str(tmp_path / "non_existent.txt")),
        ("file.txt", str(tmp_path / "file.txt")),
    ]

    # Should emit a warning for the missing file but not raise an error
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        api.download_bucket_files(bucket_read, files)

        # Verify warning was issued for missing file
        assert len(w) == 1
        assert "non_existent_file.txt" in str(w[0].message)
        assert "not found" in str(w[0].message).lower()

    # Valid file should be downloaded
    assert (tmp_path / "file.txt").exists()
    assert (tmp_path / "file.txt").read_bytes() == b"content"

    # Missing file should not be created
    assert not (tmp_path / "non_existent.txt").exists()


@requires("hf_xet")
def test_download_bucket_files_raises_on_missing_when_requested(api: HfApi, bucket_read: str, tmp_path):
    """Test that download_bucket_files raises when raise_on_missing_files=True."""
    files = [
        ("non_existent_file.txt", str(tmp_path / "non_existent.txt")),
        ("file.txt", str(tmp_path / "file.txt")),
    ]

    with pytest.raises(EntryNotFoundError) as exc_info:
        api.download_bucket_files(bucket_read, files, raise_on_missing_files=True)

    assert "non_existent_file.txt" in str(exc_info.value)


@requires("hf_xet")
def test_copy_files_bucket_to_same_bucket_file(api: HfApi, bucket_write: str, tmp_path):
    api.batch_bucket_files(bucket_write, add=[(b"bucket-content", "source.txt")])

    api.copy_files(
        f"hf://buckets/{bucket_write}/source.txt",
        f"hf://buckets/{bucket_write}/copied.txt",
    )

    output_path = tmp_path / "copied.txt"
    api.download_bucket_files(bucket_write, [("copied.txt", str(output_path))])
    assert output_path.read_bytes() == b"bucket-content"


@requires("hf_xet")
def test_copy_files_bucket_to_different_bucket_folder(api: HfApi, bucket_write: str, bucket_write_2: str, tmp_path):
    api.batch_bucket_files(bucket_write, add=[(b"a", "logs/a.txt"), (b"b", "logs/sub/b.txt"), (b"c", "other/c.txt")])

    api.copy_files(
        f"hf://buckets/{bucket_write}/logs",
        f"hf://buckets/{bucket_write_2}/backup/",
    )

    destination_files = {entry.path for entry in api.list_bucket_tree(bucket_write_2)}
    assert "backup/a.txt" in destination_files
    assert "backup/sub/b.txt" in destination_files
    assert "backup/c.txt" not in destination_files

    # Check exact content
    a_path = tmp_path / "a.txt"
    b_path = tmp_path / "b.txt"
    api.download_bucket_files(bucket_write_2, [("backup/a.txt", str(a_path)), ("backup/sub/b.txt", str(b_path))])
    assert a_path.read_bytes() == b"a"
    assert b_path.read_bytes() == b"b"


@requires("hf_xet")
def test_copy_files_repo_to_bucket_with_revision(api: HfApi, bucket_write: str, tmp_path):
    repo_id = api.create_repo(repo_id=repo_name(prefix="copy-files")).repo_id
    branch = "copy-files-branch"
    api.upload_file(repo_id=repo_id, path_in_repo="main.txt", path_or_fileobj=b"main")
    api.create_branch(repo_id=repo_id, branch=branch)
    api.upload_file(repo_id=repo_id, path_in_repo="nested/from-branch.txt", path_or_fileobj=b"branch", revision=branch)

    api.copy_files(
        f"hf://{repo_id}@{branch}/nested/from-branch.txt",
        f"hf://buckets/{bucket_write}/from-repo.txt",
    )

    output_path = tmp_path / "from-repo.txt"
    api.download_bucket_files(bucket_write, [("from-repo.txt", str(output_path))])
    assert output_path.read_bytes() == b"branch"


@requires("hf_xet")
def test_copy_files_bucket_to_repo_raises(api: HfApi, bucket_write: str):
    repo_id = api.create_repo(repo_id=repo_name(prefix="copy-files-dst")).repo_id
    api.batch_bucket_files(bucket_write, add=[(b"x", "x.txt")])
    with pytest.raises(ValueError, match="Destination must be a bucket"):
        api.copy_files(f"hf://buckets/{bucket_write}/x.txt", f"hf://{repo_id}/x.txt")


@requires("hf_xet")
def test_copy_files_folder_to_nonexistent_dest(api: HfApi, bucket_write: str, bucket_write_2: str):
    """source=folder, dest doesn't exist => files copied under dest path."""
    api.batch_bucket_files(bucket_write, add=[(b"a", "folder/a.txt"), (b"b", "folder/sub/b.txt")])

    api.copy_files(
        f"hf://buckets/{bucket_write}/folder",
        f"hf://buckets/{bucket_write_2}/target-folder",
    )

    destination_files = {entry.path for entry in api.list_bucket_tree(bucket_write_2)}
    assert "target-folder/a.txt" in destination_files
    assert "target-folder/sub/b.txt" in destination_files


@requires("hf_xet")
def test_copy_files_folder_to_existing_folder_dest(api: HfApi, bucket_write: str, bucket_write_2: str):
    """source=folder, dest is an existing folder => source folder nested under dest (like `cp -r`)."""
    api.batch_bucket_files(bucket_write, add=[(b"a", "folder/a.txt"), (b"b", "folder/sub/b.txt")])
    api.batch_bucket_files(bucket_write_2, add=[(b"existing", "target-folder/existing.txt")])

    api.copy_files(
        f"hf://buckets/{bucket_write}/folder",
        f"hf://buckets/{bucket_write_2}/target-folder",
    )

    # Like `cp -r folder target-folder` when target-folder exists: nests as target-folder/folder/...
    destination_files = {entry.path for entry in api.list_bucket_tree(bucket_write_2)}
    assert "target-folder/existing.txt" in destination_files
    assert "target-folder/folder/a.txt" in destination_files
    assert "target-folder/folder/sub/b.txt" in destination_files


@requires("hf_xet")
def test_copy_files_file_to_existing_file_dest(api: HfApi, bucket_write: str, bucket_write_2: str, tmp_path):
    """source=file, dest is an existing file => must work (overwrite)."""
    api.batch_bucket_files(bucket_write, add=[(b"new-content", "source.txt")])
    api.batch_bucket_files(bucket_write_2, add=[(b"old-content", "dest.txt")])

    api.copy_files(
        f"hf://buckets/{bucket_write}/source.txt",
        f"hf://buckets/{bucket_write_2}/dest.txt",
    )

    output_path = tmp_path / "dest.txt"
    api.download_bucket_files(bucket_write_2, [("dest.txt", str(output_path))])
    assert output_path.read_bytes() == b"new-content"


@requires("hf_xet")
def test_copy_files_file_to_nonexistent_dest(api: HfApi, bucket_write: str, bucket_write_2: str, tmp_path):
    """source=file, dest doesn't exist => must work (creates file)."""
    api.batch_bucket_files(bucket_write, add=[(b"content", "source.txt")])

    api.copy_files(
        f"hf://buckets/{bucket_write}/source.txt",
        f"hf://buckets/{bucket_write_2}/new-file.txt",
    )

    output_path = tmp_path / "new-file.txt"
    api.download_bucket_files(bucket_write_2, [("new-file.txt", str(output_path))])
    assert output_path.read_bytes() == b"content"


@requires("hf_xet")
def test_copy_files_file_to_folder_dest(api: HfApi, bucket_write: str, bucket_write_2: str, tmp_path):
    """source=file, dest is a folder (trailing '/') => file added to folder."""
    api.batch_bucket_files(bucket_write, add=[(b"content", "source.txt")])

    api.copy_files(
        f"hf://buckets/{bucket_write}/source.txt",
        f"hf://buckets/{bucket_write_2}/folder/",
    )

    output_path = tmp_path / "source.txt"
    api.download_bucket_files(bucket_write_2, [("folder/source.txt", str(output_path))])
    assert output_path.read_bytes() == b"content"


@requires("hf_xet")
def test_copy_files_folder_to_existing_folder_with_trailing_slash(api: HfApi, bucket_write: str, bucket_write_2: str):
    """source=folder, dest is existing folder with trailing '/' => source folder nested (like `cp -r`)."""
    api.batch_bucket_files(bucket_write, add=[(b"a", "logs/a.txt"), (b"b", "logs/sub/b.txt")])
    api.batch_bucket_files(bucket_write_2, add=[(b"existing", "backup/existing.txt")])

    api.copy_files(
        f"hf://buckets/{bucket_write}/logs",
        f"hf://buckets/{bucket_write_2}/backup/",
    )

    # Like `cp -r logs backup/` when backup/ exists: nests as backup/logs/...
    destination_files = {entry.path for entry in api.list_bucket_tree(bucket_write_2)}
    assert "backup/existing.txt" in destination_files
    assert "backup/logs/a.txt" in destination_files
    assert "backup/logs/sub/b.txt" in destination_files


@requires("hf_xet")
def test_copy_files_folder_to_nonexistent_dest_with_trailing_slash(api: HfApi, bucket_write: str, bucket_write_2: str):
    """source=folder, dest doesn't exist but has trailing '/' => rename semantics (no nesting)."""
    api.batch_bucket_files(bucket_write, add=[(b"a", "logs/a.txt"), (b"b", "logs/sub/b.txt")])

    api.copy_files(
        f"hf://buckets/{bucket_write}/logs",
        f"hf://buckets/{bucket_write_2}/new-backup/",
    )

    # Like `cp -r logs new-backup/` when new-backup/ doesn't exist:
    # in Unix this errors, but in object storage we create it with rename semantics.
    destination_files = {entry.path for entry in api.list_bucket_tree(bucket_write_2)}
    assert "new-backup/a.txt" in destination_files
    assert "new-backup/sub/b.txt" in destination_files


@requires("hf_xet")
def test_copy_files_folder_to_bucket_root(api: HfApi, bucket_write: str, bucket_write_2: str):
    """source=folder, dest is bucket root => source folder nested at root (like `cp -r models /`)."""
    api.batch_bucket_files(bucket_write, add=[(b"a", "models/a.txt"), (b"b", "models/sub/b.txt")])

    api.copy_files(
        f"hf://buckets/{bucket_write}/models",
        f"hf://buckets/{bucket_write_2}/",
    )

    # Bucket root always "exists" as a directory, so nesting applies
    destination_files = {entry.path for entry in api.list_bucket_tree(bucket_write_2)}
    assert "models/a.txt" in destination_files
    assert "models/sub/b.txt" in destination_files


@pytest.mark.parametrize(
    "source, destination, expected_content_type",
    [
        # Source path determines content type
        ("photo.jpg", "data/img001", "image/jpeg"),
        ("document.pdf", "blob", "application/pdf"),
        # Fallback to destination when source is bytes
        (b"raw", "output.png", "image/png"),
        (b"raw", "data.json", "application/json"),
        # Fallback to destination when source has no extension
        ("no_ext_file", "target.html", "text/html"),
        # Source takes priority over destination
        ("audio.mp3", "target.wav", "audio/mpeg"),
        # None when neither source nor destination have a guessable type
        (b"raw", "blob", None),
        ("no_ext", "no_ext_dest", None),
    ],
)
def test_bucket_add_file_content_type(source, destination, expected_content_type, tmp_path):
    """Test that _BucketAddFile resolves content_type correctly."""
    from huggingface_hub.hf_api import _BucketAddFile

    # If source is a str path, create a temp file so os.path.getmtime works
    if isinstance(source, str):
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
        source = str(path)

    entry = _BucketAddFile(source=source, destination=destination)
    assert entry.content_type == expected_content_type


@requires("hf_xet")
def test_download_file_should_truncate_existing_one(api: HfApi, bucket_write: str, tmp_path):
    """Regression test for  https://github.com/huggingface/huggingface_hub/issues/3995.

    Before this change if the local file was large than the remote one, only the first bytes of the local
    file were updated, leaving a corrupted file.
    """
    file_path = tmp_path / "file.txt"
    file_path.write_text("1234567890")

    # Upload local file by path
    api.batch_bucket_files(bucket_write, add=[(file_path, "file.txt")])

    # Overwrite local file with larger content
    file_path.write_text("a" * 40)

    # Download from bucket should restore original content
    api.download_bucket_files(bucket_write, files=[("file.txt", str(file_path))])
    assert file_path.read_text() == "1234567890"


def test_compute_sync_plan_download_skips_local_walk_without_delete(tmp_path, monkeypatch):
    """When syncing remote -> local with ``delete=False``, the plan must be computed
    by stat-ing only the paths that exist remotely, not by walking the entire dest.

    Walking the dest is the default behavior rsync/aws-cli sync have, but it is a
    large cost when dest happens to sit under a big directory (e.g. HF cache).
    """
    from datetime import datetime as _dt
    from unittest.mock import MagicMock

    from huggingface_hub._buckets import _compute_sync_plan

    (tmp_path / "kept.bin").write_bytes(b"x")  # matches remote
    (tmp_path / "unrelated").mkdir()
    (tmp_path / "unrelated" / "huge_local_file.bin").write_bytes(b"y" * 10)

    remote_file = MagicMock(
        path="kept.bin",
        size=1,
        mtime=_dt(2025, 1, 1),
    )
    # Avoid BucketFolder isinstance match.
    remote_file.__class__ = MagicMock

    api = MagicMock()
    api.list_bucket_tree.return_value = iter([remote_file])
    api.bucket_info.return_value = MagicMock(total_files=1)

    walked: list[str] = []

    def fake_list_local(path):
        walked.append(path)
        return iter([])

    monkeypatch.setattr("huggingface_hub._buckets._list_local_files", fake_list_local)

    plan = _compute_sync_plan(
        source="hf://buckets/ns/bucket",
        dest=str(tmp_path),
        api=api,
        delete=False,
    )

    assert walked == [], "_list_local_files should not be called when delete=False"
    assert {op.path for op in plan.operations} == {"kept.bin"}


def test_compute_sync_plan_download_walks_local_when_delete_is_set(tmp_path, monkeypatch):
    """With ``delete=True`` the full walk is still needed to find local-only files."""
    from unittest.mock import MagicMock

    from huggingface_hub._buckets import _compute_sync_plan

    api = MagicMock()
    api.list_bucket_tree.return_value = iter([])
    api.bucket_info.return_value = MagicMock(total_files=0)

    walked: list[str] = []

    def fake_list_local(path):
        walked.append(path)
        return iter([])

    monkeypatch.setattr("huggingface_hub._buckets._list_local_files", fake_list_local)

    _compute_sync_plan(
        source="hf://buckets/ns/bucket",
        dest=str(tmp_path),
        api=api,
        delete=True,
    )

    assert walked == [str(tmp_path)]
