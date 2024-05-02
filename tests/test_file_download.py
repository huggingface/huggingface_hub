# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import io
import os
import re
import shutil
import stat
import unittest
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable
from unittest.mock import Mock, patch

import pytest
import requests
from requests import Response

import huggingface_hub.file_download
from huggingface_hub import HfApi, RepoUrl
from huggingface_hub._local_folder import write_download_metadata
from huggingface_hub.constants import (
    CONFIG_NAME,
    HUGGINGFACE_HEADER_X_LINKED_ETAG,
    PYTORCH_WEIGHTS_NAME,
    REPO_TYPE_DATASET,
)
from huggingface_hub.file_download import (
    _CACHED_NO_EXIST,
    HfFileMetadata,
    _check_disk_space,
    _create_symlink,
    _get_pointer_path,
    _normalize_etag,
    _request_wrapper,
    cached_download,
    filename_to_url,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
    http_get,
    try_to_load_from_cache,
)
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    SoftTemporaryDirectory,
    get_session,
    hf_raise_for_status,
)

from .testing_constants import ENDPOINT_STAGING, OTHER_TOKEN, TOKEN
from .testing_utils import (
    DUMMY_MODEL_ID,
    DUMMY_MODEL_ID_PINNED_SHA1,
    DUMMY_MODEL_ID_PINNED_SHA256,
    DUMMY_MODEL_ID_REVISION_INVALID,
    DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT,
    DUMMY_RENAMED_NEW_MODEL_ID,
    DUMMY_RENAMED_OLD_MODEL_ID,
    SAMPLE_DATASET_IDENTIFIER,
    OfflineSimulationMode,
    expect_deprecation,
    offline,
    repo_name,
    use_tmp_repo,
    with_production_testing,
    xfail_on_windows,
)


REVISION_ID_DEFAULT = "main"
# Default branch name

DATASET_ID = SAMPLE_DATASET_IDENTIFIER
# An actual dataset hosted on huggingface.co


DATASET_REVISION_ID_ONE_SPECIFIC_COMMIT = "e25d55a1c4933f987c46cc75d8ffadd67f257c61"
# One particular commit for DATASET_ID
DATASET_SAMPLE_PY_FILE = "custom_squad.py"


class TestDiskUsageWarning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Test with 100MB expected file size
        cls.expected_size = 100 * 1024 * 1024

    @patch("huggingface_hub.file_download.shutil.disk_usage")
    def test_disk_usage_warning(self, disk_usage_mock: Mock) -> None:
        # Test with only 1MB free disk space / not enough disk space, with UserWarning expected
        disk_usage_mock.return_value.free = 1024 * 1024
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            _check_disk_space(expected_size=self.expected_size, target_dir=disk_usage_mock)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)

        # Test with 200MB free disk space / enough disk space, with no warning expected
        disk_usage_mock.return_value.free = 200 * 1024 * 1024
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            _check_disk_space(expected_size=self.expected_size, target_dir=disk_usage_mock)
            assert len(w) == 0

    def test_disk_usage_warning_with_non_existent_path(self) -> None:
        # Test for not existent (absolute) path
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            _check_disk_space(expected_size=self.expected_size, target_dir="path/to/not_existent_path")
            assert len(w) == 0

        # Test for not existent (relative) path
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            _check_disk_space(expected_size=self.expected_size, target_dir="/path/to/not_existent_path")
            assert len(w) == 0


class StagingDownloadTests(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)

    @use_tmp_repo()
    def test_download_from_a_gated_repo_with_hf_hub_download(self, repo_url: RepoUrl) -> None:
        """Checks `hf_hub_download` outputs error on gated repo.

        Regression test for #1121.
        https://github.com/huggingface/huggingface_hub/pull/1121

        Cannot test on staging as dynamically setting a gated repo doesn't work there.
        """
        # Set repo as gated
        response = get_session().put(
            f"{self._api.endpoint}/api/models/{repo_url.repo_id}/settings",
            json={"gated": "auto"},
            headers=self._api._build_hf_headers(),
        )
        hf_raise_for_status(response)

        # Cannot download file as repo is gated
        with SoftTemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(
                GatedRepoError, "Access to model .* is restricted and you are not in the authorized list"
            ):
                hf_hub_download(
                    repo_id=repo_url.repo_id, filename=".gitattributes", token=OTHER_TOKEN, cache_dir=tmpdir
                )

    @use_tmp_repo()
    def test_download_regular_file_from_private_renamed_repo(self, repo_url: RepoUrl) -> None:
        """Regression test for #1999.

        See https://github.com/huggingface/huggingface_hub/pull/1999.
        """
        repo_id_before = repo_url.repo_id
        repo_id_after = repo_url.repo_id + "_renamed"

        # Make private + rename + upload regular file
        self._api.update_repo_visibility(repo_id_before, private=True)
        self._api.upload_file(repo_id=repo_id_before, path_in_repo="file.txt", path_or_fileobj=b"content")
        self._api.move_repo(repo_id_before, repo_id_after)

        # Download from private renamed repo
        path = self._api.hf_hub_download(repo_id_before, filename="file.txt")
        with open(path) as f:
            self.assertEqual(f.read(), "content")

        # Move back (so that auto-cleanup works)
        self._api.move_repo(repo_id_after, repo_id_before)


@with_production_testing
class CachedDownloadTests(unittest.TestCase):
    def test_bogus_url(self):
        url = "https://bogus"
        with self.assertRaisesRegex(ValueError, "Connection error"):
            _ = cached_download(url, legacy_cache_layout=True)

    def test_no_connection(self):
        invalid_url = hf_hub_url(
            DUMMY_MODEL_ID,
            filename=CONFIG_NAME,
            revision=DUMMY_MODEL_ID_REVISION_INVALID,
        )
        valid_url = hf_hub_url(DUMMY_MODEL_ID, filename=CONFIG_NAME, revision=REVISION_ID_DEFAULT)
        self.assertIsNotNone(cached_download(valid_url, force_download=True, legacy_cache_layout=True))
        for offline_mode in OfflineSimulationMode:
            with offline(mode=offline_mode):
                with self.assertRaisesRegex(ValueError, "Connection error"):
                    _ = cached_download(invalid_url, legacy_cache_layout=True)
                with self.assertRaisesRegex(ValueError, "Connection error"):
                    _ = cached_download(valid_url, force_download=True, legacy_cache_layout=True)
                self.assertIsNotNone(cached_download(valid_url, legacy_cache_layout=True))

    def test_file_not_found_on_repo(self):
        # Valid revision (None) but missing file on repo.
        url = hf_hub_url(DUMMY_MODEL_ID, filename="missing.bin")
        with self.assertRaisesRegex(
            EntryNotFoundError,
            re.compile("404 Client Error(.*)Entry Not Found", flags=re.DOTALL),
        ):
            _ = cached_download(url, legacy_cache_layout=True)

    def test_file_not_found_locally_and_network_disabled(self):
        # Valid file but missing locally and network is disabled.
        with SoftTemporaryDirectory() as tmpdir:
            # Download a first time to get the refs ok
            filepath = hf_hub_download(
                DUMMY_MODEL_ID,
                filename=CONFIG_NAME,
                cache_dir=tmpdir,
                local_files_only=False,
            )

            # Remove local file
            os.remove(filepath)

            # Get without network must fail
            with pytest.raises(LocalEntryNotFoundError):
                hf_hub_download(
                    DUMMY_MODEL_ID,
                    filename=CONFIG_NAME,
                    cache_dir=tmpdir,
                    local_files_only=True,
                )

    def test_file_not_found_locally_and_network_disabled_legacy(self):
        # Valid file but missing locally and network is disabled.
        url = hf_hub_url(DUMMY_MODEL_ID, filename=CONFIG_NAME)
        with SoftTemporaryDirectory() as tmpdir:
            # Get without network must fail
            with pytest.raises(LocalEntryNotFoundError):
                cached_download(
                    url,
                    legacy_cache_layout=True,
                    local_files_only=True,
                    cache_dir=tmpdir,
                )

    def test_private_repo_and_file_cached_locally(self):
        api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
        repo_id = api.create_repo(repo_id=repo_name(), private=True).repo_id
        api.upload_file(path_or_fileobj=b"content", path_in_repo=CONFIG_NAME, repo_id=repo_id)

        with SoftTemporaryDirectory() as tmpdir:
            # Download a first time with token => file is cached
            filepath_1 = hf_hub_download(DUMMY_MODEL_ID, filename=CONFIG_NAME, cache_dir=tmpdir, token=TOKEN)

            # Download without token => return cached file
            filepath_2 = hf_hub_download(DUMMY_MODEL_ID, filename=CONFIG_NAME, cache_dir=tmpdir)

            self.assertEqual(filepath_1, filepath_2)

    def test_file_cached_and_read_only_access(self):
        """Should works if file is already cached and user has read-only permission.

        Regression test for https://github.com/huggingface/huggingface_hub/issues/1216.
        """
        # Valid file but missing locally and network is disabled.
        with SoftTemporaryDirectory() as tmpdir:
            # Download a first time to get the refs ok
            hf_hub_download(DUMMY_MODEL_ID, filename=CONFIG_NAME, cache_dir=tmpdir)

            # Set read-only permission recursively
            _recursive_chmod(tmpdir, 0o555)

            # Get without write-access must succeed
            hf_hub_download(DUMMY_MODEL_ID, filename=CONFIG_NAME, cache_dir=tmpdir)

            # Set permission back for cleanup
            _recursive_chmod(tmpdir, 0o777)

    def test_revision_not_found(self):
        # Valid file but missing revision
        url = hf_hub_url(
            DUMMY_MODEL_ID,
            filename=CONFIG_NAME,
            revision=DUMMY_MODEL_ID_REVISION_INVALID,
        )
        with self.assertRaisesRegex(
            RevisionNotFoundError,
            re.compile("404 Client Error(.*)Revision Not Found", flags=re.DOTALL),
        ):
            _ = cached_download(url, legacy_cache_layout=True)

    def test_repo_not_found(self):
        # Invalid model file.
        url = hf_hub_url("bert-base", filename="pytorch_model.bin")
        with self.assertRaisesRegex(
            RepositoryNotFoundError,
            re.compile("401 Client Error(.*)Repository Not Found", flags=re.DOTALL),
        ):
            _ = cached_download(url, legacy_cache_layout=True)

    def test_standard_object(self):
        url = hf_hub_url(DUMMY_MODEL_ID, filename=CONFIG_NAME, revision=REVISION_ID_DEFAULT)
        filepath = cached_download(url, force_download=True, legacy_cache_layout=True)
        metadata = filename_to_url(filepath, legacy_cache_layout=True)
        self.assertEqual(metadata, (url, f'"{DUMMY_MODEL_ID_PINNED_SHA1}"'))

    def test_standard_object_rev(self):
        # Same object, but different revision
        url = hf_hub_url(
            DUMMY_MODEL_ID,
            filename=CONFIG_NAME,
            revision=DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT,
        )
        filepath = cached_download(url, force_download=True, legacy_cache_layout=True)
        metadata = filename_to_url(filepath, legacy_cache_layout=True)
        self.assertNotEqual(metadata[1], f'"{DUMMY_MODEL_ID_PINNED_SHA1}"')
        # Caution: check that the etag is *not* equal to the one from `test_standard_object`

    def test_lfs_object(self):
        url = hf_hub_url(DUMMY_MODEL_ID, filename=PYTORCH_WEIGHTS_NAME, revision=REVISION_ID_DEFAULT)
        filepath = cached_download(url, force_download=True, legacy_cache_layout=True)
        metadata = filename_to_url(filepath, legacy_cache_layout=True)
        self.assertEqual(metadata, (url, f'"{DUMMY_MODEL_ID_PINNED_SHA256}"'))

    def test_dataset_standard_object_rev(self):
        url = hf_hub_url(
            DATASET_ID,
            filename=DATASET_SAMPLE_PY_FILE,
            repo_type=REPO_TYPE_DATASET,
            revision=DATASET_REVISION_ID_ONE_SPECIFIC_COMMIT,
        )
        # now let's download
        filepath = cached_download(url, force_download=True, legacy_cache_layout=True)
        metadata = filename_to_url(filepath, legacy_cache_layout=True)
        self.assertNotEqual(metadata[1], f'"{DUMMY_MODEL_ID_PINNED_SHA1}"')

    def test_dataset_lfs_object(self):
        url = hf_hub_url(
            DATASET_ID,
            filename="dev-v1.1.json",
            repo_type=REPO_TYPE_DATASET,
            revision=DATASET_REVISION_ID_ONE_SPECIFIC_COMMIT,
        )
        filepath = cached_download(url, force_download=True, legacy_cache_layout=True)
        metadata = filename_to_url(filepath, legacy_cache_layout=True)
        self.assertEqual(
            metadata,
            (url, '"95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"'),
        )

    @xfail_on_windows(reason="umask is UNIX-specific")
    def test_hf_hub_download_custom_cache_permission(self):
        """Checks `hf_hub_download` respect the cache dir permission.

        Regression test for #1141 #1215.
        https://github.com/huggingface/huggingface_hub/issues/1141
        https://github.com/huggingface/huggingface_hub/issues/1215
        """
        with SoftTemporaryDirectory() as tmpdir:
            # Equivalent to umask u=rwx,g=r,o=
            previous_umask = os.umask(0o037)
            try:
                filepath = hf_hub_download(DUMMY_RENAMED_OLD_MODEL_ID, "config.json", cache_dir=tmpdir)
                # Permissions are honored (640: u=rw,g=r,o=)
                self.assertEqual(stat.S_IMODE(os.stat(filepath).st_mode), 0o640)
            finally:
                os.umask(previous_umask)

    def test_download_from_a_renamed_repo_with_hf_hub_download(self):
        """Checks `hf_hub_download` works also on a renamed repo.

        Regression test for #981.
        https://github.com/huggingface/huggingface_hub/issues/981
        """
        with SoftTemporaryDirectory() as tmpdir:
            filepath = hf_hub_download(DUMMY_RENAMED_OLD_MODEL_ID, "config.json", cache_dir=tmpdir)
            self.assertTrue(os.path.exists(filepath))

    def test_download_from_a_renamed_repo_with_cached_download(self):
        """Checks `cached_download` works also on a renamed repo.

        Regression test for #981.
        https://github.com/huggingface/huggingface_hub/issues/981
        """
        with pytest.warns(FutureWarning):
            with SoftTemporaryDirectory() as tmpdir:
                filepath = cached_download(
                    hf_hub_url(
                        DUMMY_RENAMED_OLD_MODEL_ID,
                        filename="config.json",
                    ),
                    cache_dir=tmpdir,
                )
                self.assertTrue(os.path.exists(filepath))

    def test_hf_hub_download_with_empty_subfolder(self):
        """
        Check subfolder arg is processed correctly when empty string is passed to
        `hf_hub_download`.

        See https://github.com/huggingface/huggingface_hub/issues/1016.
        """
        filepath = Path(
            hf_hub_download(
                DUMMY_MODEL_ID,
                filename=CONFIG_NAME,
                subfolder="",  # Subfolder should be processed as `None`
            )
        )

        # Check file exists and is not in a subfolder in cache
        # e.g: "(...)/snapshots/<commit-id>/config.json"
        self.assertTrue(filepath.is_file())
        self.assertEqual(filepath.name, CONFIG_NAME)
        self.assertEqual(Path(filepath).parent.parent.name, "snapshots")

    def test_hf_hub_download_offline_no_refs(self):
        """Regression test for #1305.

        If "refs/" dir did not exists on "local_files_only" (or connection broken), a
        non-explicit `FileNotFoundError` was raised (for the "/refs/revision" file) instead
        of the documented `LocalEntryNotFoundError` (for the actual searched file).

        See https://github.com/huggingface/huggingface_hub/issues/1305.
        """
        with SoftTemporaryDirectory() as cache_dir:
            with self.assertRaises(LocalEntryNotFoundError):
                hf_hub_download(
                    DUMMY_MODEL_ID,
                    filename=CONFIG_NAME,
                    local_files_only=True,
                    cache_dir=cache_dir,
                )

    def test_hf_hub_download_with_user_agent(self):
        """
        Check that user agent is correctly sent to the HEAD call when downloading a file.

        Regression test for #1854.
        See https://github.com/huggingface/huggingface_hub/pull/1854.
        """

        def _check_user_agent(headers: dict):
            assert "user-agent" in headers
            assert "test/1.0.0" in headers["user-agent"]
            assert "foo/bar" in headers["user-agent"]

        with SoftTemporaryDirectory() as cache_dir:
            with patch("huggingface_hub.file_download._request_wrapper", wraps=_request_wrapper) as mock_request:
                # First download
                hf_hub_download(
                    DUMMY_MODEL_ID,
                    filename=CONFIG_NAME,
                    cache_dir=cache_dir,
                    library_name="test",
                    library_version="1.0.0",
                    user_agent="foo/bar",
                )
                calls = mock_request.call_args_list
                assert len(calls) == 3  # HEAD, HEAD, GET
                for call in calls:
                    _check_user_agent(call.kwargs["headers"])

            with patch("huggingface_hub.file_download._request_wrapper", wraps=_request_wrapper) as mock_request:
                # Second download: no GET call
                hf_hub_download(
                    DUMMY_MODEL_ID,
                    filename=CONFIG_NAME,
                    cache_dir=cache_dir,
                    library_name="test",
                    library_version="1.0.0",
                    user_agent="foo/bar",
                )
                calls = mock_request.call_args_list
                assert len(calls) == 2  # HEAD, HEAD
                for call in calls:
                    _check_user_agent(call.kwargs["headers"])

    def test_hf_hub_url_with_empty_subfolder(self):
        """
        Check subfolder arg is processed correctly when empty string is passed to
        `hf_hub_url`.

        See https://github.com/huggingface/huggingface_hub/issues/1016.
        """
        url = hf_hub_url(
            DUMMY_MODEL_ID,
            filename=CONFIG_NAME,
            subfolder="",  # Subfolder should be processed as `None`
        )
        self.assertTrue(
            url.endswith(
                # "./resolve/main/config.json" and not "./resolve/main//config.json"
                f"{DUMMY_MODEL_ID}/resolve/main/config.json",
            )
        )

    @patch("huggingface_hub.file_download.ENDPOINT", "https://huggingface.co")
    @patch(
        "huggingface_hub.file_download.HUGGINGFACE_CO_URL_TEMPLATE",
        "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}",
    )
    def test_hf_hub_url_with_endpoint(self):
        self.assertEqual(
            hf_hub_url(
                DUMMY_MODEL_ID,
                filename=CONFIG_NAME,
                endpoint="https://hf-ci.co",
            ),
            "https://hf-ci.co/julien-c/dummy-unknown/resolve/main/config.json",
        )

    def test_hf_hub_download_legacy(self):
        filepath = hf_hub_download(
            DUMMY_MODEL_ID,
            filename=CONFIG_NAME,
            revision=REVISION_ID_DEFAULT,
            force_download=True,
            legacy_cache_layout=True,
        )
        metadata = filename_to_url(filepath, legacy_cache_layout=True)
        self.assertEqual(metadata[1], f'"{DUMMY_MODEL_ID_PINNED_SHA1}"')

    def test_try_to_load_from_cache_exist(self):
        # Make sure the file is cached
        filepath = hf_hub_download(DUMMY_MODEL_ID, filename=CONFIG_NAME)

        new_file_path = try_to_load_from_cache(DUMMY_MODEL_ID, filename=CONFIG_NAME)
        self.assertEqual(filepath, new_file_path)

        new_file_path = try_to_load_from_cache(DUMMY_MODEL_ID, filename=CONFIG_NAME, revision="main")
        self.assertEqual(filepath, new_file_path)

        # If file is not cached, returns None
        self.assertIsNone(try_to_load_from_cache(DUMMY_MODEL_ID, filename="conf.json"))
        # Same for uncached revisions
        self.assertIsNone(
            try_to_load_from_cache(
                DUMMY_MODEL_ID,
                filename=CONFIG_NAME,
                revision="aaa",
            )
        )
        # Same for uncached models
        self.assertIsNone(try_to_load_from_cache("bert-base", filename=CONFIG_NAME))

    def test_try_to_load_from_cache_specific_pr_revision_exists(self):
        # Make sure the file is cached
        file_path = hf_hub_download(DUMMY_MODEL_ID, filename=CONFIG_NAME, revision="refs/pr/1")

        new_file_path = try_to_load_from_cache(DUMMY_MODEL_ID, filename=CONFIG_NAME, revision="refs/pr/1")
        self.assertEqual(file_path, new_file_path)

        # If file is not cached, returns None
        self.assertIsNone(try_to_load_from_cache(DUMMY_MODEL_ID, filename="conf.json", revision="refs/pr/1"))

        # If revision does not exist, returns None
        self.assertIsNone(try_to_load_from_cache(DUMMY_MODEL_ID, filename=CONFIG_NAME, revision="does-not-exist"))

    def test_try_to_load_from_cache_no_exist(self):
        # Make sure the file is cached
        with self.assertRaises(EntryNotFoundError):
            _ = hf_hub_download(DUMMY_MODEL_ID, filename="dummy")

        new_file_path = try_to_load_from_cache(DUMMY_MODEL_ID, filename="dummy")
        self.assertEqual(new_file_path, _CACHED_NO_EXIST)

        new_file_path = try_to_load_from_cache(DUMMY_MODEL_ID, filename="dummy", revision="main")
        self.assertEqual(new_file_path, _CACHED_NO_EXIST)

        # If file non-existence is not cached, returns None
        self.assertIsNone(try_to_load_from_cache(DUMMY_MODEL_ID, filename="dummy2"))

    def test_try_to_load_from_cache_specific_commit_id_exist(self):
        """Regression test for #1306.

        See https://github.com/huggingface/huggingface_hub/pull/1306."""
        with SoftTemporaryDirectory() as cache_dir:
            # Cache file from specific commit id (no "refs/"" folder)
            commit_id = HfApi().model_info(DUMMY_MODEL_ID).sha
            filepath = hf_hub_download(
                DUMMY_MODEL_ID,
                filename=CONFIG_NAME,
                revision=commit_id,
                cache_dir=cache_dir,
            )

            # Must be able to retrieve it "offline"
            attempt = try_to_load_from_cache(
                DUMMY_MODEL_ID,
                filename=CONFIG_NAME,
                revision=commit_id,
                cache_dir=cache_dir,
            )
            self.assertEqual(filepath, attempt)

    def test_try_to_load_from_cache_specific_commit_id_no_exist(self):
        """Regression test for #1306.

        See https://github.com/huggingface/huggingface_hub/pull/1306."""
        with SoftTemporaryDirectory() as cache_dir:
            # Cache file from specific commit id (no "refs/"" folder)
            commit_id = HfApi().model_info(DUMMY_MODEL_ID).sha
            with self.assertRaises(EntryNotFoundError):
                hf_hub_download(
                    DUMMY_MODEL_ID,
                    filename="missing_file",
                    revision=commit_id,
                    cache_dir=cache_dir,
                )

            # Must be able to retrieve it "offline"
            attempt = try_to_load_from_cache(
                DUMMY_MODEL_ID,
                filename="missing_file",
                revision=commit_id,
                cache_dir=cache_dir,
            )
            self.assertEqual(attempt, _CACHED_NO_EXIST)

    def test_get_hf_file_metadata_basic(self) -> None:
        """Test getting metadata from a file on the Hub."""
        url = hf_hub_url(
            DUMMY_MODEL_ID,
            filename=CONFIG_NAME,
            revision=DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT,
        )
        metadata = get_hf_file_metadata(url)

        # Metadata
        self.assertEqual(metadata.commit_hash, DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT)
        self.assertIsNotNone(metadata.etag)  # example: "85c2fc2dcdd86563aaa85ef4911..."
        self.assertEqual(metadata.location, url)  # no redirect
        self.assertEqual(metadata.size, 851)

    def test_get_hf_file_metadata_from_a_renamed_repo(self) -> None:
        """Test getting metadata from a file in a renamed repo on the Hub."""
        url = hf_hub_url(
            DUMMY_RENAMED_OLD_MODEL_ID,
            filename=CONFIG_NAME,
            subfolder="",  # Subfolder should be processed as `None`
        )
        metadata = get_hf_file_metadata(url)

        # Got redirected to renamed repo
        self.assertEqual(
            metadata.location,
            url.replace(DUMMY_RENAMED_OLD_MODEL_ID, DUMMY_RENAMED_NEW_MODEL_ID),
        )

    def test_get_hf_file_metadata_from_a_lfs_file(self) -> None:
        """Test getting metadata from an LFS file.

        Must get size of the LFS file, not size of the pointer file
        """
        url = hf_hub_url("gpt2", filename="tf_model.h5")
        metadata = get_hf_file_metadata(url)

        self.assertIn("cdn-lfs", metadata.location)  # Redirection
        self.assertEqual(metadata.size, 497933648)  # Size of LFS file, not pointer

    def test_file_consistency_check_fails_regular_file(self):
        """Regression test for #1396 (regular file).

        Download fails if file size is different than the expected one (from headers metadata).

        See https://github.com/huggingface/huggingface_hub/pull/1396."""
        with SoftTemporaryDirectory() as cache_dir:

            def _mocked_hf_file_metadata(*args, **kwargs):
                metadata = get_hf_file_metadata(*args, **kwargs)
                return HfFileMetadata(
                    commit_hash=metadata.commit_hash,
                    etag=metadata.etag,
                    location=metadata.location,
                    size=450,  # will expect 450 bytes but will download 496 bytes
                )

            with patch("huggingface_hub.file_download.get_hf_file_metadata", _mocked_hf_file_metadata):
                with self.assertRaises(EnvironmentError):
                    hf_hub_download(DUMMY_MODEL_ID, filename=CONFIG_NAME, cache_dir=cache_dir)

    def test_file_consistency_check_fails_LFS_file(self):
        """Regression test for #1396 (LFS file).

        Download fails if file size is different than the expected one (from headers metadata).

        See https://github.com/huggingface/huggingface_hub/pull/1396."""
        with SoftTemporaryDirectory() as cache_dir:

            def _mocked_hf_file_metadata(*args, **kwargs):
                metadata = get_hf_file_metadata(*args, **kwargs)
                return HfFileMetadata(
                    commit_hash=metadata.commit_hash,
                    etag=metadata.etag,
                    location=metadata.location,
                    size=65000,  # will expect 65000 bytes but will download 65074 bytes
                )

            with patch("huggingface_hub.file_download.get_hf_file_metadata", _mocked_hf_file_metadata):
                with self.assertRaises(EnvironmentError):
                    hf_hub_download(DUMMY_MODEL_ID, filename="pytorch_model.bin", cache_dir=cache_dir)

    @expect_deprecation("cached_download")
    def test_cached_download_from_github(self):
        """Regression test for #1449.

        File consistency check was failing due to compression in HTTP request which made the expected size smaller than
        the actual one. `cached_download` is deprecated but still heavily used so we need to make sure it works.

        See:
        - https://github.com/huggingface/huggingface_hub/issues/1449.
        - https://github.com/huggingface/diffusers/issues/3213.
        """
        with SoftTemporaryDirectory() as cache_dir:
            cached_download(
                url="https://raw.githubusercontent.com/huggingface/diffusers/v0.15.1/examples/community/lpw_stable_diffusion.py",
                token=None,
                cache_dir=cache_dir,
            )

    @unittest.skipIf(os.name == "nt", "Lock files are always deleted on Windows.")
    def test_keep_lock_file(self):
        """Lock files should not be deleted on Linux."""
        with SoftTemporaryDirectory() as tmpdir:
            hf_hub_download(DUMMY_MODEL_ID, filename=CONFIG_NAME, cache_dir=tmpdir)
            lock_file_exist = False
            locks_dir = os.path.join(tmpdir, ".locks")
            for subdir, dirs, files in os.walk(locks_dir):
                for file in files:
                    if file.endswith(".lock"):
                        lock_file_exist = True
                        break
            self.assertTrue(lock_file_exist, "no lock file can be found")


@pytest.mark.usefixtures("fx_cache_dir")
class HfHubDownloadToLocalDir(unittest.TestCase):
    # `cache_dir` is a temporary directory
    # `local_dir` is a subdirectory in which files will be downloaded
    # `hub_cache_dir` is a subdirectory in which files will be cached ("HF cache")
    cache_dir: Path
    file_name: str = "file.txt"
    lfs_name: str = "lfs.bin"

    @property
    def local_dir(self) -> Path:
        path = Path(self.cache_dir) / "local"
        path.mkdir(exist_ok=True, parents=True)
        return path

    @property
    def hub_cache_dir(self) -> Path:
        path = Path(self.cache_dir) / "cache"
        path.mkdir(exist_ok=True, parents=True)
        return path

    @property
    def file_path(self) -> Path:
        return self.local_dir / self.file_name

    @property
    def lfs_path(self) -> Path:
        return self.local_dir / self.lfs_name

    @classmethod
    def setUpClass(cls):
        cls.api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
        cls.repo_id = cls.api.create_repo(repo_id=repo_name()).repo_id
        commit_1 = cls.api.upload_file(path_or_fileobj=b"content", path_in_repo=cls.file_name, repo_id=cls.repo_id)
        commit_2 = cls.api.upload_file(path_or_fileobj=b"content", path_in_repo=cls.lfs_name, repo_id=cls.repo_id)

        info = cls.api.get_paths_info(repo_id=cls.repo_id, paths=[cls.file_name, cls.lfs_name])
        info = {item.path: item for item in info}
        cls.commit_hash_1 = commit_1.oid
        cls.commit_hash_2 = commit_2.oid
        cls.file_etag = info[cls.file_name].blob_id
        cls.lfs_etag = info[cls.lfs_name].lfs.sha256

    @classmethod
    def tearDownClass(cls) -> None:
        cls.api.delete_repo(repo_id=cls.repo_id)

    @contextmanager
    def with_patch_head(self):
        with patch("huggingface_hub.file_download._get_metadata_or_catch_error") as mock:
            yield mock

    @contextmanager
    def with_patch_download(self):
        with patch("huggingface_hub.file_download._download_to_tmp_and_move") as mock:
            yield mock

    def test_empty_local_dir(self):
        # Download to local dir
        returned_path = self.api.hf_hub_download(
            self.repo_id, filename=self.file_name, cache_dir=self.hub_cache_dir, local_dir=self.local_dir
        )
        assert self.local_dir in Path(returned_path).parents

        # Cache directory not used (no blobs, no symlinks in it)
        for path in self.hub_cache_dir.glob("**/blobs/**"):
            assert not path.is_file()
        for path in self.hub_cache_dir.glob("**/snapshots/**"):
            assert not path.is_file()

    def test_metadata_ok_and_revision_is_a_commit_hash_and_match(self):
        # File already exists + commit_hash matches (and etag not even required)
        self.file_path.write_text("content")
        write_download_metadata(self.local_dir, self.file_name, self.commit_hash_1, etag="...")

        # Download to local dir => no HEAD call needed
        with self.with_patch_head() as mock:
            self.api.hf_hub_download(
                self.repo_id, filename=self.file_name, revision=self.commit_hash_1, local_dir=self.local_dir
            )
        mock.assert_not_called()

    def test_metadata_ok_and_revision_is_a_commit_hash_and_mismatch(self):
        # 1 HEAD call + 1 download
        # File already exists + commit_hash mismatch
        self.file_path.write_text("content")
        write_download_metadata(self.local_dir, self.file_name, self.commit_hash_1, etag="...")

        # Mismatch => download
        with self.with_patch_download() as mock:
            self.api.hf_hub_download(
                self.repo_id, filename=self.file_name, revision=self.commit_hash_2, local_dir=self.local_dir
            )
        mock.assert_called_once()

    def test_metadata_not_ok_and_revision_is_a_commit_hash(self):
        # 1 HEAD call + 1 download
        # File already exists but no metadata
        self.file_path.write_text("content")

        # Mismatch => download
        with self.with_patch_download() as mock:
            self.api.hf_hub_download(
                self.repo_id, filename=self.file_name, revision=self.commit_hash_1, local_dir=self.local_dir
            )
        mock.assert_called_once()

    def test_local_files_only_and_file_exists(self):
        # must return without error
        self.file_path.write_text("content2")

        path = self.api.hf_hub_download(
            self.repo_id, filename=self.file_name, local_dir=self.local_dir, local_files_only=True
        )
        assert Path(path) == self.file_path
        assert self.file_path.read_text() == "content2"  # not overwritten even if wrong content

    def test_local_files_only_and_file_missing(self):
        # must raise
        with self.assertRaises(LocalEntryNotFoundError):
            self.api.hf_hub_download(
                self.repo_id, filename=self.file_name, local_dir=self.local_dir, local_files_only=True
            )

    def test_metadata_ok_and_etag_match(self):
        # 1 HEAD call + return early
        self.file_path.write_text("something")
        write_download_metadata(self.local_dir, self.file_name, self.commit_hash_1, etag=self.file_etag)

        with self.with_patch_download() as mock:
            # Download from main => commit_hash mismatch but etag match => return early
            self.api.hf_hub_download(self.repo_id, filename=self.file_name, local_dir=self.local_dir)
        mock.assert_not_called()

    def test_metadata_ok_and_etag_mismatch(self):
        # 1 HEAD call + 1 download
        self.file_path.write_text("something")
        write_download_metadata(self.local_dir, self.file_name, self.commit_hash_1, etag="some_other_etag")

        with self.with_patch_download() as mock:
            # Download from main => commit_hash mismatch but etag match => return early
            self.api.hf_hub_download(self.repo_id, filename=self.file_name, local_dir=self.local_dir)
        mock.assert_called_once()

    def test_metadata_ok_and_etag_match_and_force_download(self):
        # force_download=True takes precedence on any other rule
        self.file_path.write_text("something")
        write_download_metadata(self.local_dir, self.file_name, self.commit_hash_1, etag=self.file_etag)

        with self.with_patch_download() as mock:
            self.api.hf_hub_download(
                self.repo_id, filename=self.file_name, local_dir=self.local_dir, force_download=True
            )
        mock.assert_called_once()

    def test_metadata_not_ok_and_lfs_file_and_sha256_match(self):
        # 1 HEAD call + 1 hash compute + return early
        self.lfs_path.write_text("content")

        with self.with_patch_download() as mock:
            # Download from main
            # => no metadata but it's an LFS file
            # => compute local hash => matches => return early
            self.api.hf_hub_download(self.repo_id, filename=self.lfs_name, local_dir=self.local_dir)
        mock.assert_not_called()

    def test_metadata_not_ok_and_lfs_file_and_sha256_mismatch(self):
        # 1 HEAD call + 1 file hash + 1 download
        self.lfs_path.write_text("wrong_content")

        # Download from main
        # => no metadata but it's an LFS file
        # => compute local hash => mismatches => download
        path = self.api.hf_hub_download(self.repo_id, filename=self.lfs_name, local_dir=self.local_dir)

        # existing file overwritten
        assert Path(path).read_text() == "content"

    def test_file_exists_in_cache(self):
        # 1 HEAD call + return early
        self.api.hf_hub_download(self.repo_id, filename=self.file_name, cache_dir=self.hub_cache_dir)

        with self.with_patch_download() as mock:
            # Download to local dir
            # => file is already in Hub cache
            # => we assume it's faster to make a local copy rather than re-downloading
            # => duplicate file locally
            path = self.api.hf_hub_download(
                self.repo_id, filename=self.file_name, cache_dir=self.hub_cache_dir, local_dir=self.local_dir
            )
        mock.assert_not_called()

        assert Path(path) == self.file_path

    def test_file_exists_and_overwrites(self):
        # 1 HEAD call + 1 download
        self.file_path.write_text("another content")
        self.api.hf_hub_download(self.repo_id, filename=self.file_name, local_dir=self.local_dir)
        assert self.file_path.read_text() == "content"

    def test_resume_from_incomplete(self):
        # An incomplete file already exists => use it
        incomplete_path = self.local_dir / ".cache" / "huggingface" / "download" / (self.file_name + ".incomplete")
        incomplete_path.parent.mkdir(parents=True, exist_ok=True)
        incomplete_path.write_text("XXXX")  # Here we put fake data to test the resume
        self.api.hf_hub_download(self.repo_id, filename=self.file_name, local_dir=self.local_dir)
        self.file_path.read_text() == "XXXXent"

    def test_do_not_resume_on_force_download(self):
        # An incomplete file already exists but force_download=True
        incomplete_path = self.local_dir / ".cache" / "huggingface" / "download" / (self.file_name + ".incomplete")
        incomplete_path.parent.mkdir(parents=True, exist_ok=True)
        incomplete_path.write_text("XXXX")
        self.api.hf_hub_download(self.repo_id, filename=self.file_name, local_dir=self.local_dir, force_download=True)
        self.file_path.read_text() == "content"


@pytest.mark.usefixtures("fx_cache_dir")
class StagingCachedDownloadOnAwfulFilenamesTest(unittest.TestCase):
    """Implement regression tests for #1161.

    Issue was on filename not url encoded by `hf_hub_download` and `hf_hub_url`.

    See https://github.com/huggingface/huggingface_hub/issues/1161
    """

    cache_dir: Path
    subfolder = "subfolder/to?"
    filename = "awful?filename%you:should,never.give"
    filepath = f"subfolder/to?/{filename}"

    @classmethod
    def setUpClass(cls):
        cls.api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
        cls.repo_url = cls.api.create_repo(repo_id=repo_name("awful_filename"))
        cls.expected_resolve_url = (
            f"{cls.repo_url}/resolve/main/subfolder/to%3F/awful%3Ffilename%25you%3Ashould%2Cnever.give"
        )
        cls.api.upload_file(
            path_or_fileobj=b"content",
            path_in_repo=cls.filepath,
            repo_id=cls.repo_url.repo_id,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.api.delete_repo(repo_id=cls.repo_url.repo_id)

    def test_hf_hub_url_on_awful_filepath(self):
        self.assertEqual(hf_hub_url(self.repo_url.repo_id, self.filepath), self.expected_resolve_url)

    def test_hf_hub_url_on_awful_subfolder_and_filename(self):
        self.assertEqual(
            hf_hub_url(self.repo_url.repo_id, self.filename, subfolder=self.subfolder),
            self.expected_resolve_url,
        )

    @xfail_on_windows(reason="Windows paths cannot contain a '?'.")
    def test_hf_hub_download_on_awful_filepath(self):
        local_path = hf_hub_download(self.repo_url.repo_id, self.filepath, cache_dir=self.cache_dir)
        # Local path is not url-encoded
        self.assertTrue(local_path.endswith(self.filepath))

    @xfail_on_windows(reason="Windows paths cannot contain a '?'.")
    def test_hf_hub_download_on_awful_subfolder_and_filename(self):
        local_path = hf_hub_download(
            self.repo_url.repo_id,
            self.filename,
            subfolder=self.subfolder,
            cache_dir=self.cache_dir,
        )
        # Local path is not url-encoded
        self.assertTrue(local_path.endswith(self.filepath))


@pytest.mark.usefixtures("fx_cache_dir")
class TestHfHubDownloadRelativePaths(unittest.TestCase):
    """Regression test for HackerOne report 1928845.

    Issue was that any file outside of the local dir could be overwritten (Windows only).

    In the end, multiple protections have been added to prevent this (..\\ in filename forbidden on Windows, always check
    the filepath is in local_dir/snapshot_dir).
    """

    cache_dir: Path

    @classmethod
    def setUpClass(cls):
        cls.api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
        cls.repo_id = cls.api.create_repo(repo_id=repo_name()).repo_id
        cls.api.upload_file(path_or_fileobj=b"content", path_in_repo="folder/..\\..\\..\\file", repo_id=cls.repo_id)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.api.delete_repo(repo_id=cls.repo_id)

    @xfail_on_windows(reason="Windows paths cannot contain '\\..\\'.", raises=ValueError)
    def test_download_folder_file_in_cache_dir(self) -> None:
        hf_hub_download(self.repo_id, "folder/..\\..\\..\\file", cache_dir=self.cache_dir)

    @xfail_on_windows(reason="Windows paths cannot contain '\\..\\'.", raises=ValueError)
    def test_download_folder_file_to_local_dir(self) -> None:
        with SoftTemporaryDirectory() as local_dir:
            hf_hub_download(self.repo_id, "folder/..\\..\\..\\file", cache_dir=self.cache_dir, local_dir=local_dir)

    def test_get_pointer_path_and_valid_relative_filename(self) -> None:
        # Cannot happen because of other protections, but just in case.
        self.assertEqual(
            _get_pointer_path("path/to/storage", "abcdef", "path/to/file.txt"),
            os.path.join("path/to/storage", "snapshots", "abcdef", "path/to/file.txt"),
        )

    def test_get_pointer_path_but_invalid_relative_filename(self) -> None:
        # Cannot happen because of other protections, but just in case.
        relative_filename = "folder\\..\\..\\..\\file.txt" if os.name == "nt" else "folder/../../../file.txt"
        with self.assertRaises(ValueError):
            _get_pointer_path("path/to/storage", "abcdef", relative_filename)


class TestHttpGet(unittest.TestCase):
    def test_http_get_with_ssl_and_timeout_error(self):
        def _iter_content_1() -> Iterable[bytes]:
            yield b"0" * 10
            yield b"0" * 10
            raise requests.exceptions.SSLError("Fake SSLError")

        def _iter_content_2() -> Iterable[bytes]:
            yield b"0" * 10
            raise requests.ReadTimeout("Fake ReadTimeout")

        def _iter_content_3() -> Iterable[bytes]:
            yield b"0" * 10
            yield b"0" * 10
            yield b"0" * 10
            raise requests.ConnectionError("Fake ConnectionError")

        def _iter_content_4() -> Iterable[bytes]:
            yield b"0" * 10
            yield b"0" * 10
            yield b"0" * 10
            yield b"0" * 10

        with patch("huggingface_hub.file_download._request_wrapper") as mock:
            mock.return_value.headers = {"Content-Length": 100}
            mock.return_value.iter_content.side_effect = [
                _iter_content_1(),
                _iter_content_2(),
                _iter_content_3(),
                _iter_content_4(),
            ]

            temp_file = io.BytesIO()

            with self.assertLogs("huggingface_hub.file_download", level="WARNING") as records:
                http_get("fake_url", temp_file=temp_file)

        # Check 3 warnings
        self.assertEqual(len(records.records), 3)

        # Check final value
        self.assertEqual(temp_file.tell(), 100)
        self.assertEqual(temp_file.getvalue(), b"0" * 100)

        # Check number of calls + correct range headers
        self.assertEqual(len(mock.call_args_list), 4)
        self.assertEqual(mock.call_args_list[0].kwargs["headers"], {})
        self.assertEqual(mock.call_args_list[1].kwargs["headers"], {"Range": "bytes=20-"})
        self.assertEqual(mock.call_args_list[2].kwargs["headers"], {"Range": "bytes=30-"})
        self.assertEqual(mock.call_args_list[3].kwargs["headers"], {"Range": "bytes=60-"})


class CreateSymlinkTest(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "No symlinks on Windows")
    @patch("huggingface_hub.file_download.are_symlinks_supported")
    def test_create_symlink_concurrent_access(self, mock_are_symlinks_supported: Mock) -> None:
        with SoftTemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "source")
            other = os.path.join(tmpdir, "other")
            dst = os.path.join(tmpdir, "destination")

            # Normal case: symlink does not exist
            mock_are_symlinks_supported.return_value = True
            _create_symlink(src, dst)
            self.assertEqual(os.path.realpath(dst), os.path.realpath(src))

            # Symlink already exists when it tries to create it (most probably from a
            # concurrent access) but do not raise exception
            def _are_symlinks_supported(cache_dir: str) -> bool:
                os.symlink(src, dst)
                return True

            mock_are_symlinks_supported.side_effect = _are_symlinks_supported
            _create_symlink(src, dst)

            # Symlink already exists but pointing to a different source file. This should
            # never happen in the context of HF cache system -> raise exception
            def _are_symlinks_supported(cache_dir: str) -> bool:
                os.symlink(other, dst)
                return True

            mock_are_symlinks_supported.side_effect = _are_symlinks_supported
            with self.assertRaises(FileExistsError):
                _create_symlink(src, dst)

    def test_create_symlink_relative_src(self) -> None:
        """Regression test for #1388.

        See https://github.com/huggingface/huggingface_hub/issues/1388.
        """
        # Test dir has to be relative
        test_dir = Path(".") / "dir_for_create_symlink_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        src = Path(test_dir) / "source"
        src.touch()
        dst = Path(test_dir) / "destination"

        _create_symlink(str(src), str(dst))
        self.assertTrue(dst.resolve().is_file())
        if os.name != "nt":
            self.assertEqual(dst.resolve(), src.resolve())
        shutil.rmtree(test_dir)


class TestNormalizeEtag(unittest.TestCase):
    """Unit tests implemented after a server-side change broke the ETag normalization once (see #1428).

    TL;DR: _normalize_etag was expecting only strong references, but the server started to return weak references after
    a config update. Problem was quickly fixed server-side but we prefer to make sure this doesn't happen again by
    supporting weak etags. For context, etags are used to build the cache-system structure.

    For more details, see https://github.com/huggingface/huggingface_hub/pull/1428 and related issues.
    """

    def test_strong_reference(self):
        self.assertEqual(
            _normalize_etag('"a16a55fda99d2f2e7b69cce5cf93ff4ad3049930"'), "a16a55fda99d2f2e7b69cce5cf93ff4ad3049930"
        )

    def test_weak_reference(self):
        self.assertEqual(
            _normalize_etag('W/"a16a55fda99d2f2e7b69cce5cf93ff4ad3049930"'), "a16a55fda99d2f2e7b69cce5cf93ff4ad3049930"
        )

    @with_production_testing
    def test_resolve_endpoint_on_regular_file(self):
        url = "https://huggingface.co/gpt2/resolve/e7da7f221d5bf496a48136c0cd264e630fe9fcc8/README.md"
        response = requests.head(url)
        self.assertEqual(self._get_etag_and_normalize(response), "a16a55fda99d2f2e7b69cce5cf93ff4ad3049930")

    @with_production_testing
    def test_resolve_endpoint_on_lfs_file(self):
        url = "https://huggingface.co/gpt2/resolve/e7da7f221d5bf496a48136c0cd264e630fe9fcc8/pytorch_model.bin"
        response = requests.head(url)
        self.assertEqual(
            self._get_etag_and_normalize(response), "7c5d3f4b8b76583b422fcb9189ad6c89d5d97a094541ce8932dce3ecabde1421"
        )

    @staticmethod
    def _get_etag_and_normalize(response: Response) -> str:
        response.raise_for_status()
        return _normalize_etag(response.headers.get(HUGGINGFACE_HEADER_X_LINKED_ETAG) or response.headers.get("ETag"))


@with_production_testing
class TestEtagTimeoutConfig(unittest.TestCase):
    @patch("huggingface_hub.file_download.DEFAULT_ETAG_TIMEOUT", 10)
    @patch("huggingface_hub.file_download.HF_HUB_ETAG_TIMEOUT", 10)
    def test_etag_timeout_default_value(self):
        with SoftTemporaryDirectory() as cache_dir:
            with patch.object(
                huggingface_hub.file_download,
                "get_hf_file_metadata",
                wraps=huggingface_hub.file_download.get_hf_file_metadata,
            ) as mock_etag_call:
                hf_hub_download(DUMMY_MODEL_ID, filename=CONFIG_NAME, cache_dir=cache_dir)
                kwargs = mock_etag_call.call_args.kwargs
                self.assertIn("timeout", kwargs)
                self.assertEqual(kwargs["timeout"], 10)

    @patch("huggingface_hub.file_download.DEFAULT_ETAG_TIMEOUT", 10)
    @patch("huggingface_hub.file_download.HF_HUB_ETAG_TIMEOUT", 10)
    def test_etag_timeout_parameter_value(self):
        with SoftTemporaryDirectory() as cache_dir:
            with patch.object(
                huggingface_hub.file_download,
                "get_hf_file_metadata",
                wraps=huggingface_hub.file_download.get_hf_file_metadata,
            ) as mock_etag_call:
                hf_hub_download(DUMMY_MODEL_ID, filename=CONFIG_NAME, cache_dir=cache_dir, etag_timeout=12)
                kwargs = mock_etag_call.call_args.kwargs
                self.assertIn("timeout", kwargs)
                self.assertEqual(kwargs["timeout"], 12)  # passed as parameter, takes priority

    @patch("huggingface_hub.file_download.DEFAULT_ETAG_TIMEOUT", 10)
    @patch("huggingface_hub.file_download.HF_HUB_ETAG_TIMEOUT", 15)  # takes priority
    def test_etag_timeout_set_as_env_variable(self):
        with SoftTemporaryDirectory() as cache_dir:
            with patch.object(
                huggingface_hub.file_download,
                "get_hf_file_metadata",
                wraps=huggingface_hub.file_download.get_hf_file_metadata,
            ) as mock_etag_call:
                hf_hub_download(DUMMY_MODEL_ID, filename=CONFIG_NAME, cache_dir=cache_dir)
                kwargs = mock_etag_call.call_args.kwargs
                self.assertIn("timeout", kwargs)
                self.assertEqual(kwargs["timeout"], 15)

    @patch("huggingface_hub.file_download.DEFAULT_ETAG_TIMEOUT", 10)
    @patch("huggingface_hub.file_download.HF_HUB_ETAG_TIMEOUT", 15)  # takes priority
    def test_etag_timeout_set_as_env_variable_parameter_ignored(self):
        with SoftTemporaryDirectory() as cache_dir:
            with patch.object(
                huggingface_hub.file_download,
                "get_hf_file_metadata",
                wraps=huggingface_hub.file_download.get_hf_file_metadata,
            ) as mock_etag_call:
                hf_hub_download(DUMMY_MODEL_ID, filename=CONFIG_NAME, cache_dir=cache_dir, etag_timeout=12)
                kwargs = mock_etag_call.call_args.kwargs
                self.assertIn("timeout", kwargs)
                self.assertEqual(kwargs["timeout"], 15)  # passed value ignored, HF_HUB_ETAG_TIMEOUT takes priority


def _recursive_chmod(path: str, mode: int) -> None:
    # Taken from https://stackoverflow.com/a/2853934
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), mode)
        for f in files:
            os.chmod(os.path.join(root, f), mode)
