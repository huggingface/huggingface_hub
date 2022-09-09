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
import os
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from huggingface_hub.constants import (
    CONFIG_NAME,
    PYTORCH_WEIGHTS_NAME,
    REPO_TYPE_DATASET,
)
from huggingface_hub.file_download import (
    cached_download,
    filename_to_url,
    hf_hub_download,
    hf_hub_url,
    try_to_load_from_cache,
)
from huggingface_hub.utils import (
    EntryNotFoundError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from .testing_utils import (
    DUMMY_MODEL_ID,
    DUMMY_MODEL_ID_PINNED_SHA1,
    DUMMY_MODEL_ID_PINNED_SHA256,
    DUMMY_MODEL_ID_REVISION_INVALID,
    DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT,
    DUMMY_RENAMED_MODEL_ID,
    SAMPLE_DATASET_IDENTIFIER,
    OfflineSimulationMode,
    offline,
    with_production_testing,
)


REVISION_ID_DEFAULT = "main"
# Default branch name

DATASET_ID = SAMPLE_DATASET_IDENTIFIER
# An actual dataset hosted on huggingface.co


DATASET_REVISION_ID_ONE_SPECIFIC_COMMIT = "e25d55a1c4933f987c46cc75d8ffadd67f257c61"
# One particular commit for DATASET_ID
DATASET_SAMPLE_PY_FILE = "custom_squad.py"


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
        valid_url = hf_hub_url(
            DUMMY_MODEL_ID, filename=CONFIG_NAME, revision=REVISION_ID_DEFAULT
        )
        self.assertIsNotNone(
            cached_download(valid_url, force_download=True, legacy_cache_layout=True)
        )
        for offline_mode in OfflineSimulationMode:
            with offline(mode=offline_mode):
                with self.assertRaisesRegex(ValueError, "Connection error"):
                    _ = cached_download(invalid_url, legacy_cache_layout=True)
                with self.assertRaisesRegex(ValueError, "Connection error"):
                    _ = cached_download(
                        valid_url, force_download=True, legacy_cache_layout=True
                    )
                self.assertIsNotNone(
                    cached_download(valid_url, legacy_cache_layout=True)
                )

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
        with TemporaryDirectory() as tmpdir:
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
        with TemporaryDirectory() as tmpdir:
            # Get without network must fail
            with pytest.raises(LocalEntryNotFoundError):
                cached_download(
                    url,
                    legacy_cache_layout=True,
                    local_files_only=True,
                    cache_dir=tmpdir,
                )

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
        url = hf_hub_url(
            DUMMY_MODEL_ID, filename=CONFIG_NAME, revision=REVISION_ID_DEFAULT
        )
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
        url = hf_hub_url(
            DUMMY_MODEL_ID, filename=PYTORCH_WEIGHTS_NAME, revision=REVISION_ID_DEFAULT
        )
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

    def test_download_from_a_renamed_repo_with_hf_hub_download(self):
        """Checks `hf_hub_download` works also on a renamed repo.

        Regression test for #981.
        https://github.com/huggingface/huggingface_hub/issues/981
        """
        with TemporaryDirectory() as tmpdir:
            filepath = hf_hub_download(
                DUMMY_RENAMED_MODEL_ID, "config.json", cache_dir=tmpdir
            )
            self.assertTrue(os.path.exists(filepath))

    def test_download_from_a_renamed_repo_with_cached_download(self):
        """Checks `cached_download` works also on a renamed repo.

        Regression test for #981.
        https://github.com/huggingface/huggingface_hub/issues/981
        """
        with pytest.warns(FutureWarning):
            with TemporaryDirectory() as tmpdir:
                filepath = cached_download(
                    hf_hub_url(
                        DUMMY_RENAMED_MODEL_ID,
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

    def test_try_to_load_from_cache(self):
        # Make sure the file is cached
        filepath = hf_hub_download(DUMMY_MODEL_ID, filename=CONFIG_NAME)

        new_file_path = try_to_load_from_cache(DUMMY_MODEL_ID, filename=CONFIG_NAME)
        self.assertEqual(filepath, new_file_path)

        new_file_path = try_to_load_from_cache(
            DUMMY_MODEL_ID, filename=CONFIG_NAME, revision="main"
        )
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
