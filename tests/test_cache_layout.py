import os
import tempfile
import time
import unittest

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.utils import logging

from .testing_utils import with_production_testing


logger = logging.get_logger(__name__)
MODEL_IDENTIFIER = "hf-internal-testing/hfh-cache-layout"


@with_production_testing
class CacheFileLayout(unittest.TestCase):
    def test_file_downloaded_in_cache(self):
        with tempfile.TemporaryDirectory() as cache:
            hf_hub_download(MODEL_IDENTIFIER, "file_0.txt", cache_dir=cache)

            expected_directory_name = f'models--{MODEL_IDENTIFIER.replace("/", "--")}'
            expected_path = os.path.join(HUGGINGFACE_HUB_CACHE, expected_directory_name)

            refs = os.listdir(os.path.join(expected_path, "refs"))
            snapshots = os.listdir(os.path.join(expected_path, "snapshots"))

            expected_reference = "main"

            # Only reference should be `main`.
            self.assertListEqual(refs, [expected_reference])

            with open(os.path.join(expected_path, "refs", expected_reference)) as f:
                snapshot_name = f.readline().strip()

            # The `main` reference should point to the only snapshot we have downloaded
            self.assertListEqual(snapshots, [snapshot_name])

            snapshot_path = os.path.join(expected_path, "snapshots", snapshot_name)
            snapshot_content = os.listdir(snapshot_path)

            # Only a single file in the snapshot
            self.assertEqual(len(snapshot_content), 1)

            snapshot_content_path = os.path.join(snapshot_path, snapshot_content[0])

            # The snapshot content should link to a blob
            self.assertTrue(os.path.islink(snapshot_content_path))

            resolved_blob_relative = os.readlink(snapshot_content_path)
            resolved_blob_absolute = os.path.normpath(
                os.path.join(snapshot_path, resolved_blob_relative)
            )

            with open(resolved_blob_absolute) as f:
                blob_contents = f.readline().strip()

            # The contents of the file should be 'File 0'.
            self.assertEqual(blob_contents, "File 0")

    def test_file_downloaded_in_cache_with_revision(self):
        with tempfile.TemporaryDirectory() as cache:
            hf_hub_download(
                MODEL_IDENTIFIER, "file_0.txt", cache_dir=cache, revision="file-2"
            )

            expected_directory_name = f'models--{MODEL_IDENTIFIER.replace("/", "--")}'
            expected_path = os.path.join(cache, expected_directory_name)

            refs = os.listdir(os.path.join(expected_path, "refs"))
            snapshots = os.listdir(os.path.join(expected_path, "snapshots"))

            expected_reference = "file-2"

            # Only reference should be `file-2`.
            self.assertListEqual(refs, [expected_reference])

            with open(os.path.join(expected_path, "refs", expected_reference)) as f:
                snapshot_name = f.readline().strip()

            # The `main` reference should point to the only snapshot we have downloaded
            self.assertListEqual(snapshots, [snapshot_name])

            snapshot_path = os.path.join(expected_path, "snapshots", snapshot_name)
            snapshot_content = os.listdir(snapshot_path)

            # Only a single file in the snapshot
            self.assertEqual(len(snapshot_content), 1)

            snapshot_content_path = os.path.join(snapshot_path, snapshot_content[0])

            # The snapshot content should link to a blob
            self.assertTrue(os.path.islink(snapshot_content_path))

            resolved_blob_relative = os.readlink(snapshot_content_path)
            resolved_blob_absolute = os.path.normpath(
                os.path.join(snapshot_path, resolved_blob_relative)
            )

            with open(resolved_blob_absolute) as f:
                blob_contents = f.readline().strip()

            # The contents of the file should be 'File 0'.
            self.assertEqual(blob_contents, "File 0")

    def test_file_download_happens_once(self):
        # Tests that a file is only downloaded once if it's not updated.

        with tempfile.TemporaryDirectory() as cache:
            path = hf_hub_download(MODEL_IDENTIFIER, "file_0.txt", cache_dir=cache)
            creation_time_0 = os.path.getmtime(path)

            time.sleep(2)

            path = hf_hub_download(MODEL_IDENTIFIER, "file_0.txt", cache_dir=cache)
            creation_time_1 = os.path.getmtime(path)

            self.assertEqual(creation_time_0, creation_time_1)

    def test_file_download_happens_once_intra_revision(self):
        # Tests that a file is only downloaded once if it's not updated, even across different revisions.

        with tempfile.TemporaryDirectory() as cache:
            path = hf_hub_download(MODEL_IDENTIFIER, "file_0.txt", cache_dir=cache)
            creation_time_0 = os.path.getmtime(path)

            time.sleep(2)

            path = hf_hub_download(
                MODEL_IDENTIFIER, "file_0.txt", cache_dir=cache, revision="file-2"
            )
            creation_time_1 = os.path.getmtime(path)

            self.assertEqual(creation_time_0, creation_time_1)

    def test_multiple_refs_for_same_file(self):
        with tempfile.TemporaryDirectory() as cache:
            hf_hub_download(MODEL_IDENTIFIER, "file_0.txt", cache_dir=cache)
            hf_hub_download(
                MODEL_IDENTIFIER, "file_0.txt", cache_dir=cache, revision="file-2"
            )

            expected_directory_name = f'models--{MODEL_IDENTIFIER.replace("/", "--")}'
            expected_path = os.path.join(cache, expected_directory_name)

            refs = os.listdir(os.path.join(expected_path, "refs"))
            refs.sort()

            snapshots = os.listdir(os.path.join(expected_path, "snapshots"))
            snapshots.sort()

            # Directory should contain two revisions
            self.assertListEqual(refs, ["file-2", "main"])

            def get_file_contents(path):
                with open(path) as f:
                    content = f.read()

                return content

            refs_contents = [
                get_file_contents(os.path.join(expected_path, "refs", f)) for f in refs
            ]
            refs_contents.sort()

            # snapshots directory should contain two snapshots
            self.assertListEqual(refs_contents, snapshots)

            # snapshots_paths = [os.path.join(expected_path, s) for s in snapshots]
