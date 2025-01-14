import os
import time
import unittest
from io import BytesIO

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.utils import SoftTemporaryDirectory, logging

from .testing_constants import ENDPOINT_STAGING, TOKEN
from .testing_utils import (
    repo_name,
    with_production_testing,
    xfail_on_windows,
)


logger = logging.get_logger(__name__)
MODEL_IDENTIFIER = "hf-internal-testing/hfh-cache-layout"


def get_file_contents(path):
    with open(path) as f:
        content = f.read()

    return content


@with_production_testing
class CacheFileLayoutHfHubDownload(unittest.TestCase):
    @xfail_on_windows(reason="Symlinks are deactivated in Windows tests.")
    def test_file_downloaded_in_cache(self):
        for revision, expected_reference in (
            (None, "main"),
            ("file-2", "file-2"),
        ):
            with self.subTest(revision), SoftTemporaryDirectory() as cache:
                hf_hub_download(
                    MODEL_IDENTIFIER,
                    "file_0.txt",
                    cache_dir=cache,
                    revision=revision,
                )

                expected_directory_name = f"models--{MODEL_IDENTIFIER.replace('/', '--')}"
                expected_path = os.path.join(cache, expected_directory_name)

                refs = os.listdir(os.path.join(expected_path, "refs"))
                snapshots = os.listdir(os.path.join(expected_path, "snapshots"))

                # Only reference should be the expected one.
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
                resolved_blob_absolute = os.path.normpath(os.path.join(snapshot_path, resolved_blob_relative))

                with open(resolved_blob_absolute) as f:
                    blob_contents = f.readline().strip()

                # The contents of the file should be 'File 0'.
                self.assertEqual(blob_contents, "File 0")

    def test_no_exist_file_is_cached(self):
        revisions = [None, "file-2"]
        expected_references = ["main", "file-2"]
        for revision, expected_reference in zip(revisions, expected_references):
            with self.subTest(revision), SoftTemporaryDirectory() as cache:
                filename = "this_does_not_exist.txt"
                with self.assertRaises(EntryNotFoundError):
                    # The file does not exist, so we get an exception.
                    hf_hub_download(MODEL_IDENTIFIER, filename, cache_dir=cache, revision=revision)

                expected_directory_name = f"models--{MODEL_IDENTIFIER.replace('/', '--')}"
                expected_path = os.path.join(cache, expected_directory_name)

                refs = os.listdir(os.path.join(expected_path, "refs"))
                no_exist_snapshots = os.listdir(os.path.join(expected_path, ".no_exist"))

                # Only reference should be `main`.
                self.assertListEqual(refs, [expected_reference])

                with open(os.path.join(expected_path, "refs", expected_reference)) as f:
                    snapshot_name = f.readline().strip()

                # The `main` reference should point to the only snapshot we have downloaded
                self.assertListEqual(no_exist_snapshots, [snapshot_name])

                no_exist_path = os.path.join(expected_path, ".no_exist", snapshot_name)
                no_exist_content = os.listdir(no_exist_path)

                # Only a single file in the no_exist snapshot
                self.assertEqual(len(no_exist_content), 1)

                # The no_exist content should be our file
                self.assertEqual(no_exist_content[0], filename)

                with open(os.path.join(no_exist_path, filename)) as f:
                    content = f.read().strip()

                # The contents of the file should be empty.
                self.assertEqual(content, "")

    def test_file_download_happens_once(self):
        # Tests that a file is only downloaded once if it's not updated.
        with SoftTemporaryDirectory() as cache:
            path = hf_hub_download(MODEL_IDENTIFIER, "file_0.txt", cache_dir=cache)
            creation_time_0 = os.path.getmtime(path)

            time.sleep(2)

            path = hf_hub_download(MODEL_IDENTIFIER, "file_0.txt", cache_dir=cache)
            creation_time_1 = os.path.getmtime(path)

            self.assertEqual(creation_time_0, creation_time_1)

    @xfail_on_windows(reason="Symlinks are deactivated in Windows tests.")
    def test_file_download_happens_once_intra_revision(self):
        # Tests that a file is only downloaded once if it's not updated, even across different revisions.

        with SoftTemporaryDirectory() as cache:
            path = hf_hub_download(MODEL_IDENTIFIER, "file_0.txt", cache_dir=cache)
            creation_time_0 = os.path.getmtime(path)

            time.sleep(2)

            path = hf_hub_download(MODEL_IDENTIFIER, "file_0.txt", cache_dir=cache, revision="file-2")
            creation_time_1 = os.path.getmtime(path)

            self.assertEqual(creation_time_0, creation_time_1)

    @xfail_on_windows(reason="Symlinks are deactivated in Windows tests.")
    def test_multiple_refs_for_same_file(self):
        with SoftTemporaryDirectory() as cache:
            hf_hub_download(MODEL_IDENTIFIER, "file_0.txt", cache_dir=cache)
            hf_hub_download(MODEL_IDENTIFIER, "file_0.txt", cache_dir=cache, revision="file-2")

            expected_directory_name = f"models--{MODEL_IDENTIFIER.replace('/', '--')}"
            expected_path = os.path.join(cache, expected_directory_name)

            refs = os.listdir(os.path.join(expected_path, "refs"))
            refs.sort()

            snapshots = os.listdir(os.path.join(expected_path, "snapshots"))
            snapshots.sort()

            # Directory should contain two revisions
            self.assertListEqual(refs, ["file-2", "main"])

            refs_contents = [get_file_contents(os.path.join(expected_path, "refs", f)) for f in refs]
            refs_contents.sort()

            # snapshots directory should contain two snapshots
            self.assertListEqual(refs_contents, snapshots)

            snapshot_links = [
                os.readlink(os.path.join(expected_path, "snapshots", filename, "file_0.txt")) for filename in snapshots
            ]

            # All snapshot links should point to the same file.
            self.assertEqual(*snapshot_links)


@with_production_testing
class CacheFileLayoutSnapshotDownload(unittest.TestCase):
    @xfail_on_windows(reason="Symlinks are deactivated in Windows tests.")
    def test_file_downloaded_in_cache(self):
        with SoftTemporaryDirectory() as cache:
            snapshot_download(MODEL_IDENTIFIER, cache_dir=cache)

            expected_directory_name = f"models--{MODEL_IDENTIFIER.replace('/', '--')}"
            expected_path = os.path.join(cache, expected_directory_name)

            refs = os.listdir(os.path.join(expected_path, "refs"))

            snapshots = os.listdir(os.path.join(expected_path, "snapshots"))
            snapshots.sort()

            # Directory should contain two revisions
            self.assertListEqual(refs, ["main"])

            ref_content = get_file_contents(os.path.join(expected_path, "refs", refs[0]))

            # snapshots directory should contain two snapshots
            self.assertListEqual([ref_content], snapshots)

            snapshot_path = os.path.join(expected_path, "snapshots", snapshots[0])

            files_in_snapshot = os.listdir(snapshot_path)

            snapshot_links = [os.readlink(os.path.join(snapshot_path, filename)) for filename in files_in_snapshot]

            resolved_snapshot_links = [os.path.normpath(os.path.join(snapshot_path, link)) for link in snapshot_links]

            self.assertTrue(all([os.path.isfile(link) for link in resolved_snapshot_links]))

    @xfail_on_windows(reason="Symlinks are deactivated in Windows tests.")
    def test_file_downloaded_in_cache_several_revisions(self):
        with SoftTemporaryDirectory() as cache:
            snapshot_download(MODEL_IDENTIFIER, cache_dir=cache, revision="file-3")
            snapshot_download(MODEL_IDENTIFIER, cache_dir=cache, revision="file-2")

            expected_directory_name = f"models--{MODEL_IDENTIFIER.replace('/', '--')}"
            expected_path = os.path.join(cache, expected_directory_name)

            refs = os.listdir(os.path.join(expected_path, "refs"))
            refs.sort()

            snapshots = os.listdir(os.path.join(expected_path, "snapshots"))
            snapshots.sort()

            # Directory should contain two revisions
            self.assertListEqual(refs, ["file-2", "file-3"])

            refs_content = [get_file_contents(os.path.join(expected_path, "refs", ref)) for ref in refs]
            refs_content.sort()

            # snapshots directory should contain two snapshots
            self.assertListEqual(refs_content, snapshots)

            snapshots_paths = [os.path.join(expected_path, "snapshots", s) for s in snapshots]

            files_in_snapshots = {s: os.listdir(s) for s in snapshots_paths}
            links_in_snapshots = {
                k: [os.readlink(os.path.join(k, _v)) for _v in v] for k, v in files_in_snapshots.items()
            }

            resolved_snapshots_links = {
                k: [os.path.normpath(os.path.join(k, link)) for link in v] for k, v in links_in_snapshots.items()
            }

            all_links = [b for a in resolved_snapshots_links.values() for b in a]
            all_unique_links = set(all_links)

            # [ 100]  .
            # ├── [ 140]  blobs
            # │   ├── [   7]  4475433e279a71203927cbe80125208a3b5db560
            # │   ├── [   7]  50fcd26d6ce3000f9d5f12904e80eccdc5685dd1
            # │   ├── [   7]  80146afc836c60e70ba67933fec439ab05b478f6
            # │   ├── [   7]  8cf9e18f080becb674b31c21642538269fe886a4
            # │   └── [1.1K]  ac481c8eb05e4d2496fbe076a38a7b4835dd733d
            # ├── [  80]  refs
            # │   ├── [  40]  file-2
            # │   └── [  40]  file-3
            # └── [  80]  snapshots
            #     ├── [ 120]  5e23cb3ae7f904919a442e1b27dcddae6c6bc292
            #     │   ├── [  52]  file_0.txt -> ../../blobs/80146afc836c60e70ba67933fec439ab05b478f6
            #     │   ├── [  52]  file_1.txt -> ../../blobs/50fcd26d6ce3000f9d5f12904e80eccdc5685dd1
            #     │   ├── [  52]  file_2.txt -> ../../blobs/4475433e279a71203927cbe80125208a3b5db560
            #     │   └── [  52]  .gitattributes -> ../../blobs/ac481c8eb05e4d2496fbe076a38a7b4835dd733d
            #     └── [ 120]  78aa2ebdb60bba086496a8792ba506e58e587b4c
            #         ├── [  52]  file_0.txt -> ../../blobs/80146afc836c60e70ba67933fec439ab05b478f6
            #         ├── [  52]  file_1.txt -> ../../blobs/50fcd26d6ce3000f9d5f12904e80eccdc5685dd1
            #         ├── [  52]  file_3.txt -> ../../blobs/8cf9e18f080becb674b31c21642538269fe886a4
            #         └── [  52]  .gitattributes -> ../../blobs/ac481c8eb05e4d2496fbe076a38a7b4835dd733d

            # Across the two revisions, there should be 8 total links
            self.assertEqual(len(all_links), 8)

            # Across the two revisions, there should only be 5 unique files.
            self.assertEqual(len(all_unique_links), 5)


class ReferenceUpdates(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)

    def test_update_reference(self):
        repo_id = self._api.create_repo(repo_name(), exist_ok=True).repo_id

        try:
            self._api.upload_file(path_or_fileobj=BytesIO(b"Some string"), path_in_repo="file.txt", repo_id=repo_id)

            with SoftTemporaryDirectory() as cache:
                hf_hub_download(repo_id, "file.txt", cache_dir=cache)

                expected_directory_name = f"models--{repo_id.replace('/', '--')}"
                expected_path = os.path.join(cache, expected_directory_name)

                refs = os.listdir(os.path.join(expected_path, "refs"))

                # Directory should contain two revisions
                self.assertListEqual(refs, ["main"])

                initial_ref_content = get_file_contents(os.path.join(expected_path, "refs", refs[0]))

                # Upload a new file on the same branch
                self._api.upload_file(
                    path_or_fileobj=BytesIO(b"Some new string"),
                    path_in_repo="file.txt",
                    repo_id=repo_id,
                )

                hf_hub_download(repo_id, "file.txt", cache_dir=cache)

                final_ref_content = get_file_contents(os.path.join(expected_path, "refs", refs[0]))

                # The `main` reference should point to two different, but existing snapshots which contain
                # a 'file.txt'
                self.assertNotEqual(initial_ref_content, final_ref_content)
                self.assertTrue(os.path.isdir(os.path.join(expected_path, "snapshots", initial_ref_content)))
                self.assertTrue(
                    os.path.isfile(os.path.join(expected_path, "snapshots", initial_ref_content, "file.txt"))
                )
                self.assertTrue(os.path.isdir(os.path.join(expected_path, "snapshots", final_ref_content)))
                self.assertTrue(
                    os.path.isfile(os.path.join(expected_path, "snapshots", final_ref_content, "file.txt"))
                )
        except Exception:
            raise
        finally:
            self._api.delete_repo(repo_id)
