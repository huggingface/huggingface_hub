import os
import unittest

from huggingface_hub.snapshot_download import snapshot_download


SNAPSHOT_MODEL_ID = "lysandre/pair-classification-roberta-mnli"
SNAPSHOT_MODEL_ID_REVISION_MAIN = "084226a48571f3e227230bdbdb7cad43b98efc44"
# Commit at the top of main
# todo(replace with a smaller model as this test downloads large files.)


class SnapshotDownloadTests(unittest.TestCase):
    def test_download_model(self):
        storage_folder = snapshot_download(SNAPSHOT_MODEL_ID, revision="main")
        self.assertNotEquals(os.listdir(storage_folder), 0)
        # folder is not empty
        self.assertTrue(SNAPSHOT_MODEL_ID_REVISION_MAIN in storage_folder)
        # folder name contains the revision's commit sha.
