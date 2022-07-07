import time
import unittest
from unittest.mock import patch

import fsspec
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.constants import ENDPOINT
from huggingface_hub.hf_filesystem import HfFileSystem, TempFileUploader

from .testing_constants import ENDPOINT_STAGING, TOKEN, USER


class HfFileSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._api = HfApi(endpoint=ENDPOINT_STAGING)
        cls._token = TOKEN
        cls._api.set_access_token(TOKEN)

        cls._hf_api_model_info_patch = patch(
            "huggingface_hub.hf_filesystem.model_info", cls._api.model_info
        )
        cls._hf_api_space_info_patch = patch(
            "huggingface_hub.hf_filesystem.space_info", cls._api.space_info
        )
        cls._hf_api_dataset_info_patch = patch(
            "huggingface_hub.hf_filesystem.dataset_info", cls._api.dataset_info
        )
        cls._hf_api_upload_file_patch = patch(
            "huggingface_hub.hf_filesystem.upload_file", cls._api.upload_file
        )
        cls._hf_api_delete_file_patch = patch(
            "huggingface_hub.hf_filesystem.delete_file", cls._api.delete_file
        )

        def _hf_hub_url_staging(*args, **kwargs):
            return hf_hub_url(*args, **kwargs).replace(ENDPOINT, ENDPOINT_STAGING)

        cls._hf_hub_url_patch = patch(
            "huggingface_hub.hf_filesystem.hf_hub_url", side_effect=_hf_hub_url_staging
        )

        cls._hf_api_model_info_patch.start()
        cls._hf_api_space_info_patch.start()
        cls._hf_api_dataset_info_patch.start()
        cls._hf_api_upload_file_patch.start()
        cls._hf_api_delete_file_patch.start()
        cls._hf_hub_url_patch.start()

        if HfFileSystem.protocol not in fsspec.available_protocols():
            fsspec.register_implementation(HfFileSystem.protocol, HfFileSystem)

    @classmethod
    def tearDownClass(cls):
        cls._api.unset_access_token()

        cls._hf_api_model_info_patch.stop()
        cls._hf_api_space_info_patch.stop()
        cls._hf_api_dataset_info_patch.stop()
        cls._hf_api_upload_file_patch.stop()
        cls._hf_api_delete_file_patch.stop()
        cls._hf_hub_url_patch.stop()

    def setUp(self):
        repo_name = f"repo_txt_data-{int(time.time() * 10e3)}"
        repo_id = f"{USER}/{repo_name}"
        self._api.create_repo(
            repo_id,
            token=self._token,
            repo_type="dataset",
            private=True,
        )
        self._api.upload_file(
            repo_id=repo_id,
            path_or_fileobj="dummy text data".encode("utf-8"),
            token=TOKEN,
            path_in_repo="data/text_data.txt",
            repo_type="dataset",
        )
        self.repo_id = repo_id

    def tearDown(self):
        self._api.delete_repo(self.repo_id, token=self._token, repo_type="dataset")

    def test_glob(self):
        hffs = HfFileSystem(self.repo_id, token=self._token, repo_type="dataset")
        self.assertEqual(sorted(hffs.glob("*")), sorted([".gitattributes", "data"]))

    def test_file_type(self):
        hffs = HfFileSystem(self.repo_id, token=self._token, repo_type="dataset")
        self.assertTrue(hffs.isdir("data") and not hffs.isdir(".gitattributes"))
        self.assertTrue(hffs.isfile("data/text_data.txt") and not hffs.isfile("data"))

    def test_remove_file(self):
        hffs = HfFileSystem(self.repo_id, token=self._token, repo_type="dataset")
        hffs.rm("data/text_data.txt")
        self.assertEqual(hffs.glob("data/*"), [])

    def test_read_file(self):
        hffs = HfFileSystem(self.repo_id, token=self._token, repo_type="dataset")
        with hffs.open("data/text_data.txt", "r") as f:
            self.assertEqual(f.read(), "dummy text data")

    def test_write_file(self):
        hffs = HfFileSystem(self.repo_id, token=self._token, repo_type="dataset")
        data = "new text data"
        with hffs.open("data/new_text_data.txt", "w") as f:
            f.write(data)
        self.assertIn("data/new_text_data.txt", hffs.glob("data/*"))
        with hffs.open("data/new_text_data.txt", "r") as f:
            self.assertEqual(f.read(), data)

    def test_write_file_multiple_chunks(self):
        hffs = HfFileSystem(self.repo_id, token=self._token, repo_type="dataset")
        data = "new text data big" * TempFileUploader.DEFAULT_BLOCK_SIZE
        with hffs.open("data/new_text_data_big.txt", "w") as f:
            for _ in range(2):
                f.write(data)

        self.assertIn("data/new_text_data_big.txt", hffs.glob("data/*"))
        with hffs.open("data/new_text_data_big.txt", "r") as f:
            for _ in range(2):
                self.assertEqual(f.read(len(data)), data)

    def test_append_file(self):
        hffs = HfFileSystem(self.repo_id, token=self._token, repo_type="dataset")
        with hffs.open("data/text_data.txt", "a") as f:
            f.write(" appended text")

        with hffs.open("data/text_data.txt", "r") as f:
            self.assertEqual(f.read(), "dummy text data appended text")

    def test_initialize_from_fsspec(self):
        fs, _, paths = fsspec.get_fs_token_paths(
            f"hf://{self.repo_id}:/data/text_data.txt",
            storage_options={"token": self._token, "repo_type": "dataset"},
        )
        self.assertIsInstance(fs, HfFileSystem)
        self.assertEqual(fs.repo_id, self.repo_id)
        self.assertEqual(fs.token, self._token)
        self.assertEqual(fs.repo_type, "dataset")
        self.assertEqual(paths, ["data/text_data.txt"])
