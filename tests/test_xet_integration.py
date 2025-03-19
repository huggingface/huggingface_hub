import os
from unittest import TestCase, skip

from huggingface_hub import HfApi
from huggingface_hub.constants import ENDPOINT
from huggingface_hub.utils import SoftTemporaryDirectory

from .testing_constants import TOKEN
from .testing_utils import repo_name


WORKING_REPO_SUBDIR = f"fixtures/working_repo_{__name__.split('.')[-1]}"
WORKING_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), WORKING_REPO_SUBDIR)


def is_hf_xet_available():
    """
    Checks if the `hf_xet` module is available and can be imported. Used to skip tests that require `hf_xet` when it is
    not available.
    Returns:
        bool: True if the `hf_xet` module is available and can be imported, False otherwise.
    """
    try:
        from hf_xet import PyPointerFile

        _p = PyPointerFile("path", "hash", 100)
    except ImportError:
        return False
    return True


def require_hf_xet(test_case):
    """
    Decorator marking a test that requires hf_xet.
    These tests are skipped when hf_xet is not installed.
    """
    if not is_hf_xet_available():
        return skip("Test requires hf_xet")(test_case)
    else:
        return test_case


@require_hf_xet
class TestHfXet(TestCase):
    def setUp(self) -> None:
        """
        Set up the test environment.
        This method initializes the following attributes:
        - `self.content`: A byte string containing random content for testing.
        - `self._token`: The authentication token for accessing the Hugging Face API.
        - `self._api`: An instance of `HfApi` initialized with the endpoint and token.
        - `self._repo_id`: The repository ID created by the Hugging Face API.
        """
        self.content = b"RandOm Xet ConTEnT" * 1024
        self._token = TOKEN
        self._api = HfApi(endpoint=ENDPOINT, token=self._token)

        self._repo_id = self._api.create_repo(repo_name()).repo_id

    def tearDown(self) -> None:
        """
        Tear down the test environment.
        This method deletes the repository created for testing.
        """
        self._api.delete_repo(repo_id=self._repo_id)

    def test__xet_available(self):
        """
        Test to check if the Hugging Face Xet integration is available.
        """
        self.assertTrue(is_hf_xet_available())

    def test_upload_and_download_with_hf_xet(self):
        """
        Test the upload and download functionality with Hugging Face's Xet integration.

        * Uses upload_file to upload a file to the repository.
        * Uses hf_hub_download to download the file from the repository.

        The test ensures that the upload and download processes work correctly and that the file integrity is maintained.
        """
        # create a temporary file
        with SoftTemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "download")
            file_path = os.path.join(tmpdir, "file.bin")
            with open(file_path, "wb+") as file:
                file.write(self.content)

            # Upload a file
            self._api.upload_file(repo_id=self._repo_id, path_or_fileobj=file_path, path_in_repo="file.bin")

            # Download a file
            downloaded_file = self._api.hf_hub_download(
                repo_id=self._repo_id, filename="file.bin", cache_dir=cache_dir
            )

            # Check that the downloaded file is the same as the uploaded file
            with open(downloaded_file, "rb") as file:
                downloaded_content = file.read()
            self.assertEqual(downloaded_content, self.content)

    def test_upload_folder_download_snapshot_with_hf_xet(self):
        """
        Test the upload and download functionality of a folder using the Hugging Face API.

        * Uses upload_folder to upload a folder to the repository.
        * Uses snapshot_download to download the folder from the repository.

        The test ensures that the content of the files remains consistent during the upload and download process.
        """
        # create a temporary directory
        with SoftTemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "download")
            folder_path = os.path.join(tmpdir, "folder")
            os.makedirs(folder_path)
            file_path = os.path.join(folder_path, "file.bin")
            with open(file_path, "wb+") as file:
                file.write(self.content)
            file_path = os.path.join(folder_path, "file2.bin")
            with open(file_path, "wb+") as file:
                file.write(self.content)

            # Upload a folder
            self._api.upload_folder(repo_id=self._repo_id, folder_path=folder_path, path_in_repo="folder")

            # Download & verify files
            local_dir = os.path.join(tmpdir, "snapshot")
            os.makedirs(local_dir)
            snapshot_path = self._api.snapshot_download(
                repo_id=self._repo_id, local_dir=local_dir, cache_dir=cache_dir
            )

            for downloaded_file in ["folder/file.bin", "folder/file2.bin"]:
                # Check that the downloaded file is the same as the uploaded file
                print(downloaded_file)
                with open(os.path.join(snapshot_path, downloaded_file), "rb") as file:
                    downloaded_content = file.read()
                self.assertEqual(downloaded_content, self.content)

    def test_upload_large_folder_download_snapshot_with_hf_xet(self):
        """
        Test the upload and download of a large folder using the Hugging Face API.

        * Uses upload_large_folder to upload a large folder to the repository.
        * Uses snapshot_download to download the large folder from the repository.

        The test ensures that the upload and download processes handle large folders correctly and that the integrity of the files is maintained.
        """
        # create a temporary directory
        with SoftTemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "download")
            folder_path = os.path.join(tmpdir, "folder")
            os.makedirs(folder_path)
            for i in range(200):
                file_path = os.path.join(folder_path, f"file_{i}.bin")
                with open(file_path, "wb+") as file:
                    file.write(self.content)

            # Upload a large folder - should require two batches
            self._api.upload_large_folder(repo_id=self._repo_id, folder_path=folder_path, repo_type="model")

            # Download & verify files
            local_dir = os.path.join(tmpdir, "snapshot")
            os.makedirs(local_dir)
            snapshot_path = self._api.snapshot_download(
                repo_id=self._repo_id, local_dir=local_dir, cache_dir=cache_dir
            )

            for i in range(200):
                downloaded_file = f"file_{i}.bin"
                # Check that the downloaded file is the same as the uploaded file
                with open(os.path.join(snapshot_path, downloaded_file), "rb") as file:
                    downloaded_content = file.read()
                self.assertEqual(downloaded_content, self.content)
