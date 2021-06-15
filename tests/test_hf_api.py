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
import shutil
import subprocess
import tempfile
import time
import unittest
from io import BytesIO

from huggingface_hub.constants import REPO_TYPE_DATASET, REPO_TYPE_SPACE
from huggingface_hub.file_download import cached_download
from huggingface_hub.hf_api import HfApi, HfFolder, ModelInfo, RepoObj
from requests.exceptions import HTTPError

from .testing_constants import ENDPOINT_STAGING, ENDPOINT_STAGING_BASIC_AUTH, PASS, USER
from .testing_utils import (
    DUMMY_MODEL_ID,
    DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT,
    require_git_lfs,
    set_write_permission_and_retry,
)


REPO_NAME = "my-model-{}".format(int(time.time() * 10e3))
REPO_NAME_LARGE_FILE = "my-model-largefiles-{}".format(int(time.time() * 10e3))
DATASET_REPO_NAME = "my-dataset-{}".format(int(time.time() * 10e3))
SPACE_REPO_NAME = "my-space-{}".format(int(time.time() * 10e3))
WORKING_REPO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures/working_repo"
)
LARGE_FILE_14MB = "https://cdn-media.huggingface.co/lfs-largefiles/progit.epub"
LARGE_FILE_18MB = "https://cdn-media.huggingface.co/lfs-largefiles/progit.pdf"


class HfApiCommonTest(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING)


class HfApiLoginTest(HfApiCommonTest):
    def test_login_invalid(self):
        with self.assertRaises(HTTPError):
            self._api.login(username=USER, password="fake")

    def test_login_valid(self):
        token = self._api.login(username=USER, password=PASS)
        self.assertIsInstance(token, str)


class HfApiCommonTestWithLogin(HfApiCommonTest):
    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = cls._api.login(username=USER, password=PASS)


class HfApiEndpointsTest(HfApiCommonTestWithLogin):
    def test_whoami(self):
        user, orgs = self._api.whoami(token=self._token)
        self.assertEqual(user, USER)
        self.assertIsInstance(orgs, list)

    def test_list_repos_objs(self):
        objs = self._api.list_repos_objs(token=self._token)
        self.assertIsInstance(objs, list)
        if len(objs) > 0:
            o = objs[-1]
            self.assertIsInstance(o, RepoObj)

    def test_create_update_and_delete_repo(self):
        self._api.create_repo(token=self._token, name=REPO_NAME)
        res = self._api.update_repo_visibility(
            token=self._token, name=REPO_NAME, private=True
        )
        self.assertTrue(res["private"])
        res = self._api.update_repo_visibility(
            token=self._token, name=REPO_NAME, private=False
        )
        self.assertFalse(res["private"])
        self._api.delete_repo(token=self._token, name=REPO_NAME)

    def test_create_update_and_delete_dataset_repo(self):
        self._api.create_repo(
            token=self._token, name=DATASET_REPO_NAME, repo_type=REPO_TYPE_DATASET
        )
        res = self._api.update_repo_visibility(
            token=self._token,
            name=DATASET_REPO_NAME,
            private=True,
            repo_type=REPO_TYPE_DATASET,
        )
        self.assertTrue(res["private"])
        res = self._api.update_repo_visibility(
            token=self._token,
            name=DATASET_REPO_NAME,
            private=False,
            repo_type=REPO_TYPE_DATASET,
        )
        self.assertFalse(res["private"])
        self._api.delete_repo(
            token=self._token, name=DATASET_REPO_NAME, repo_type=REPO_TYPE_DATASET
        )

    @unittest.skip("skipped while spaces in beta")
    def test_create_update_and_delete_space_repo(self):
        self._api.create_repo(
            token=self._token, name=SPACE_REPO_NAME, repo_type=REPO_TYPE_SPACE
        )
        res = self._api.update_repo_visibility(
            token=self._token,
            name=SPACE_REPO_NAME,
            private=True,
            repo_type=REPO_TYPE_SPACE,
        )
        self.assertTrue(res["private"])
        res = self._api.update_repo_visibility(
            token=self._token,
            name=SPACE_REPO_NAME,
            private=False,
            repo_type=REPO_TYPE_SPACE,
        )
        self.assertFalse(res["private"])
        self._api.delete_repo(
            token=self._token, name=SPACE_REPO_NAME, repo_type=REPO_TYPE_SPACE
        )


class HfApiUploadFileTest(HfApiCommonTestWithLogin):
    def setUp(self) -> None:
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_file = os.path.join(self.tmp_dir, "temp")
        self.tmp_file_content = "Content of the file"
        with open(self.tmp_file, "w+") as f:
            f.write(self.tmp_file_content)
        self.addCleanup(
            lambda: shutil.rmtree(self.tmp_dir, onerror=set_write_permission_and_retry)
        )

    def test_upload_file_validation(self):
        with self.assertRaises(ValueError, msg="Wrong repo type"):
            self._api.upload_file(
                path_or_fileobj=self.tmp_file,
                path_in_repo="README.md",
                repo_id=f"{USER}/{REPO_NAME}",
                repo_type="this type does not exist",
                token=self._token,
            )

        with self.assertRaises(ValueError, msg="File opened in text mode"):
            with open(self.tmp_file, "rt") as ftext:
                self._api.upload_file(
                    path_or_fileobj=ftext,
                    path_in_repo="README.md",
                    repo_id=f"{USER}/{REPO_NAME}",
                    token=self._token,
                )

        with self.assertRaises(
            ValueError, msg="path_or_fileobj is str but does not point to a file"
        ):
            self._api.upload_file(
                path_or_fileobj=os.path.join(self.tmp_dir, "nofile.pth"),
                path_in_repo="README.md",
                repo_id=f"{USER}/{REPO_NAME}",
                token=self._token,
            )

        for (invalid_path, msg) in [
            ("Remote\\README.md", "Has a backslash"),
            ("/Remote/README.md", "Starts with a slash"),
            ("Remote/../subtree/./README.md", "Has relative parts"),
        ]:
            with self.subTest(msg=msg):
                with self.assertRaises(ValueError, msg="path_in_repo is invalid"):
                    self._api.upload_file(
                        path_or_fileobj=self.tmp_file,
                        path_in_repo=invalid_path,
                        repo_id=f"{USER}/{REPO_NAME}",
                        token=self._token,
                    )

    def test_upload_file_path(self):
        self._api.create_repo(token=self._token, name=REPO_NAME)
        try:
            self._api.upload_file(
                path_or_fileobj=self.tmp_file,
                path_in_repo="temp/new_file.md",
                repo_id=f"{USER}/{REPO_NAME}",
                token=self._token,
            )
            url = "{}/{user}/{repo}/resolve/main/temp/new_file.md".format(
                ENDPOINT_STAGING,
                user=USER,
                repo=REPO_NAME,
            )
            filepath = cached_download(url, force_download=True)
            with open(filepath) as downloaded_file:
                content = downloaded_file.read()
            self.assertEqual(content, self.tmp_file_content)

        except Exception as err:
            self.fail(err)
        finally:
            self._api.delete_repo(token=self._token, name=REPO_NAME)

    def test_upload_file_fileobj(self):
        self._api.create_repo(token=self._token, name=REPO_NAME)
        try:
            with open(self.tmp_file, "rb") as filestream:
                self._api.upload_file(
                    path_or_fileobj=filestream,
                    path_in_repo="temp/new_file.md",
                    repo_id=f"{USER}/{REPO_NAME}",
                    token=self._token,
                )
            url = "{}/{user}/{repo}/resolve/main/temp/new_file.md".format(
                ENDPOINT_STAGING,
                user=USER,
                repo=REPO_NAME,
            )
            filepath = cached_download(url, force_download=True)
            with open(filepath) as downloaded_file:
                content = downloaded_file.read()
            self.assertEqual(content, self.tmp_file_content)

        except Exception as err:
            self.fail(err)
        finally:
            self._api.delete_repo(token=self._token, name=REPO_NAME)

    def test_upload_file_bytesio(self):
        self._api.create_repo(token=self._token, name=REPO_NAME)
        try:
            filecontent = BytesIO(b"File content, but in bytes IO")
            self._api.upload_file(
                path_or_fileobj=filecontent,
                path_in_repo="temp/new_file.md",
                repo_id=f"{USER}/{REPO_NAME}",
                token=self._token,
            )
            url = "{}/{user}/{repo}/resolve/main/temp/new_file.md".format(
                ENDPOINT_STAGING,
                user=USER,
                repo=REPO_NAME,
            )
            filepath = cached_download(url, force_download=True)
            with open(filepath) as downloaded_file:
                content = downloaded_file.read()
            self.assertEqual(content, filecontent.getvalue().decode())

        except Exception as err:
            self.fail(err)
        finally:
            self._api.delete_repo(token=self._token, name=REPO_NAME)

    def test_upload_file_conflict(self):
        self._api.create_repo(token=self._token, name=REPO_NAME)
        try:
            filecontent = BytesIO(b"File content, but in bytes IO")
            self._api.upload_file(
                path_or_fileobj=filecontent,
                path_in_repo="temp/new_file.md",
                repo_id=f"{USER}/{REPO_NAME}",
                token=self._token,
                identical_ok=True,
            )

            # No exception raised when identical_ok is True
            self._api.upload_file(
                path_or_fileobj=filecontent,
                path_in_repo="temp/new_file.md",
                repo_id=f"{USER}/{REPO_NAME}",
                token=self._token,
                identical_ok=True,
            )

            with self.assertRaises(HTTPError) as err_ctx:
                self._api.upload_file(
                    path_or_fileobj=filecontent,
                    path_in_repo="temp/new_file.md",
                    repo_id=f"{USER}/{REPO_NAME}",
                    token=self._token,
                    identical_ok=False,
                )
                self.assertEqual(err_ctx.exception.response.status_code, 409)

        except Exception as err:
            self.fail(err)
        finally:
            self._api.delete_repo(token=self._token, name=REPO_NAME)


class HfApiPublicTest(unittest.TestCase):
    def test_staging_list_models(self):
        _api = HfApi(endpoint=ENDPOINT_STAGING)
        _ = _api.list_models()

    def test_list_models(self):
        _api = HfApi()
        models = _api.list_models()
        self.assertGreater(len(models), 100)
        self.assertIsInstance(models[0], ModelInfo)

    def test_list_models_complex_query(self):
        # Let's list the 10 most recent models
        # with tags "bert" and "jax",
        # ordered by last modified date.
        _api = HfApi()
        models = _api.list_models(
            filter=("bert", "jax"), sort="lastModified", direction=-1, limit=10
        )
        # we have at least 1 models
        self.assertGreater(len(models), 1)
        self.assertLessEqual(len(models), 10)
        model = models[0]
        self.assertIsInstance(model, ModelInfo)
        self.assertTrue(all(tag in model.tags for tag in ["bert", "jax"]))

    def test_model_info(self):
        _api = HfApi()
        model = _api.model_info(repo_id=DUMMY_MODEL_ID)
        self.assertIsInstance(model, ModelInfo)
        self.assertNotEqual(model.sha, DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT)
        # One particular commit (not the top of `main`)
        model = _api.model_info(
            repo_id=DUMMY_MODEL_ID, revision=DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT
        )
        self.assertIsInstance(model, ModelInfo)
        self.assertEqual(model.sha, DUMMY_MODEL_ID_REVISION_ONE_SPECIFIC_COMMIT)


class HfFolderTest(unittest.TestCase):
    def test_token_workflow(self):
        """
        Test the whole token save/get/delete workflow,
        with the desired behavior with respect to non-existent tokens.
        """
        token = "token-{}".format(int(time.time()))
        HfFolder.save_token(token)
        self.assertEqual(HfFolder.get_token(), token)
        HfFolder.delete_token()
        HfFolder.delete_token()
        # ^^ not an error, we test that the
        # second call does not fail.
        self.assertEqual(HfFolder.get_token(), None)


@require_git_lfs
class HfLargefilesTest(HfApiCommonTest):
    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = cls._api.login(username=USER, password=PASS)

    def setUp(self):
        try:
            shutil.rmtree(WORKING_REPO_DIR, onerror=set_write_permission_and_retry)
        except FileNotFoundError:
            pass

    def tearDown(self):
        self._api.delete_repo(token=self._token, name=REPO_NAME_LARGE_FILE)

    def setup_local_clone(self, REMOTE_URL):
        REMOTE_URL_AUTH = REMOTE_URL.replace(
            ENDPOINT_STAGING, ENDPOINT_STAGING_BASIC_AUTH
        )
        subprocess.run(
            ["git", "clone", REMOTE_URL_AUTH, WORKING_REPO_DIR],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            ["git", "lfs", "track", "*.pdf"], check=True, cwd=WORKING_REPO_DIR
        )
        subprocess.run(
            ["git", "lfs", "track", "*.epub"], check=True, cwd=WORKING_REPO_DIR
        )

    def test_end_to_end_thresh_6M(self):
        REMOTE_URL = self._api.create_repo(
            token=self._token, name=REPO_NAME_LARGE_FILE, lfsmultipartthresh=6 * 10 ** 6
        )
        self.setup_local_clone(REMOTE_URL)

        subprocess.run(
            ["wget", LARGE_FILE_18MB],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=WORKING_REPO_DIR,
        )
        subprocess.run(["git", "add", "*"], check=True, cwd=WORKING_REPO_DIR)
        subprocess.run(
            ["git", "commit", "-m", "commit message"], check=True, cwd=WORKING_REPO_DIR
        )

        # This will fail as we haven't set up our custom transfer agent yet.
        failed_process = subprocess.run(
            ["git", "push"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=WORKING_REPO_DIR,
        )
        self.assertEqual(failed_process.returncode, 1)
        self.assertIn("cli lfs-enable-largefiles", failed_process.stderr.decode())
        # ^ Instructions on how to fix this are included in the error message.

        subprocess.run(
            ["huggingface-cli", "lfs-enable-largefiles", WORKING_REPO_DIR], check=True
        )

        start_time = time.time()
        subprocess.run(["git", "push"], check=True, cwd=WORKING_REPO_DIR)
        print("took", time.time() - start_time)

        # To be 100% sure, let's download the resolved file
        pdf_url = f"{REMOTE_URL}/resolve/main/progit.pdf"
        DEST_FILENAME = "uploaded.pdf"
        subprocess.run(
            ["wget", pdf_url, "-O", DEST_FILENAME],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=WORKING_REPO_DIR,
        )
        dest_filesize = os.stat(os.path.join(WORKING_REPO_DIR, DEST_FILENAME)).st_size
        self.assertEqual(dest_filesize, 18685041)

    def test_end_to_end_thresh_16M(self):
        # Here we'll push one multipart and one non-multipart file in the same commit, and see what happens
        REMOTE_URL = self._api.create_repo(
            token=self._token,
            name=REPO_NAME_LARGE_FILE,
            lfsmultipartthresh=16 * 10 ** 6,
        )
        self.setup_local_clone(REMOTE_URL)

        subprocess.run(
            ["wget", LARGE_FILE_18MB],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=WORKING_REPO_DIR,
        )
        subprocess.run(
            ["wget", LARGE_FILE_14MB],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=WORKING_REPO_DIR,
        )
        subprocess.run(["git", "add", "*"], check=True, cwd=WORKING_REPO_DIR)
        subprocess.run(
            ["git", "commit", "-m", "both files in same commit"],
            check=True,
            cwd=WORKING_REPO_DIR,
        )

        subprocess.run(
            ["huggingface-cli", "lfs-enable-largefiles", WORKING_REPO_DIR], check=True
        )

        start_time = time.time()
        subprocess.run(["git", "push"], check=True, cwd=WORKING_REPO_DIR)
        print("took", time.time() - start_time)
