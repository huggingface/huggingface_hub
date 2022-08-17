import json
import os
import unittest
from pathlib import Path
from unittest.mock import Mock

import pytest

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.file_download import is_torch_available
from huggingface_hub.hub_mixin import PyTorchModelHubMixin
from huggingface_hub.repository import Repository

from .testing_constants import ENDPOINT_STAGING, TOKEN, USER
from .testing_utils import expect_deprecation, repo_name, safe_chdir


def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.

    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)
    else:
        return test_case


if is_torch_available():
    import torch.nn as nn

    class DummyModel(nn.Module, PyTorchModelHubMixin):
        def __init__(self, **kwargs):
            super().__init__()
            self.config = kwargs.pop("config", None)
            self.l1 = nn.Linear(2, 2)

        def forward(self, x):
            return self.l1(x)

else:
    DummyModel = None


@require_torch
@pytest.mark.usefixtures("fx_cache_dir")
class HubMixingTest(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING)
    cache_dir: Path  # from fx_cache_dir fixture
    cache_dir_str: str  # from fx_cache_dir fixture

    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = TOKEN
        cls._api.set_access_token(TOKEN)

    def test_save_pretrained(self):
        model = DummyModel()

        model.save_pretrained(self.cache_dir)
        files = os.listdir(self.cache_dir)
        self.assertTrue("pytorch_model.bin" in files)
        self.assertEqual(len(files), 1)

        model.save_pretrained(self.cache_dir, config={"num": 12, "act": "gelu"})
        files = os.listdir(self.cache_dir)
        self.assertTrue("config.json" in files)
        self.assertTrue("pytorch_model.bin" in files)
        self.assertEqual(len(files), 2)

    def test_save_pretrained_with_push_to_hub(self):
        REPO_NAME = repo_name("save")
        save_directory = str(self.cache_dir / REPO_NAME)
        config = {"hello": "world"}
        mocked_model = DummyModel()
        mocked_model.push_to_hub = Mock()
        mocked_model._save_pretrained = Mock()  # disable _save_pretrained to speed-up

        # Not pushed to hub
        mocked_model.save_pretrained(save_directory)
        mocked_model.push_to_hub.assert_not_called()

        # Push to hub with repo_id
        mocked_model.save_pretrained(
            save_directory, push_to_hub=True, repo_id="CustomID", config=config
        )
        mocked_model.push_to_hub.assert_called_with(repo_id="CustomID", config=config)

        # Push to hub with default repo_id (based on dir name)
        mocked_model.save_pretrained(save_directory, push_to_hub=True, config=config)
        mocked_model.push_to_hub.assert_called_with(repo_id=REPO_NAME, config=config)

        # Push to hub with deprecated kwargs (git-based)
        mocked_model.save_pretrained(
            save_directory,
            push_to_hub=True,
            config=config,
            repo_path_or_name="custom_repo_name",
            git_email="myemail",
            git_user="gituser",
        )
        mocked_model.push_to_hub.assert_called_with(
            repo_path_or_name="custom_repo_name",
            config=config,
            git_email="myemail",
            git_user="gituser",
        )

        # Push to hub with deprecated kwargs + use default repo_name + no config
        mocked_model.save_pretrained(
            save_directory,
            push_to_hub=True,
            git_email="myemail",
            git_user="gituser",
        )
        mocked_model.push_to_hub.assert_called_with(
            repo_path_or_name=save_directory,
            git_email="myemail",
            git_user="gituser",
        )

    def test_rel_path_from_pretrained(self):
        with safe_chdir(self.cache_dir.parent):
            rel_path = "./" + self.cache_dir.name  # building dumb relative path in /tmp

            model = DummyModel()
            model.save_pretrained(rel_path, config={"num": 10, "act": "gelu_fast"})

            model = DummyModel.from_pretrained(rel_path)
            self.assertTrue(model.config == {"num": 10, "act": "gelu_fast"})

    def test_abs_path_from_pretrained(self):
        model = DummyModel()
        model.save_pretrained(
            self.cache_dir_str, config={"num": 10, "act": "gelu_fast"}
        )

        model = DummyModel.from_pretrained(self.cache_dir_str)
        self.assertDictEqual(model.config, {"num": 10, "act": "gelu_fast"})

    def test_push_to_hub_via_http_basic(self):
        REPO_NAME = repo_name("PUSH_TO_HUB_via_http")
        repo_id = f"{USER}/{REPO_NAME}"

        DummyModel().push_to_hub(
            repo_id=repo_id,
            api_endpoint=ENDPOINT_STAGING,
            token=self._token,
            config={"num": 7, "act": "gelu_fast"},
        )

        # Test model id exists
        model_info = self._api.model_info(repo_id, token=self._token)
        self.assertEqual(model_info.modelId, repo_id)

        # Test config has been pushed to hub
        tmp_config_path = hf_hub_download(
            repo_id=repo_id, filename="config.json", use_auth_token=self._token
        )
        with open(tmp_config_path) as f:
            self.assertEqual(json.load(f), {"num": 7, "act": "gelu_fast"})

        # Delete tmp file and repo
        os.remove(tmp_config_path)
        self._api.delete_repo(repo_id=repo_id, token=self._token)

    @expect_deprecation("push_to_hub")
    def test_push_to_hub_via_git_deprecated(self):
        # TODO: remove in 0.12 when git method will be removed
        REPO_NAME = repo_name("PUSH_TO_HUB_via_git")
        repo_id = f"{USER}/{REPO_NAME}"

        DummyModel().push_to_hub(
            repo_path_or_name=f"{self.cache_dir_str}/{REPO_NAME}",
            api_endpoint=ENDPOINT_STAGING,
            use_auth_token=self._token,
        )

        model_info = self._api.model_info(repo_id, token=self._token)
        self.assertEqual(model_info.modelId, repo_id)
        self._api.delete_repo(repo_id=repo_id, token=self._token)

    @expect_deprecation("push_to_hub")
    def test_push_to_hub_via_git_use_lfs_by_default(self):
        # TODO: remove in 0.12 when git method will be removed
        REPO_NAME = repo_name("PUSH_TO_HUB_with_lfs_file")

        os.makedirs(f"{self.cache_dir_str}/{REPO_NAME}")
        self._repo_url = self._api.create_repo(repo_id=REPO_NAME, token=self._token)
        Repository(
            local_dir=f"{self.cache_dir_str}/{REPO_NAME}",
            clone_from=self._repo_url,
            use_auth_token=self._token,
        )

        model = DummyModel()
        large_file = [100] * int(4e6)
        with open(f"{self.cache_dir_str}/{REPO_NAME}/large_file.txt", "w+") as f:
            f.write(json.dumps(large_file))

        model.push_to_hub(
            f"{self.cache_dir_str}/{REPO_NAME}",
            api_endpoint=ENDPOINT_STAGING,
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        model_info = self._api.model_info(f"{USER}/{REPO_NAME}", token=self._token)

        self.assertTrue("large_file.txt" in [f.rfilename for f in model_info.siblings])
        self._api.delete_repo(repo_id=f"{REPO_NAME}", token=self._token)
