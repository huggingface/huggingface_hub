import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hub_mixin import PyTorchModelHubMixin
from huggingface_hub.repository import Repository
from huggingface_hub.utils import is_torch_available, logging

from .testing_constants import ENDPOINT_STAGING, TOKEN, USER
from .testing_utils import expect_deprecation, repo_name, set_write_permission_and_retry


logger = logging.get_logger(__name__)
WORKING_REPO_SUBDIR = "fixtures/working_repo_2"
WORKING_REPO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), WORKING_REPO_SUBDIR
)

if is_torch_available():
    import torch.nn as nn


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
class HubMixingCommonTest(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING)


@require_torch
class HubMixingTest(HubMixingCommonTest):
    def tearDown(self) -> None:
        if os.path.exists(WORKING_REPO_DIR):
            shutil.rmtree(WORKING_REPO_DIR, onerror=set_write_permission_and_retry)
        logger.info(
            f"Does {WORKING_REPO_DIR} exist: {os.path.exists(WORKING_REPO_DIR)}"
        )

    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = TOKEN
        cls._api.token = TOKEN
        cls._api.set_access_token(TOKEN)

    def test_save_pretrained(self):
        REPO_NAME = repo_name("save")
        model = DummyModel()

        model.save_pretrained(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertTrue("pytorch_model.bin" in files)
        self.assertEqual(len(files), 1)

        model.save_pretrained(
            f"{WORKING_REPO_DIR}/{REPO_NAME}", config={"num": 12, "act": "gelu"}
        )
        files = os.listdir(f"{WORKING_REPO_DIR}/{REPO_NAME}")
        self.assertTrue("config.json" in files)
        self.assertTrue("pytorch_model.bin" in files)
        self.assertEqual(len(files), 2)

    def test_save_pretrained_with_push_to_hub(self):
        REPO_NAME = repo_name("save")
        save_directory = f"{WORKING_REPO_DIR}/{REPO_NAME}"
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
        model = DummyModel()
        model.save_pretrained(
            f"tests/{WORKING_REPO_SUBDIR}/FROM_PRETRAINED",
            config={"num": 10, "act": "gelu_fast"},
        )

        model = DummyModel.from_pretrained(
            f"tests/{WORKING_REPO_SUBDIR}/FROM_PRETRAINED"
        )
        self.assertTrue(model.config == {"num": 10, "act": "gelu_fast"})

    def test_abs_path_from_pretrained(self):
        REPO_NAME = repo_name("FROM_PRETRAINED")
        model = DummyModel()
        model.save_pretrained(
            f"{WORKING_REPO_DIR}/{REPO_NAME}",
            config={"num": 10, "act": "gelu_fast"},
        )

        model = DummyModel.from_pretrained(f"{WORKING_REPO_DIR}/{REPO_NAME}")
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
        model_info = self._api.model_info(repo_id)
        self.assertEqual(model_info.modelId, repo_id)

        # Test config has been pushed to hub
        tmp_config_path = hf_hub_download(
            repo_id=repo_id, filename="config.json", token=self._token
        )
        with open(tmp_config_path) as f:
            self.assertEqual(json.load(f), {"num": 7, "act": "gelu_fast"})

        # Delete tmp file and repo
        os.remove(tmp_config_path)
        self._api.delete_repo(repo_id=repo_id)

    @expect_deprecation("push_to_hub")
    def test_push_to_hub_via_git_deprecated(self):
        # TODO: remove in 0.12 when git method will be removed
        REPO_NAME = repo_name("PUSH_TO_HUB_via_git")
        repo_id = f"{USER}/{REPO_NAME}"

        DummyModel().push_to_hub(
            repo_path_or_name=f"{WORKING_REPO_DIR}/{REPO_NAME}",
            api_endpoint=ENDPOINT_STAGING,
            token=self._token,
        )

        model_info = self._api.model_info(repo_id)
        self.assertEqual(model_info.modelId, repo_id)
        self._api.delete_repo(repo_id=repo_id)

    @expect_deprecation("push_to_hub")
    def test_push_to_hub_via_git_use_lfs_by_default(self):
        # TODO: remove in 0.12 when git method will be removed
        REPO_NAME = repo_name("PUSH_TO_HUB_with_lfs_file")
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.makedirs(f"{tmpdirname}/{WORKING_REPO_DIR}/{REPO_NAME}")
            self._repo_url = self._api.create_repo(repo_id=REPO_NAME)
            Repository(
                local_dir=f"{tmpdirname}/{WORKING_REPO_DIR}/{REPO_NAME}",
                clone_from=self._repo_url,
                token=self._token,
            )

            model = DummyModel()
            large_file = [100] * int(4e6)
            with open(
                f"{tmpdirname}/{WORKING_REPO_DIR}/{REPO_NAME}/large_file.txt", "w+"
            ) as f:
                f.write(json.dumps(large_file))

            model.push_to_hub(
                f"{tmpdirname}/{WORKING_REPO_DIR}/{REPO_NAME}",
                api_endpoint=ENDPOINT_STAGING,
                token=self._token,
                git_user="ci",
                git_email="ci@dummy.com",
            )

        model_info = self._api.model_info(f"{USER}/{REPO_NAME}")

        self.assertTrue("large_file.txt" in [f.rfilename for f in model_info.siblings])
        self._api.delete_repo(repo_id=f"{REPO_NAME}")
