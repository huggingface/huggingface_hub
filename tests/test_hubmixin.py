import json
import os
import pickle
import unittest
from pathlib import Path
from typing import TypeVar
from unittest.mock import Mock, patch

import pytest

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.constants import PYTORCH_WEIGHTS_NAME
from huggingface_hub.hub_mixin import ModelHubMixin, PyTorchModelHubMixin
from huggingface_hub.utils import SoftTemporaryDirectory, is_torch_available, EntryNotFoundError

from .testing_constants import ENDPOINT_STAGING, TOKEN, USER
from .testing_utils import repo_name


if is_torch_available():
    import torch.nn as nn
    import torch


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
    CONFIG = {"num": 10, "act": "gelu_fast"}

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
    cache_dir: Path

    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)

    def test_save_pretrained_basic(self):
        DummyModel().save_pretrained(self.cache_dir)
        files = os.listdir(self.cache_dir)
        self.assertTrue("model.safetensors" in files)
        self.assertEqual(len(files), 1)

    def test_save_pretrained_with_config(self):
        DummyModel().save_pretrained(self.cache_dir, config=CONFIG)
        files = os.listdir(self.cache_dir)
        self.assertTrue("config.json" in files)
        self.assertTrue("model.safetensors" in files)
        self.assertEqual(len(files), 2)

    def test_save_as_safetensors(self):
        DummyModel().save_pretrained(self.cache_dir, config=TOKEN)
        modelFile = self.cache_dir / "model.safetensors"
        # check for safetensors header to ensure we are saving the model in safetensors format
        # while an implementation detail, assert as this has safety implications
        # https://github.com/huggingface/safetensors?tab=readme-ov-file#format
        with open(modelFile, "rb") as f:
            self.assertEqual(f.read(8), b"\x80\x00\x00\x00\x00\x00\x00\x00")


    def test_save_pretrained_with_push_to_hub(self):
        repo_id = repo_name("save")
        save_directory = self.cache_dir / repo_id

        config = {"hello": "world"}
        mocked_model = DummyModel()
        mocked_model.push_to_hub = Mock()
        mocked_model._save_pretrained = Mock()  # disable _save_pretrained to speed-up

        # Not pushed to hub
        mocked_model.save_pretrained(save_directory)
        mocked_model.push_to_hub.assert_not_called()

        # Push to hub with repo_id
        mocked_model.save_pretrained(save_directory, push_to_hub=True, repo_id="CustomID", config=config)
        mocked_model.push_to_hub.assert_called_with(repo_id="CustomID", config=config)

        # Push to hub with default repo_id (based on dir name)
        mocked_model.save_pretrained(save_directory, push_to_hub=True, config=config)
        mocked_model.push_to_hub.assert_called_with(repo_id=repo_id, config=config)

    @patch.object(DummyModel, "_from_pretrained")
    def test_from_pretrained_model_id_only(self, from_pretrained_mock: Mock) -> None:
        model = DummyModel.from_pretrained("namespace/repo_name")
        from_pretrained_mock.assert_called_once()
        self.assertIs(model, from_pretrained_mock.return_value)

    def pretend_file_download(self, **kwargs):
        DummyModel().save_pretrained(self.cache_dir, config=TOKEN)
        return self.cache_dir / "model.safetensors"
    
    @patch.object(DummyModel, "_hf_hub_download")
    def test_from_pretrained_model_from_hub_prefer_safetensor(self, hf_hub_download_mock: Mock) -> None:
        hf_hub_download_mock.side_effect = self.pretend_file_download
        model = DummyModel.from_pretrained("namespace/repo_name")
        hf_hub_download_mock.assert_called_once_with(
            repo_id="namespace/repo_name",
            filename="model.safetensors",
            revision=None,
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=False,
            token=None,
            local_files_only=False
        )
        self.assertIsNotNone(model)

    def pretend_file_download_fallback(self, **kwargs):
        if kwargs.get("filename") == "model.safetensors":
            raise EntryNotFoundError("not found")
        
        with open(self.cache_dir / PYTORCH_WEIGHTS_NAME, "wb") as f:
            class TestMixin(ModelHubMixin):
                def _save_pretrained(self, save_directory: Path) -> None:
                    torch.save(DummyModel().state_dict(), save_directory / PYTORCH_WEIGHTS_NAME)

            TestMixin().save_pretrained(self.cache_dir)
        return self.cache_dir / PYTORCH_WEIGHTS_NAME
    
    @patch.object(DummyModel, "_hf_hub_download")
    def test_from_pretrained_model_from_hub_fallback_pickle(self, hf_hub_download_mock: Mock) -> None:
        hf_hub_download_mock.side_effect = self.pretend_file_download_fallback
        model = DummyModel.from_pretrained("namespace/repo_name")
        hf_hub_download_mock.assert_any_call(
            repo_id="namespace/repo_name",
            filename="model.safetensors",
            revision=None,
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=False,
            token=None,
            local_files_only=False
        )
        hf_hub_download_mock.assert_any_call(
            repo_id="namespace/repo_name",
            filename="pytorch_model.bin",
            revision=None,
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=False,
            token=None,
            local_files_only=False
        )
        self.assertIsNotNone(model)

    @patch.object(DummyModel, "_from_pretrained")
    def test_from_pretrained_model_id_and_revision(self, from_pretrained_mock: Mock) -> None:
        """Regression test for #1313.
        See https://github.com/huggingface/huggingface_hub/issues/1313."""
        model = DummyModel.from_pretrained("namespace/repo_name", revision="123456789")
        from_pretrained_mock.assert_called_once_with(
            model_id="namespace/repo_name",
            revision="123456789",  # Revision is passed correctly!
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=False,
            local_files_only=False,
            token=None,
        )
        self.assertIs(model, from_pretrained_mock.return_value)

    def test_from_pretrained_to_relative_path(self):
        with SoftTemporaryDirectory(dir=Path(".")) as tmp_relative_dir:
            relative_save_directory = Path(tmp_relative_dir) / "model"
            DummyModel().save_pretrained(relative_save_directory, config=CONFIG)
            model = DummyModel.from_pretrained(relative_save_directory)
            self.assertDictEqual(model.config, CONFIG)

    def test_from_pretrained_to_absolute_path(self):
        save_directory = self.cache_dir / "subfolder"
        DummyModel().save_pretrained(save_directory, config=CONFIG)
        model = DummyModel.from_pretrained(save_directory)
        self.assertDictEqual(model.config, CONFIG)

    def test_from_pretrained_to_absolute_string_path(self):
        save_directory = str(self.cache_dir / "subfolder")
        DummyModel().save_pretrained(save_directory, config=CONFIG)
        model = DummyModel.from_pretrained(save_directory)
        self.assertDictEqual(model.config, CONFIG)

    def test_return_type_hint_from_pretrained(self):
        self.assertIn(
            "return",
            DummyModel.from_pretrained.__annotations__,
            "`PyTorchModelHubMixin.from_pretrained` does not set a return type annotation.",
        )
        self.assertIsInstance(
            DummyModel.from_pretrained.__annotations__["return"],
            TypeVar,
            "`PyTorchModelHubMixin.from_pretrained` return type annotation is not a TypeVar.",
        )
        self.assertEqual(
            DummyModel.from_pretrained.__annotations__["return"].__bound__.__forward_arg__,
            "ModelHubMixin",
            "`PyTorchModelHubMixin.from_pretrained` return type annotation is not a TypeVar bound by `ModelHubMixin`.",
        )

    def test_push_to_hub(self):
        repo_id = f"{USER}/{repo_name('push_to_hub')}"
        DummyModel().push_to_hub(repo_id=repo_id, api_endpoint=ENDPOINT_STAGING, token=TOKEN, config=CONFIG)

        # Test model id exists
        model_info = self._api.model_info(repo_id)
        self.assertEqual(model_info.modelId, repo_id)

        # Test config has been pushed to hub
        tmp_config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            use_auth_token=TOKEN,
            cache_dir=self.cache_dir,
        )
        with open(tmp_config_path) as f:
            self.assertDictEqual(json.load(f), CONFIG)

        # Delete repo
        self._api.delete_repo(repo_id=repo_id)
