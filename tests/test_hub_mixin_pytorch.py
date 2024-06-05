import json
import os
import struct
import unittest
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar
from unittest.mock import Mock, patch

import pytest

from huggingface_hub import HfApi, ModelCard, hf_hub_download
from huggingface_hub.constants import PYTORCH_WEIGHTS_NAME
from huggingface_hub.hub_mixin import ModelHubMixin, PyTorchModelHubMixin
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, SoftTemporaryDirectory, is_torch_available

from .testing_constants import ENDPOINT_STAGING, TOKEN, USER
from .testing_utils import repo_name, requires


DUMMY_OBJECT = object()

DUMMY_MODEL_CARD_TEMPLATE = """
---
{{ card_data }}
---

This is a dummy model card.
Arxiv ID: 1234.56789
"""

DUMMY_MODEL_CARD_TEMPLATE_WITH_CUSTOM_KWARGS = """
---
{{ card_data }}
---

This is a dummy model card with kwargs.
Arxiv ID: 1234.56789

{{ custom_data }}
"""

if is_torch_available():
    import torch
    import torch.nn as nn

    CONFIG = {"num": 10, "act": "gelu_fast"}

    class DummyModel(nn.Module, PyTorchModelHubMixin):
        def __init__(self, **kwargs):
            super().__init__()
            self.config = kwargs.pop("config", None)
            self.l1 = nn.Linear(2, 2)

        def forward(self, x):
            return self.l1(x)

    class DummyModelWithModelCard(
        nn.Module,
        PyTorchModelHubMixin,
        model_card_template=DUMMY_MODEL_CARD_TEMPLATE,
        languages=["en", "zh"],
        library_name="my-dummy-lib",
        license="apache-2.0",
        tags=["tag1", "tag2"],
        pipeline_tag="text-classification",
    ):
        def __init__(self, linear_layer: int = 4):
            super().__init__()
            self.l1 = nn.Linear(linear_layer, linear_layer)

        def forward(self, x):
            return self.l1(x)

    class DummyModelNoConfig(nn.Module, PyTorchModelHubMixin):
        def __init__(
            self,
            num_classes: int = 42,
            state: str = "layernorm",
            not_jsonable: Any = DUMMY_OBJECT,
        ):
            super().__init__()
            self.num_classes = num_classes
            self.state = state
            self.not_jsonable = not_jsonable

    class DummyModelWithConfigAndKwargs(nn.Module, PyTorchModelHubMixin):
        def __init__(self, num_classes: int = 42, state: str = "layernorm", config: Optional[Dict] = None, **kwargs):
            super().__init__()

    class DummyModelWithModelCardAndCustomKwargs(
        nn.Module,
        PyTorchModelHubMixin,
        model_card_template=DUMMY_MODEL_CARD_TEMPLATE_WITH_CUSTOM_KWARGS,
    ):
        def __init__(self, linear_layer: int = 4):
            super().__init__()

else:
    DummyModel = None
    DummyModelWithModelCard = None
    DummyModelNoConfig = None
    DummyModelWithConfigAndKwargs = None
    DummyModelWithModelCardAndCustomKwargs = None


@requires("torch")
@pytest.mark.usefixtures("fx_cache_dir")
class PytorchHubMixinTest(unittest.TestCase):
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
        assert set(files) == {"README.md", "model.safetensors"}

    def test_save_pretrained_with_config(self):
        DummyModel().save_pretrained(self.cache_dir, config=CONFIG)
        files = os.listdir(self.cache_dir)
        assert set(files) == {"README.md", "config.json", "model.safetensors"}

    def test_save_as_safetensors(self):
        DummyModel().save_pretrained(self.cache_dir, config=TOKEN)
        modelFile = self.cache_dir / "model.safetensors"
        # check for safetensors header to ensure we are saving the model in safetensors format
        # while an implementation detail, assert as this has safety implications
        # https://github.com/huggingface/safetensors?tab=readme-ov-file#format
        with open(modelFile, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            self.assertEqual(header_size, 128)

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
        mocked_model.push_to_hub.assert_called_with(repo_id="CustomID", config=config, model_card_kwargs={})

        # Push to hub with default repo_id (based on dir name)
        mocked_model.save_pretrained(save_directory, push_to_hub=True, config=config)
        mocked_model.push_to_hub.assert_called_with(repo_id=repo_id, config=config, model_card_kwargs={})

    @patch.object(DummyModel, "_from_pretrained")
    def test_from_pretrained_model_id_only(self, from_pretrained_mock: Mock) -> None:
        model = DummyModel.from_pretrained("namespace/repo_name")
        from_pretrained_mock.assert_called_once()
        self.assertIs(model, from_pretrained_mock.return_value)

    def pretend_file_download(self, **kwargs):
        if kwargs.get("filename") == "config.json":
            raise HfHubHTTPError("no config")
        DummyModel().save_pretrained(self.cache_dir)
        return self.cache_dir / "model.safetensors"

    @patch("huggingface_hub.hub_mixin.hf_hub_download")
    def test_from_pretrained_model_from_hub_prefer_safetensor(self, hf_hub_download_mock: Mock) -> None:
        hf_hub_download_mock.side_effect = self.pretend_file_download
        model = DummyModel.from_pretrained("namespace/repo_name")
        hf_hub_download_mock.assert_any_call(
            repo_id="namespace/repo_name",
            filename="model.safetensors",
            revision=None,
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=None,
            token=None,
            local_files_only=False,
        )
        self.assertIsNotNone(model)

    def pretend_file_download_fallback(self, **kwargs):
        filename = kwargs.get("filename")
        if filename == "model.safetensors" or filename == "config.json":
            raise EntryNotFoundError("not found")

        class TestMixin(ModelHubMixin):
            def _save_pretrained(self, save_directory: Path) -> None:
                torch.save(DummyModel().state_dict(), save_directory / PYTORCH_WEIGHTS_NAME)

        TestMixin().save_pretrained(self.cache_dir)
        return self.cache_dir / PYTORCH_WEIGHTS_NAME

    @patch("huggingface_hub.hub_mixin.hf_hub_download")
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
            resume_download=None,
            token=None,
            local_files_only=False,
        )
        hf_hub_download_mock.assert_any_call(
            repo_id="namespace/repo_name",
            filename="pytorch_model.bin",
            revision=None,
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=None,
            token=None,
            local_files_only=False,
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
            resume_download=None,
            local_files_only=False,
            token=None,
        )
        self.assertIs(model, from_pretrained_mock.return_value)

    def test_from_pretrained_to_relative_path(self):
        with SoftTemporaryDirectory(dir=Path(".")) as tmp_relative_dir:
            relative_save_directory = Path(tmp_relative_dir) / "model"
            DummyModel().save_pretrained(relative_save_directory, config=CONFIG)
            model = DummyModel.from_pretrained(relative_save_directory)
            self.assertDictEqual(model._hub_mixin_config, CONFIG)

    def test_from_pretrained_to_absolute_path(self):
        save_directory = self.cache_dir / "subfolder"
        DummyModel().save_pretrained(save_directory, config=CONFIG)
        model = DummyModel.from_pretrained(save_directory)
        self.assertDictEqual(model._hub_mixin_config, CONFIG)

    def test_from_pretrained_to_absolute_string_path(self):
        save_directory = str(self.cache_dir / "subfolder")
        DummyModel().save_pretrained(save_directory, config=CONFIG)
        model = DummyModel.from_pretrained(save_directory)
        self.assertDictEqual(model._hub_mixin_config, CONFIG)

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
        DummyModel().push_to_hub(repo_id=repo_id, token=TOKEN, config=CONFIG)

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

    def test_generate_model_card(self):
        model = DummyModelWithModelCard()
        card = model.generate_model_card()
        assert card.data.language == ["en", "zh"]
        assert card.data.library_name == "my-dummy-lib"
        assert card.data.license == "apache-2.0"
        assert card.data.pipeline_tag == "text-classification"
        assert card.data.tags == ["model_hub_mixin", "pytorch_model_hub_mixin", "tag1", "tag2"]

        # Model card template has been used
        assert "This is a dummy model card" in str(card)

        model.save_pretrained(self.cache_dir)
        card_reloaded = ModelCard.load(self.cache_dir / "README.md")

        assert str(card) == str(card_reloaded)
        assert card.data == card_reloaded.data

    def test_load_no_config(self):
        config_file = self.cache_dir / "config.json"

        # Test creating model => auto-generated config
        model = DummyModelNoConfig(num_classes=50)
        assert model._hub_mixin_config == {"num_classes": 50, "state": "layernorm"}

        # Test saving model => auto-generated config is saved
        model.save_pretrained(self.cache_dir)
        assert config_file.exists()
        assert json.loads(config_file.read_text()) == {"num_classes": 50, "state": "layernorm"}

        # Reload model => config is reloaded
        reloaded = DummyModelNoConfig.from_pretrained(self.cache_dir)
        assert reloaded.num_classes == 50
        assert reloaded.state == "layernorm"
        assert reloaded._hub_mixin_config == {"num_classes": 50, "state": "layernorm"}

        # Reload model with custom config => custom config is used
        reloaded_with_default = DummyModelNoConfig.from_pretrained(self.cache_dir, state="other")
        assert reloaded_with_default.num_classes == 50
        assert reloaded_with_default.state == "other"
        assert reloaded_with_default._hub_mixin_config == {"num_classes": 50, "state": "other"}

        config_file.unlink()  # Remove config file
        reloaded_with_default.save_pretrained(self.cache_dir)
        assert json.loads(config_file.read_text()) == {"num_classes": 50, "state": "other"}

    def test_save_with_non_jsonable_config(self):
        # Save with a non-jsonable value
        my_object = object()
        model = DummyModelNoConfig(not_jsonable=my_object)
        assert model.not_jsonable is my_object
        assert "not_jsonable" not in model._hub_mixin_config

        # Reload with default value
        model.save_pretrained(self.cache_dir)
        reloaded_model = DummyModelNoConfig.from_pretrained(self.cache_dir)
        assert reloaded_model.not_jsonable is DUMMY_OBJECT
        assert "not_jsonable" not in model._hub_mixin_config

        # If jsonable value passed by user, it's saved in the config
        (self.cache_dir / "config.json").unlink()
        new_model = DummyModelNoConfig(not_jsonable=123)
        new_model.save_pretrained(self.cache_dir)
        assert new_model._hub_mixin_config["not_jsonable"] == 123

        reloaded_new_model = DummyModelNoConfig.from_pretrained(self.cache_dir)
        assert reloaded_new_model.not_jsonable == 123
        assert reloaded_new_model._hub_mixin_config["not_jsonable"] == 123

    def test_save_model_with_shared_tensors(self):
        """
        Regression test for #2086. Shared tensors should be saved correctly.

        See https://github.com/huggingface/huggingface_hub/pull/2086 for more details.
        """

        class ModelWithSharedTensors(nn.Module, PyTorchModelHubMixin):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(100, 100)
                self.b = self.a

            def forward(self, x):
                return self.b(self.a(x))

        # Save and reload model
        model = ModelWithSharedTensors()
        model.save_pretrained(self.cache_dir)
        reloaded = ModelWithSharedTensors.from_pretrained(self.cache_dir)

        # Linear layers should share weights and biases in memory
        state_dict = reloaded.state_dict()
        a_weight_ptr = state_dict["a.weight"].untyped_storage().data_ptr()
        b_weight_ptr = state_dict["b.weight"].untyped_storage().data_ptr()
        a_bias_ptr = state_dict["a.bias"].untyped_storage().data_ptr()
        b_bias_ptr = state_dict["b.bias"].untyped_storage().data_ptr()
        assert a_weight_ptr == b_weight_ptr
        assert a_bias_ptr == b_bias_ptr

    def test_save_pretrained_when_config_and_kwargs_are_passed(self):
        # Test creating model with config and kwargs => all values are saved together in config.json
        model = DummyModelWithConfigAndKwargs(num_classes=50, state="layernorm", config={"a": 1}, b=2, c=3)
        model.save_pretrained(self.cache_dir)
        assert model._hub_mixin_config == {"num_classes": 50, "state": "layernorm", "a": 1, "b": 2, "c": 3}

        reloaded = DummyModelWithConfigAndKwargs.from_pretrained(self.cache_dir)
        assert reloaded._hub_mixin_config == model._hub_mixin_config

    def test_model_card_with_custom_kwargs(self):
        model_card_kwargs = {"custom_data": "This is a model custom data: 42."}

        # Test creating model with custom kwargs => custom data is saved in model card
        model = DummyModelWithModelCardAndCustomKwargs()
        card = model.generate_model_card(**model_card_kwargs)
        assert model_card_kwargs["custom_data"] in str(card)

        # Test saving card => model card is saved and restored with custom data
        model.save_pretrained(self.cache_dir, model_card_kwargs=model_card_kwargs)
        card_reloaded = ModelCard.load(self.cache_dir / "README.md")
        assert str(card) == str(card_reloaded)
