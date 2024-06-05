import inspect
import json
import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union
from unittest.mock import Mock, patch

import pytest

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hub_mixin import ModelHubMixin
from huggingface_hub.utils import SoftTemporaryDirectory

from .testing_constants import ENDPOINT_STAGING, TOKEN, USER
from .testing_utils import repo_name


@dataclass
class ConfigAsDataclass:
    foo: int = 10
    bar: str = "baz"


CONFIG_AS_DATACLASS = ConfigAsDataclass(foo=20, bar="qux")
CONFIG_AS_DICT = {"foo": 20, "bar": "qux"}


class BaseModel:
    def _save_pretrained(self, save_directory: Path) -> None:
        return

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        **kwargs,
    ) -> "BaseModel":
        # Little hack but in practice NO-ONE is creating 5 inherited classes for their framework :D
        init_parameters = inspect.signature(cls.__init__).parameters
        if init_parameters.get("config"):
            return cls(config=kwargs.get("config"))
        if init_parameters.get("kwargs"):
            return cls(**kwargs)
        return cls()


class DummyModelNoConfig(BaseModel, ModelHubMixin):
    def __init__(self):
        pass


class DummyModelConfigAsDataclass(BaseModel, ModelHubMixin):
    def __init__(self, config: ConfigAsDataclass):
        pass


class DummyModelConfigAsDict(BaseModel, ModelHubMixin):
    def __init__(self, config: Dict):
        pass


class DummyModelConfigAsOptionalDataclass(BaseModel, ModelHubMixin):
    def __init__(self, config: Optional[ConfigAsDataclass] = None):
        pass


class DummyModelConfigAsOptionalDict(BaseModel, ModelHubMixin):
    def __init__(self, config: Optional[Dict] = None):
        pass


class DummyModelWithKwargs(BaseModel, ModelHubMixin):
    def __init__(self, **kwargs):
        pass


class DummyModelFromPretrainedExpectsConfig(ModelHubMixin):
    def _save_pretrained(self, save_directory: Path) -> None:
        return

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Optional[Dict] = None,
        **kwargs,
    ) -> "BaseModel":
        return cls(**kwargs)


class BaseModelForInheritance(ModelHubMixin, repo_url="https://hf.co/my-repo", library_name="my-cool-library"):
    pass


class DummyModelInherited(BaseModelForInheritance):
    pass


class DummyModelSavingConfig(ModelHubMixin):
    def _save_pretrained(self, save_directory: Path) -> None:
        """Implementation that uses `config.json` to serialize the config.

        This file must not be overwritten by the default config saved by `ModelHubMixin`.
        """
        (save_directory / "config.json").write_text(json.dumps({"custom_config": "custom_config"}))


@dataclass
class DummyModelThatIsAlsoADataclass(ModelHubMixin):
    foo: int
    bar: str

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        **model_kwargs,
    ):
        return cls(**model_kwargs)


class CustomType:
    def __init__(self, value: str):
        self.value = value


class DummyModelWithCustomTypes(
    ModelHubMixin,
    coders={
        CustomType: (
            lambda x: {"value": x.value},
            lambda x: CustomType(x["value"]),
        )
    },
):
    def __init__(
        self, foo: int, bar: str, custom: CustomType, custom_default: CustomType = CustomType("default"), **kwargs
    ):
        self.foo = foo
        self.bar = bar
        self.custom = custom
        self.custom_default = custom_default

    @classmethod
    def _from_pretrained(cls, **kwargs):
        return cls(**kwargs)

    @classmethod
    def _save_pretrained(cls, save_directory: Path):
        return


@pytest.mark.usefixtures("fx_cache_dir")
class HubMixinTest(unittest.TestCase):
    cache_dir: Path

    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)

    def assert_valid_config_json(self) -> None:
        # config.json saved correctly
        with open(self.cache_dir / "config.json") as f:
            assert json.load(f) == CONFIG_AS_DICT

    def assert_no_config_json(self) -> None:
        # config.json not saved
        files = os.listdir(self.cache_dir)
        assert "config.json" not in files

    def test_save_pretrained_no_config(self):
        model = DummyModelNoConfig()
        model.save_pretrained(self.cache_dir)
        self.assert_no_config_json()

    def test_save_pretrained_as_dataclass_basic(self):
        model = DummyModelConfigAsDataclass(CONFIG_AS_DATACLASS)
        model.save_pretrained(self.cache_dir)
        self.assert_valid_config_json()

    def test_save_pretrained_as_dict_basic(self):
        model = DummyModelConfigAsDict(CONFIG_AS_DICT)
        model.save_pretrained(self.cache_dir)
        self.assert_valid_config_json()

    def test_save_pretrained_optional_dataclass(self):
        model = DummyModelConfigAsOptionalDataclass()
        model.save_pretrained(self.cache_dir)
        self.assert_no_config_json()

        model = DummyModelConfigAsOptionalDataclass(CONFIG_AS_DATACLASS)
        model.save_pretrained(self.cache_dir)
        self.assert_valid_config_json()

    def test_save_pretrained_optional_dict(self):
        model = DummyModelConfigAsOptionalDict()
        model.save_pretrained(self.cache_dir)
        self.assert_no_config_json()

        model = DummyModelConfigAsOptionalDict(CONFIG_AS_DICT)
        model.save_pretrained(self.cache_dir)
        self.assert_valid_config_json()

    def test_save_pretrained_with_dataclass_config(self):
        model = DummyModelConfigAsOptionalDataclass()
        model.save_pretrained(self.cache_dir, config=CONFIG_AS_DATACLASS)
        self.assert_valid_config_json()

    def test_save_pretrained_with_dict_config(self):
        model = DummyModelConfigAsOptionalDict()
        model.save_pretrained(self.cache_dir, config=CONFIG_AS_DICT)
        self.assert_valid_config_json()

    def test_init_accepts_kwargs_no_config(self):
        """
        Test that if `__init__` accepts **kwargs and config file doesn't exist then no 'config' kwargs is passed.

        Regression test. See https://github.com/huggingface/huggingface_hub/pull/2058.
        """
        model = DummyModelWithKwargs()
        model.save_pretrained(self.cache_dir)
        with patch.object(
            DummyModelWithKwargs, "_from_pretrained", return_value=DummyModelWithKwargs()
        ) as from_pretrained_mock:
            model = DummyModelWithKwargs.from_pretrained(self.cache_dir)
            assert "config" not in from_pretrained_mock.call_args_list[0].kwargs

    def test_init_accepts_kwargs_with_config(self):
        """
        Test that if `config_inject_mode="as_kwargs"` and config file exists then the 'config' kwarg is passed.

        Regression test.
        See https://github.com/huggingface/huggingface_hub/pull/2058.
        And https://github.com/huggingface/huggingface_hub/pull/2099.
        """
        model = DummyModelFromPretrainedExpectsConfig()
        model.save_pretrained(self.cache_dir, config=CONFIG_AS_DICT)
        with patch.object(
            DummyModelFromPretrainedExpectsConfig,
            "_from_pretrained",
            return_value=DummyModelFromPretrainedExpectsConfig(),
        ) as from_pretrained_mock:
            DummyModelFromPretrainedExpectsConfig.from_pretrained(self.cache_dir)
        assert "config" in from_pretrained_mock.call_args_list[0].kwargs

    def test_init_accepts_kwargs_save_and_load(self):
        model = DummyModelWithKwargs(something="else")
        model.save_pretrained(self.cache_dir)
        assert model._hub_mixin_config == {"something": "else"}

        with patch.object(DummyModelWithKwargs, "__init__", return_value=None) as init_call_mock:
            DummyModelWithKwargs.from_pretrained(self.cache_dir)

        # 'something' is passed to __init__ both as kwarg and in config.
        init_kwargs = init_call_mock.call_args_list[0].kwargs
        assert init_kwargs["something"] == "else"

    def test_save_pretrained_with_push_to_hub(self):
        repo_id = repo_name("save")
        save_directory = self.cache_dir / repo_id

        mocked_model = DummyModelConfigAsDataclass(CONFIG_AS_DATACLASS)
        mocked_model.push_to_hub = Mock()
        mocked_model._save_pretrained = Mock()  # disable _save_pretrained to speed-up

        # Not pushed to hub
        mocked_model.save_pretrained(save_directory)
        mocked_model.push_to_hub.assert_not_called()

        # Push to hub with repo_id (config is pushed)
        mocked_model.save_pretrained(save_directory, push_to_hub=True, repo_id="CustomID")
        mocked_model.push_to_hub.assert_called_with(repo_id="CustomID", config=CONFIG_AS_DICT, model_card_kwargs={})

        # Push to hub with default repo_id (based on dir name)
        mocked_model.save_pretrained(save_directory, push_to_hub=True)
        mocked_model.push_to_hub.assert_called_with(repo_id=repo_id, config=CONFIG_AS_DICT, model_card_kwargs={})

    @patch.object(DummyModelNoConfig, "_from_pretrained")
    def test_from_pretrained_model_id_only(self, from_pretrained_mock: Mock) -> None:
        model = DummyModelNoConfig.from_pretrained("namespace/repo_name")
        from_pretrained_mock.assert_called_once()
        assert model is from_pretrained_mock.return_value

    @patch.object(DummyModelNoConfig, "_from_pretrained")
    def test_from_pretrained_model_id_and_revision(self, from_pretrained_mock: Mock) -> None:
        """Regression test for #1313.
        See https://github.com/huggingface/huggingface_hub/issues/1313."""
        model = DummyModelNoConfig.from_pretrained("namespace/repo_name", revision="123456789")
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
        assert model is from_pretrained_mock.return_value

    def test_from_pretrained_from_relative_path(self):
        with SoftTemporaryDirectory(dir=Path(".")) as tmp_relative_dir:
            relative_save_directory = Path(tmp_relative_dir) / "model"
            DummyModelConfigAsDataclass(config=CONFIG_AS_DATACLASS).save_pretrained(relative_save_directory)
            model = DummyModelConfigAsDataclass.from_pretrained(relative_save_directory)
            assert model._hub_mixin_config == CONFIG_AS_DATACLASS

    def test_from_pretrained_from_absolute_path(self):
        save_directory = self.cache_dir / "subfolder"
        DummyModelConfigAsDataclass(config=CONFIG_AS_DATACLASS).save_pretrained(save_directory)
        model = DummyModelConfigAsDataclass.from_pretrained(save_directory)
        assert model._hub_mixin_config == CONFIG_AS_DATACLASS

    def test_from_pretrained_from_absolute_string_path(self):
        save_directory = str(self.cache_dir / "subfolder")
        DummyModelConfigAsDataclass(config=CONFIG_AS_DATACLASS).save_pretrained(save_directory)
        model = DummyModelConfigAsDataclass.from_pretrained(save_directory)
        assert model._hub_mixin_config == CONFIG_AS_DATACLASS

    def test_push_to_hub(self):
        repo_id = f"{USER}/{repo_name('push_to_hub')}"
        DummyModelConfigAsDataclass(CONFIG_AS_DATACLASS).push_to_hub(repo_id=repo_id, token=TOKEN)

        # Test model id exists
        self._api.model_info(repo_id)

        # Test config has been pushed to hub
        tmp_config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            use_auth_token=TOKEN,
            cache_dir=self.cache_dir,
        )
        with open(tmp_config_path) as f:
            assert json.load(f) == CONFIG_AS_DICT

        # from_pretrained with correct serialization
        from_pretrained_kwargs = {
            "pretrained_model_name_or_path": repo_id,
            "cache_dir": self.cache_dir,
            "api_endpoint": ENDPOINT_STAGING,
            "token": TOKEN,
        }
        for cls in (DummyModelConfigAsDataclass, DummyModelConfigAsOptionalDataclass):
            assert cls.from_pretrained(**from_pretrained_kwargs)._hub_mixin_config == CONFIG_AS_DATACLASS

        for cls in (DummyModelConfigAsDict, DummyModelConfigAsOptionalDict):
            assert cls.from_pretrained(**from_pretrained_kwargs)._hub_mixin_config == CONFIG_AS_DICT

        # Delete repo
        self._api.delete_repo(repo_id=repo_id)

    def test_save_pretrained_do_not_overwrite_new_config(self):
        """Regression test for https://github.com/huggingface/huggingface_hub/issues/2102.

        If `_from_pretrained` does save a config file, we should not overwrite it.
        """
        model = DummyModelSavingConfig()
        model.save_pretrained(self.cache_dir)
        # config.json is not overwritten
        with open(self.cache_dir / "config.json") as f:
            assert json.load(f) == {"custom_config": "custom_config"}

    def test_save_pretrained_does_overwrite_legacy_config(self):
        """Regression test for https://github.com/huggingface/huggingface_hub/issues/2142.

        If a previously existing config file exists, it should be overwritten.
        """
        # Something existing in the cache dir
        (self.cache_dir / "config.json").write_text(json.dumps({"something_legacy": 123}))

        # Save model
        model = DummyModelWithKwargs(a=1, b=2)
        model.save_pretrained(self.cache_dir)

        # config.json IS overwritten
        with open(self.cache_dir / "config.json") as f:
            assert json.load(f) == {"a": 1, "b": 2}

    def test_from_pretrained_when_cls_is_a_dataclass(self):
        """Regression test for #2157.

        When the ModelHubMixin class happens to be a dataclass, `__init__` method will accept `**kwargs` when
        inspecting it. However, due to how dataclasses work, we cannot forward arbitrary kwargs to the `__init__`.
        This test ensures that the `from_pretrained` method does not raise an error when the class is a dataclass.

        See https://github.com/huggingface/huggingface_hub/issues/2157.
        """
        (self.cache_dir / "config.json").write_text('{"foo": 42, "bar": "baz", "other": "value"}')
        model = DummyModelThatIsAlsoADataclass.from_pretrained(self.cache_dir)
        assert model.foo == 42
        assert model.bar == "baz"
        assert not hasattr(model, "other")

    def test_from_cls_with_custom_type(self):
        model = DummyModelWithCustomTypes(1, bar="bar", custom=CustomType("custom"))
        model.save_pretrained(self.cache_dir)

        config = json.loads((self.cache_dir / "config.json").read_text())
        assert config == {
            "foo": 1,
            "bar": "bar",
            "custom": {"value": "custom"},
            "custom_default": {"value": "default"},
        }

        model_reloaded = DummyModelWithCustomTypes.from_pretrained(self.cache_dir)
        assert model_reloaded.foo == 1
        assert model_reloaded.bar == "bar"
        assert model_reloaded.custom.value == "custom"
        assert model_reloaded.custom_default.value == "default"

    def test_inherited_class(self):
        """Test MixinInfo attributes are inherited from the parent class."""
        model = DummyModelInherited()
        assert model._hub_mixin_info.repo_url == "https://hf.co/my-repo"
        assert model._hub_mixin_info.model_card_data.library_name == "my-cool-library"
