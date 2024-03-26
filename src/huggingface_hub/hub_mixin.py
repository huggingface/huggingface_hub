import inspect
import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, TypeVar, Union, get_args

from .constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME, SAFETENSORS_SINGLE_FILE
from .file_download import hf_hub_download
from .hf_api import HfApi
from .repocard import ModelCard, ModelCardData
from .utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    SoftTemporaryDirectory,
    is_jsonable,
    is_safetensors_available,
    is_torch_available,
    logging,
    validate_hf_hub_args,
)
from .utils._deprecation import _deprecate_arguments


if TYPE_CHECKING:
    from _typeshed import DataclassInstance

if is_torch_available():
    import torch  # type: ignore

if is_safetensors_available():
    from safetensors.torch import load_model as load_model_as_safetensor
    from safetensors.torch import save_model as save_model_as_safetensor


logger = logging.get_logger(__name__)

# Generic variable that is either ModelHubMixin or a subclass thereof
T = TypeVar("T", bound="ModelHubMixin")

DEFAULT_MODEL_CARD = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

This model has been pushed to the Hub using **{{ library_name }}**:
- Repo: {{ repo_url | default("[More Information Needed]", true) }}
- Docs: {{ docs_url | default("[More Information Needed]", true) }}
"""


@dataclass
class MixinInfo:
    library_name: Optional[str] = None
    tags: Optional[List[str]] = None
    repo_url: Optional[str] = None
    docs_url: Optional[str] = None


class ModelHubMixin:
    """
    A generic mixin to integrate ANY machine learning framework with the Hub.

    To integrate your framework, your model class must inherit from this class. Custom logic for saving/loading models
    have to be overwritten in  [`_from_pretrained`] and [`_save_pretrained`]. [`PyTorchModelHubMixin`] is a good example
    of mixin integration with the Hub. Check out our [integration guide](../guides/integrations) for more instructions.

    When inheriting from [`ModelHubMixin`], you can define class-level attributes. These attributes are not passed to
    `__init__` but to the class definition itself. This is useful to define metadata about the library integrating
    [`ModelHubMixin`].

    Args:
        library_name (`str`, *optional*):
            Name of the library integrating ModelHubMixin. Used to generate model card.
        tags (`List[str]`, *optional*):
            Tags to be added to the model card. Used to generate model card.
        repo_url (`str`, *optional*):
            URL of the library repository. Used to generate model card.
        docs_url (`str`, *optional*):
            URL of the library documentation. Used to generate model card.

    Example:

    ```python
    >>> from huggingface_hub import ModelHubMixin

    # Inherit from ModelHubMixin
    >>> class MyCustomModel(
    ...         ModelHubMixin,
    ...         library_name="my-library",
    ...         tags=["x-custom-tag"],
    ...         repo_url="https://github.com/huggingface/my-cool-library",
    ...         docs_url="https://huggingface.co/docs/my-cool-library",
    ...         # ^ optional metadata to generate model card
    ...     ):
    ...     def __init__(self, size: int = 512, device: str = "cpu"):
    ...         # define how to initialize your model
    ...         super().__init__()
    ...         ...
    ...
    ...     def _save_pretrained(self, save_directory: Path) -> None:
    ...         # define how to serialize your model
    ...         ...
    ...
    ...     @classmethod
    ...     def from_pretrained(
    ...         cls: Type[T],
    ...         pretrained_model_name_or_path: Union[str, Path],
    ...         *,
    ...         force_download: bool = False,
    ...         resume_download: bool = False,
    ...         proxies: Optional[Dict] = None,
    ...         token: Optional[Union[str, bool]] = None,
    ...         cache_dir: Optional[Union[str, Path]] = None,
    ...         local_files_only: bool = False,
    ...         revision: Optional[str] = None,
    ...         **model_kwargs,
    ...     ) -> T:
    ...         # define how to deserialize your model
    ...         ...

    >>> model = MyCustomModel(size=256, device="gpu")

    # Save model weights to local directory
    >>> model.save_pretrained("my-awesome-model")

    # Push model weights to the Hub
    >>> model.push_to_hub("my-awesome-model")

    # Download and initialize weights from the Hub
    >>> reloaded_model = MyCustomModel.from_pretrained("username/my-awesome-model")
    >>> reloaded_model._hub_mixin_config
    {"size": 256, "device": "gpu"}

    # Model card has been correctly populated
    >>> from huggingface_hub import ModelCard
    >>> card = ModelCard.load("username/my-awesome-model")
    >>> card.data.tags
    ["x-custom-tag", "pytorch_model_hub_mixin", "model_hub_mixin"]
    >>> card.data.library_name
    "my-library"
    ```
    """

    _hub_mixin_config: Optional[Union[dict, "DataclassInstance"]] = None
    # ^ optional config attribute automatically set in `from_pretrained`
    _hub_mixin_info: MixinInfo
    # ^ information about the library integrating ModelHubMixin (used to generate model card)
    _hub_mixin_init_parameters: Dict[str, inspect.Parameter]
    _hub_mixin_jsonable_default_values: Dict[str, Any]
    _hub_mixin_inject_config: bool
    # ^ internal values to handle config

    def __init_subclass__(
        cls,
        *,
        library_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        repo_url: Optional[str] = None,
        docs_url: Optional[str] = None,
    ) -> None:
        """Inspect __init__ signature only once when subclassing + handle modelcard."""
        super().__init_subclass__()

        # Will be reused when creating modelcard
        tags = tags or []
        tags.append("model_hub_mixin")
        cls._hub_mixin_info = MixinInfo(
            library_name=library_name,
            tags=tags,
            repo_url=repo_url,
            docs_url=docs_url,
        )

        # Inspect __init__ signature to handle config
        cls._hub_mixin_init_parameters = dict(inspect.signature(cls.__init__).parameters)
        cls._hub_mixin_jsonable_default_values = {
            param.name: param.default
            for param in cls._hub_mixin_init_parameters.values()
            if param.default is not inspect.Parameter.empty and is_jsonable(param.default)
        }
        cls._hub_mixin_inject_config = "config" in inspect.signature(cls._from_pretrained).parameters

    def __new__(cls, *args, **kwargs) -> "ModelHubMixin":
        """Create a new instance of the class and handle config.

        3 cases:
        - If `self._hub_mixin_config` is already set, do nothing.
        - If `config` is passed as a dataclass, set it as `self._hub_mixin_config`.
        - Otherwise, build `self._hub_mixin_config` from default values and passed values.
        """
        instance = super().__new__(cls)

        # If `config` is already set, return early
        if instance._hub_mixin_config is not None:
            return instance

        # Infer passed values
        passed_values = {
            **{
                key: value
                for key, value in zip(
                    # [1:] to skip `self` parameter
                    list(cls._hub_mixin_init_parameters)[1:],
                    args,
                )
            },
            **kwargs,
        }

        # If config passed as dataclass => set it and return early
        if is_dataclass(passed_values.get("config")):
            instance._hub_mixin_config = passed_values["config"]
            return instance

        # Otherwise, build config from default + passed values
        init_config = {
            # default values
            **cls._hub_mixin_jsonable_default_values,
            # passed values
            **{key: value for key, value in passed_values.items() if is_jsonable(value)},
        }
        init_config.pop("config", {})

        # Populate `init_config` with provided config
        provided_config = passed_values.get("config")
        if isinstance(provided_config, dict):
            init_config.update(provided_config)

        # Set `config` attribute and return
        if init_config != {}:
            instance._hub_mixin_config = init_config
        return instance

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        config: Optional[Union[dict, "DataclassInstance"]] = None,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        **push_to_hub_kwargs,
    ) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            kwargs:
                Additional key word arguments passed along to the [`~ModelHubMixin.push_to_hub`] method.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Remove config.json if already exists. After `_save_pretrained` we don't want to overwrite config.json
        # as it might have been saved by the custom `_save_pretrained` already. However we do want to overwrite
        # an existing config.json if it was not saved by `_save_pretrained`.
        config_path = save_directory / CONFIG_NAME
        config_path.unlink(missing_ok=True)

        # save model weights/files (framework-specific)
        self._save_pretrained(save_directory)

        # save config (if provided and if not serialized yet in `_save_pretrained`)
        if config is None:
            config = self._hub_mixin_config
        if config is not None:
            if is_dataclass(config):
                config = asdict(config)  # type: ignore[arg-type]
            if not config_path.exists():
                config_str = json.dumps(config, sort_keys=True, indent=2)
                config_path.write_text(config_str)

        # save model card
        model_card_path = save_directory / "README.md"
        if not model_card_path.exists():  # do not overwrite if already exists
            self.generate_model_card().save(save_directory / "README.md")

        # push to the Hub if required
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()  # soft-copy to avoid mutating input
            if config is not None:  # kwarg for `push_to_hub`
                kwargs["config"] = config
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, **kwargs)
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        """
        Overwrite this method in subclass to define how to save your model.
        Check out our [integration guide](../guides/integrations) for instructions.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
        """
        raise NotImplementedError

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls: Type[T],
        pretrained_model_name_or_path: Union[str, Path],
        *,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict] = None,
        token: Optional[Union[str, bool]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        **model_kwargs,
    ) -> T:
        """
        Download a model from the Huggingface Hub and instantiate it.

        Args:
            pretrained_model_name_or_path (`str`, `Path`):
                - Either the `model_id` (string) of a model hosted on the Hub, e.g. `bigscience/bloom`.
                - Or a path to a `directory` containing model weights saved using
                    [`~transformers.PreTrainedModel.save_pretrained`], e.g., `../path/to/my_model_directory/`.
            revision (`str`, *optional*):
                Revision of the model on the Hub. Can be a branch name, a git tag or any commit id.
                Defaults to the latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the model weights and configuration files from the Hub, overriding
                the existing cache.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether to delete incompletely received files. Will attempt to resume the download if such a file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on every request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the local cached file if it exists.
            model_kwargs (`Dict`, *optional*):
                Additional kwargs to pass to the model during initialization.
        """
        model_id = str(pretrained_model_name_or_path)
        config_file: Optional[str] = None
        if os.path.isdir(model_id):
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                logger.warning(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                logger.info(f"{CONFIG_NAME} not found on the HuggingFace Hub: {str(e)}")

        # Read config
        config = None
        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Populate model_kwargs from config
            for param in cls._hub_mixin_init_parameters.values():
                if param.name not in model_kwargs and param.name in config:
                    model_kwargs[param.name] = config[param.name]

            # Check if `config` argument was passed at init
            if "config" in cls._hub_mixin_init_parameters:
                # Check if `config` argument is a dataclass
                config_annotation = cls._hub_mixin_init_parameters["config"].annotation
                if config_annotation is inspect.Parameter.empty:
                    pass  # no annotation
                elif is_dataclass(config_annotation):
                    config = _load_dataclass(config_annotation, config)
                else:
                    # if Optional/Union annotation => check if a dataclass is in the Union
                    for _sub_annotation in get_args(config_annotation):
                        if is_dataclass(_sub_annotation):
                            config = _load_dataclass(_sub_annotation, config)
                            break

                # Forward config to model initialization
                model_kwargs["config"] = config

            # Inject config if `**kwargs` are expected
            if is_dataclass(cls):
                for key in cls.__dataclass_fields__:
                    if key not in model_kwargs and key in config:
                        model_kwargs[key] = config[key]
            elif any(param.kind == inspect.Parameter.VAR_KEYWORD for param in cls._hub_mixin_init_parameters.values()):
                for key, value in config.items():
                    if key not in model_kwargs:
                        model_kwargs[key] = value

            # Finally, also inject if `_from_pretrained` expects it
            if cls._hub_mixin_inject_config:
                model_kwargs["config"] = config

        instance = cls._from_pretrained(
            model_id=str(model_id),
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            **model_kwargs,
        )

        # Implicitly set the config as instance attribute if not already set by the class
        # This way `config` will be available when calling `save_pretrained` or `push_to_hub`.
        if config is not None and (getattr(instance, "_hub_mixin_config", None) in (None, {})):
            instance._hub_mixin_config = config

        return instance

    @classmethod
    def _from_pretrained(
        cls: Type[T],
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
    ) -> T:
        """Overwrite this method in subclass to define how to load your model from pretrained.

        Use [`hf_hub_download`] or [`snapshot_download`] to download files from the Hub before loading them. Most
        args taken as input can be directly passed to those 2 methods. If needed, you can add more arguments to this
        method using "model_kwargs". For example [`PyTorchModelHubMixin._from_pretrained`] takes as input a `map_location`
        parameter to set on which device the model should be loaded.

        Check out our [integration guide](../guides/integrations) for more instructions.

        Args:
            model_id (`str`):
                ID of the model to load from the Huggingface Hub (e.g. `bigscience/bloom`).
            revision (`str`, *optional*):
                Revision of the model on the Hub. Can be a branch name, a git tag or any commit id. Defaults to the
                latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the model weights and configuration files from the Hub, overriding
                the existing cache.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether to delete incompletely received files. Will attempt to resume the download if such a file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint (e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the local cached file if it exists.
            model_kwargs:
                Additional keyword arguments passed along to the [`~ModelHubMixin._from_pretrained`] method.
        """
        raise NotImplementedError

    @_deprecate_arguments(
        version="0.23.0",
        deprecated_args=["api_endpoint"],
        custom_message="Use `HF_ENDPOINT` environment variable instead.",
    )
    @validate_hf_hub_args
    def push_to_hub(
        self,
        repo_id: str,
        *,
        config: Optional[Union[dict, "DataclassInstance"]] = None,
        commit_message: str = "Push model using huggingface_hub.",
        private: bool = False,
        token: Optional[str] = None,
        branch: Optional[str] = None,
        create_pr: Optional[bool] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        delete_patterns: Optional[Union[List[str], str]] = None,
        # TODO: remove once deprecated
        api_endpoint: Optional[str] = None,
    ) -> str:
        """
        Upload model checkpoint to the Hub.

        Use `allow_patterns` and `ignore_patterns` to precisely filter which files should be pushed to the hub. Use
        `delete_patterns` to delete existing remote files in the same commit. See [`upload_folder`] reference for more
        details.

        Args:
            repo_id (`str`):
                ID of the repository to push to (example: `"username/my-model"`).
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
            commit_message (`str`, *optional*):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `False`):
                Whether the repository created should be private.
            api_endpoint (`str`, *optional*):
                The API endpoint to use when pushing the model to the hub.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            branch (`str`, *optional*):
                The git branch on which to push the model. This defaults to `"main"`.
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request from `branch` with that commit. Defaults to `False`.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are pushed.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not pushed.
            delete_patterns (`List[str]` or `str`, *optional*):
                If provided, remote files matching any of the patterns will be deleted from the repo.

        Returns:
            The url of the commit of your model in the given repository.
        """
        api = HfApi(endpoint=api_endpoint, token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

        # Push the files to the repo in a single commit
        with SoftTemporaryDirectory() as tmp:
            saved_path = Path(tmp) / repo_id
            self.save_pretrained(saved_path, config=config)
            return api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message=commit_message,
                revision=branch,
                create_pr=create_pr,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
            )

    def generate_model_card(self, *args, **kwargs) -> ModelCard:
        card = ModelCard.from_template(
            card_data=ModelCardData(**asdict(self._hub_mixin_info)),
            template_str=DEFAULT_MODEL_CARD,
        )
        return card


class PyTorchModelHubMixin(ModelHubMixin):
    """
    Implementation of [`ModelHubMixin`] to provide model Hub upload/download capabilities to PyTorch models. The model
    is set in evaluation mode by default using `model.eval()` (dropout modules are deactivated). To train the model,
    you should first set it back in training mode with `model.train()`.

    Example:

    ```python
    >>> import torch
    >>> import torch.nn as nn
    >>> from huggingface_hub import PyTorchModelHubMixin

    >>> class MyModel(
    ...         nn.Module,
    ...         PyTorchModelHubMixin,
    ...         library_name="keras-nlp",
    ...         repo_url="https://github.com/keras-team/keras-nlp",
    ...         docs_url="https://keras.io/keras_nlp/",
    ...         # ^ optional metadata to generate model card
    ...     ):
    ...     def __init__(self, hidden_size: int = 512, vocab_size: int = 30000, output_size: int = 4):
    ...         super().__init__()
    ...         self.param = nn.Parameter(torch.rand(hidden_size, vocab_size))
    ...         self.linear = nn.Linear(output_size, vocab_size)

    ...     def forward(self, x):
    ...         return self.linear(x + self.param)
    >>> model = MyModel(hidden_size=256)

    # Save model weights to local directory
    >>> model.save_pretrained("my-awesome-model")

    # Push model weights to the Hub
    >>> model.push_to_hub("my-awesome-model")

    # Download and initialize weights from the Hub
    >>> model = MyModel.from_pretrained("username/my-awesome-model")
    >>> model.hidden_size
    256
    ```
    """

    def __init_subclass__(cls, *args, tags: Optional[List[str]] = None, **kwargs) -> None:
        tags = tags or []
        tags.append("pytorch_model_hub_mixin")
        kwargs["tags"] = tags
        return super().__init_subclass__(*args, **kwargs)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights from a Pytorch model to a local directory."""
        model_to_save = self.module if hasattr(self, "module") else self  # type: ignore
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

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
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""
        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            return cls._load_as_safetensor(model, model_file, map_location, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_safetensor(model, model_file, map_location, strict)
            except EntryNotFoundError:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=PYTORCH_WEIGHTS_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_pickle(model, model_file, map_location, strict)

    @classmethod
    def _load_as_pickle(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        state_dict = torch.load(model_file, map_location=torch.device(map_location))
        model.load_state_dict(state_dict, strict=strict)  # type: ignore
        model.eval()  # type: ignore
        return model

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        load_model_as_safetensor(model, model_file, strict=strict)  # type: ignore [arg-type]
        if map_location != "cpu":
            # TODO: remove this once https://github.com/huggingface/safetensors/pull/449 is merged.
            logger.warning(
                "Loading model weights on other devices than 'cpu' is not supported natively."
                " This means that the model is loaded on 'cpu' first and then copied to the device."
                " This leads to a slower loading time."
                " Support for loading directly on other devices is planned to be added in future releases."
                " See https://github.com/huggingface/huggingface_hub/pull/2086 for more details."
            )
            model.to(map_location)  # type: ignore [attr-defined]
        return model


def _load_dataclass(datacls: Type["DataclassInstance"], data: dict) -> "DataclassInstance":
    """Load a dataclass instance from a dictionary.

    Fields not expected by the dataclass are ignored.
    """
    return datacls(**{k: v for k, v in data.items() if k in datacls.__dataclass_fields__})
