import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests

from .constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME
from .file_download import hf_hub_download, is_torch_available
from .hf_api import HfApi
from .utils import SoftTemporaryDirectory, logging, validate_hf_hub_args
from .utils._deprecation import _deprecate_positional_args


if is_torch_available():
    import torch  # type: ignore

logger = logging.get_logger(__name__)


class ModelHubMixin:
    """
    A generic Hub mixin for machine learning models. Define your own mixin for any framework
    by inheriting from this class and overwriting the [`_from_pretrained`] and [`_save_pretrained`]
    methods to define custom logic for saving and loading your classes.

    See [`PyTorchModelHubMixin`] for an example.
    """

    @_deprecate_positional_args(version="0.16")
    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        config: Optional[dict] = None,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save weights in local directory.

        Parameters:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            config (`dict`, *optional*):
                Model configuration specified as a key/value dictionary.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will
                default to the folder name if not provided.
            kwargs:
                Additional key word arguments passed along to the
                [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        os.makedirs(save_directory, exist_ok=True)

        # saving model weights/files
        self._save_pretrained(save_directory)

        # saving config
        if isinstance(config, dict):
            path = os.path.join(save_directory, CONFIG_NAME)
            with open(path, "w") as f:
                json.dump(config, f)

        if push_to_hub:
            kwargs = kwargs.copy()  # soft-copy to avoid mutating input
            if config is not None:  # kwarg for `push_to_hub`
                kwargs["config"] = config

            if repo_id is None:
                # Repo name defaults to `save_directory` name
                repo_id = Path(save_directory).name

            return self.push_to_hub(repo_id=repo_id, **kwargs)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Overwrite this method in subclass to define how to save your model.
        """
        raise NotImplementedError

    @classmethod
    @validate_hf_hub_args
    @_deprecate_positional_args(version="0.16")
    def from_pretrained(
        cls,
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
    ):
        """
        Download a model from the Huggingface Hub and instantiate it.

        Parameters:
            pretrained_model_name_or_path (`str`, `Path`):
                - Either the `model_id` (string) of a model hosted on the Hub, e.g. `bigscience/bloom`.
                - Or a path to a `directory` containing model weights saved using
                    [`~transformers.PreTrainedModel.save_pretrained`], e.g., `../path/to/my_model_directory/`.
            revision (`str`, *optional*):
                Revision of the model on the Hub. Can be a branch name, a git tag or any commit id.
                Defaults to the latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the model weights and configuration
                files from the Hub, overriding the existing cache.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether to delete incompletely received files. Will attempt to resume the
                download if such a file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on every request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default,
                it will use the token cached when running `huggingface-cli login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the
                local cached file if it exists.
            model_kwargs (`Dict`, *optional*):
                Additional kwargs to pass to the model during initialization.
        """
        model_id = pretrained_model_name_or_path

        if isinstance(model_id, str) and len(model_id.split("@")) == 2:
            warnings.warn(
                "Passing a revision using 'namespace/model_id@revision' pattern is"
                " deprecated and will be removed in version v0.16. Please pass"
                " 'revision=...' as argument.",
                FutureWarning,
            )
            model_id, revision = model_id.split("@")

        config_file: Optional[str] = None
        if os.path.isdir(model_id):
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                logger.warning(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif isinstance(model_id, str):
            try:
                config_file = hf_hub_download(
                    repo_id=str(model_id),
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except requests.exceptions.RequestException:
                logger.warning(f"{CONFIG_NAME} not found in HuggingFace Hub.")

        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            model_kwargs.update({"config": config})

        return cls._from_pretrained(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            **model_kwargs,
        )

    @classmethod
    @_deprecate_positional_args(version="0.16")
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str,
        cache_dir: str,
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        **model_kwargs,
    ):
        """Overwrite this method in subclass to define how to load your model from
        pretrained"""
        raise NotImplementedError

    @validate_hf_hub_args
    def push_to_hub(
        self,
        repo_id: str,
        *,
        config: Optional[dict] = None,
        commit_message: str = "Push model using huggingface_hub.",
        private: bool = False,
        api_endpoint: Optional[str] = None,
        token: Optional[str] = None,
        branch: Optional[str] = None,
        create_pr: Optional[bool] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
    ) -> str:
        """
        Upload model checkpoint to the Hub.

        Use `allow_patterns` and `ignore_patterns` to precisely filter which files
        should be pushed to the hub. See [`upload_folder`] reference for more details.

        Parameters:
            repo_id (`str`):
                Repository name to which push.
            config (`dict`, *optional*):
                Configuration object to be saved alongside the model weights.
            commit_message (`str`, *optional*):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `False`):
                Whether the repository created should be private.
            api_endpoint (`str`, *optional*):
                The API endpoint to use when pushing the model to the hub.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files.
                If not set, will use the token set when logging in with
                `transformers-cli login` (stored in `~/.huggingface`).
            branch (`str`, *optional*):
                The git branch on which to push the model. This defaults to
                the default branch as specified in your repository, which
                defaults to `"main"`.
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request from `branch` with that commit.
                Defaults to `False`.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are pushed.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not pushed.

        Returns:
            The url of the commit of your model in the given repository.
        """
        api = HfApi(endpoint=api_endpoint, token=token)
        api.create_repo(
            repo_id=repo_id, repo_type="model", private=private, exist_ok=True
        )

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
            )


class PyTorchModelHubMixin(ModelHubMixin):
    """
    Implementation of [`ModelHubMixin`] to provide model Hub upload/download
    capabilities to PyTorch models. The model is set in evaluation mode by
    default using `model.eval()` (dropout modules are deactivated). To train
    the model, you should first set it back in training mode with
    `model.train()`.

    Example:

    ```python
    >>> import torch
    >>> import torch.nn as nn
    >>> from huggingface_hub import PyTorchModelHubMixin


    >>> class MyModel(nn.Module, PyTorchModelHubMixin):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.param = nn.Parameter(torch.rand(3, 4))
    ...         self.linear = nn.Linear(4, 5)

    ...     def forward(self, x):
    ...         return self.linear(x + self.param)


    >>> model = MyModel()
    >>> # Save model weights to local directory
    >>> model.save_pretrained("my-awesome-model")
    >>> # Push model weights to the Hub
    >>> model.push_to_hub("my-awesome-model")
    >>> # Download and initialize weights from the Hub
    >>> model = MyModel.from_pretrained("username/my-awesome-model")
    ```
    """

    def _save_pretrained(self, save_directory):
        """
        Overwrite this method if you wish to save specific layers instead of the
        complete model.
        """
        path = os.path.join(save_directory, PYTORCH_WEIGHTS_NAME)
        model_to_save = self.module if hasattr(self, "module") else self
        torch.save(model_to_save.state_dict(), path)

    @classmethod
    @_deprecate_positional_args(version="0.16")
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str,
        cache_dir: str,
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
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
        else:
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
        model = cls(**model_kwargs)

        state_dict = torch.load(model_file, map_location=torch.device(map_location))
        model.load_state_dict(state_dict, strict=strict)  # type: ignore
        model.eval()  # type: ignore

        return model
