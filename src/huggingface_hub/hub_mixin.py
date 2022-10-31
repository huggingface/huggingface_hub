import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests

from .constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME
from .file_download import hf_hub_download, is_torch_available
from .hf_api import HfApi
from .repository import Repository
from .utils import HfFolder, logging, validate_hf_hub_args
from .utils._deprecation import _deprecate_arguments, _deprecate_positional_args


if is_torch_available():
    import torch  # type: ignore

logger = logging.get_logger(__name__)


class ModelHubMixin:
    """
    A generic Hub mixin for machine learning models. Define your own mixin for
    any framework by inheriting from this class and overwriting the
    [`_from_pretrained`] and [`_save_pretrained`] methods to define custom logic
    for saving and loading your classes. See [`PyTorchModelHubMixin`] for an
    example.
    """

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        config: Optional[dict] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save weights in local directory.

        Parameters:
            save_directory (`str` or `Path`):
                Specify directory in which you want to save weights.
            config (`dict`, *optional*):
                Specify config (must be dict) in case you want to save
                it.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after
                saving it. You can specify the repository you want to push to with
                `repo_id` (will default to the name of `save_directory` in your
                namespace).
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

            if (
                # If a deprecated argument is passed, we have to use the deprecated
                # version of `push_to_hub`.
                # TODO: remove this possibility in v0.12
                kwargs.get("repo_url") is not None
                or kwargs.get("repo_path_or_name") is not None
                or kwargs.get("organization") is not None
                or kwargs.get("git_user") is not None
                or kwargs.get("git_email") is not None
                or kwargs.get("skip_lfs_files") is not None
            ):
                if kwargs.get("repo_path_or_name") is None:
                    # Repo name defaults to `save_directory` name
                    kwargs["repo_path_or_name"] = save_directory
            elif kwargs.get("repo_id") is None:
                # Repo name defaults to `save_directory` name
                kwargs["repo_id"] = Path(save_directory).name

            return self.push_to_hub(**kwargs)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Overwrite this method in subclass to define how to save your model.
        """
        raise NotImplementedError

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict] = None,
        token: Optional[Union[str, bool]] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        **model_kwargs,
    ):
        r"""
        Download and instantiate a model from the Hugging Face Hub.

                Parameters:
                    pretrained_model_name_or_path (`str` or `os.PathLike`):
                        Can be either:
                            - A string, the `model id` of a pretrained model
                              hosted inside a model repo on huggingface.co.
                              Valid model ids can be located at the root-level,
                              like `bert-base-uncased`, or namespaced under a
                              user or organization name, like
                              `dbmdz/bert-base-german-cased`.
                            - You can add `revision` by appending `@` at the end
                              of model_id simply like this:
                              `dbmdz/bert-base-german-cased@main` Revision is
                              the specific model version to use. It can be a
                              branch name, a tag name, or a commit id, since we
                              use a git-based system for storing models and
                              other artifacts on huggingface.co, so `revision`
                              can be any identifier allowed by git.
                            - A path to a `directory` containing model weights
                              saved using
                              [`~transformers.PreTrainedModel.save_pretrained`],
                              e.g., `./my_model_directory/`.
                            - `None` if you are both providing the configuration
                              and state dictionary (resp. with keyword arguments
                              `config` and `state_dict`).
                    force_download (`bool`, *optional*, defaults to `False`):
                        Whether to force the (re-)download of the model weights
                        and configuration files, overriding the cached versions
                        if they exist.
                    resume_download (`bool`, *optional*, defaults to `False`):
                        Whether to delete incompletely received files. Will
                        attempt to resume the download if such a file exists.
                    proxies (`Dict[str, str]`, *optional*):
                        A dictionary of proxy servers to use by protocol or
                        endpoint, e.g., `{'http': 'foo.bar:3128',
                        'http://hostname': 'foo.bar:4012'}`. The proxies are
                        used on each request.
                    token (`str` or `bool`, *optional*):
                        The token to use as HTTP bearer authorization for remote
                        files. If `True`, will use the token generated when
                        running `transformers-cli login` (stored in
                        `~/.huggingface`).
                    cache_dir (`Union[str, os.PathLike]`, *optional*):
                        Path to a directory in which a downloaded pretrained
                        model configuration should be cached if the standard
                        cache should not be used.
                    local_files_only(`bool`, *optional*, defaults to `False`):
                        Whether to only look at local files (i.e., do not try to
                        download the model).
                    model_kwargs (`Dict`, *optional*):
                        model_kwargs will be passed to the model during
                        initialization

                <Tip>

                Passing `token=True` is required when you want to use a
                private model.

                </Tip>
        """

        model_id = pretrained_model_name_or_path

        revision = None
        if len(model_id.split("@")) == 2:
            model_id, revision = model_id.split("@")

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
            except requests.exceptions.RequestException:
                logger.warning(f"{CONFIG_NAME} not found in HuggingFace Hub")

        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            model_kwargs.update({"config": config})

        return cls._from_pretrained(
            model_id,
            revision,
            cache_dir,
            force_download,
            proxies,
            resume_download,
            local_files_only,
            token,
            **model_kwargs,
        )

    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        revision,
        cache_dir,
        force_download,
        proxies,
        resume_download,
        local_files_only,
        token,
        **model_kwargs,
    ):
        """Overwrite this method in subclass to define how to load your model from
        pretrained"""
        raise NotImplementedError

    @_deprecate_positional_args(version="0.12")
    @_deprecate_arguments(
        version="0.12",
        deprecated_args={
            "repo_url",
            "repo_path_or_name",
            "organization",
            "git_user",
            "git_email",
            "skip_lfs_files",
        },
    )
    @validate_hf_hub_args
    def push_to_hub(
        self,
        # NOTE: deprecated signature that will change in 0.12
        *,
        repo_path_or_name: Optional[str] = None,
        repo_url: Optional[str] = None,
        commit_message: str = "Add model",
        organization: Optional[str] = None,
        private: bool = False,
        api_endpoint: Optional[str] = None,
        token: Optional[str] = None,
        git_user: Optional[str] = None,
        git_email: Optional[str] = None,
        config: Optional[dict] = None,
        skip_lfs_files: bool = False,
        # NOTE: New arguments since 0.9
        repo_id: Optional[str] = None,  # optional only until 0.12
        branch: Optional[str] = None,
        create_pr: Optional[bool] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        # TODO (release 0.12): signature must be the following
        # repo_id: str,
        # *,
        # commit_message: str = "Add model",
        # private: bool = False,
        # api_endpoint: Optional[str] = None,
        # token: Optional[str] = None,
        # branch: Optional[str] = None,
        # create_pr: Optional[bool] = None,
        # config: Optional[dict] = None,
        # allow_patterns: Optional[Union[List[str], str]] = None,
        # ignore_patterns: Optional[Union[List[str], str]] = None,
    ) -> str:
        """
        Upload model checkpoint to the Hub.

        Use `allow_patterns` and `ignore_patterns` to precisely filter which files
        should be pushed to the hub. See [`upload_folder`] reference for more details.

        Parameters:
            repo_id (`str`, *optional*):
                Repository name to which push.
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
            config (`dict`, *optional*):
                Configuration object to be saved alongside the model weights.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are pushed.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not pushed.

        Returns:
            The url of the commit of your model in the given repository.
        """
        # If the repo id is set, it means we use the new version using HTTP endpoint
        # (introduced in v0.9).
        if repo_id is not None:
            api = HfApi(endpoint=api_endpoint)
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                token=token,
                private=private,
                exist_ok=True,
            )

            # Push the files to the repo in a single commit
            with tempfile.TemporaryDirectory() as tmp:
                saved_path = Path(tmp) / repo_id
                self.save_pretrained(saved_path, config=config)
                return api.upload_folder(
                    repo_id=repo_id,
                    repo_type="model",
                    token=token,
                    folder_path=saved_path,
                    commit_message=commit_message,
                    revision=branch,
                    create_pr=create_pr,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                )

        # If the repo id is None, it means we use the deprecated version using Git
        # TODO: remove code between here and `return repo.git_push()` in release 0.12
        if repo_path_or_name is None and repo_url is None:
            raise ValueError(
                "You need to specify a `repo_path_or_name` or a `repo_url`."
            )

        if token is None and repo_url is None:
            token = HfFolder.get_token()
            if token is None:
                raise ValueError(
                    "You must login to the Hugging Face hub on this computer by typing"
                    " `huggingface-cli login` and entering your credentials to use"
                    " `token=True`. Alternatively, you can pass your own token"
                    " as the `token` argument."
                )
        elif isinstance(token, str):
            token = token
        else:
            token = None

        if repo_path_or_name is None:
            assert repo_url is not None, "A `None` repo URL would have raised above"
            repo_path_or_name = repo_url.split("/")[-1]

        # If no URL is passed and there's no path to a directory containing files, create a repo
        if repo_url is None and not os.path.exists(repo_path_or_name):
            repo_id = Path(repo_path_or_name).name
            if organization:
                repo_id = f"{organization}/{repo_id}"
            repo_url = HfApi(endpoint=api_endpoint).create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                repo_type=None,
                exist_ok=True,
            )

        repo = Repository(
            repo_path_or_name,
            clone_from=repo_url,
            token=token,
            git_user=git_user,
            git_email=git_email,
            skip_lfs_files=skip_lfs_files,
        )
        repo.git_pull(rebase=True)

        # Save the files in the cloned repo
        self.save_pretrained(repo_path_or_name, config=config)

        # Commit and push!
        repo.git_add(auto_lfs_track=True)
        repo.git_commit(commit_message)
        return repo.git_push()


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
    def _from_pretrained(
        cls,
        model_id,
        revision,
        cache_dir,
        force_download,
        proxies,
        resume_download,
        local_files_only,
        token,
        map_location="cpu",
        strict=False,
        **model_kwargs,
    ):
        """
        Overwrite this method to initialize your model in a different way.
        """
        map_location = torch.device(map_location)

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

        state_dict = torch.load(model_file, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()

        return model
