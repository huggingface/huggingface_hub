import json
import logging
import os
from typing import Dict, Optional, Union

import requests

from .constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME
from .file_download import cached_download, hf_hub_url, is_torch_available
from .hf_api import HfApi, HfFolder
from .repository import Repository


if is_torch_available():
    import torch


logger = logging.getLogger(__name__)


class ModelHubMixin(object):
    def __init__(self, *args, **kwargs):
        """
        Mix this class with your torch-model class for ease process of saving & loading from huggingface-hub

        Example::

            >>> from huggingface_hub import ModelHubMixin

            >>> class MyModel(nn.Module, ModelHubMixin):
            ...    def __init__(self, **kwargs):
            ...        super().__init__()
            ...        self.config = kwargs.pop("config", None)
            ...        self.layer = ...
            ...    def forward(self, ...)
            ...        return ...

            >>> model = MyModel()
            >>> model.save_pretrained("mymodel", push_to_hub=False) # Saving model weights in the directory
            >>> model.push_to_hub("mymodel", "model-1") # Pushing model-weights to hf-hub

            >>> # Downloading weights from hf-hub & model will be initialized from those weights
            >>> model = MyModel.from_pretrained("username/mymodel@main")
        """

    def save_pretrained(
        self,
        save_directory: str,
        config: Optional[dict] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Saving weights in local directory.

        Parameters:
            save_directory (:obj:`str`):
                Specify directory in which you want to save weights.
            config (:obj:`dict`, `optional`):
                specify config (must be dict) incase you want to save it.
            push_to_hub (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set it to `True` in case you want to push your weights to huggingface_hub
            model_id (:obj:`str`, `optional`, defaults to :obj:`save_directory`):
                Repo name in huggingface_hub. If not specified, repo name will be same as `save_directory`
            kwargs (:obj:`Dict`, `optional`):
                kwargs will be passed to `push_to_hub`
        """

        os.makedirs(save_directory, exist_ok=True)

        # saving config
        if isinstance(config, dict):
            path = os.path.join(save_directory, CONFIG_NAME)
            with open(path, "w") as f:
                json.dump(config, f)

        # saving model weights
        path = os.path.join(save_directory, PYTORCH_WEIGHTS_NAME)
        self._save_pretrained(path)

        if push_to_hub:
            return self.push_to_hub(save_directory, **kwargs)

    def _save_pretrained(self, path):
        """
        Overwrite this method in case you don't want to save complete model, rather some specific layers
        """
        model_to_save = self.module if hasattr(self, "module") else self
        torch.save(model_to_save.state_dict(), path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str],
        strict: bool = True,
        map_location: Optional[str] = "cpu",
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Dict = None,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        **model_kwargs,
    ):
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration from huggingface-hub.
        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated). To
        train the model, you should first set it back in training mode with ``model.train()``.

        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Can be either:
                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - You can add `revision` by appending `@` at the end of model_id simply like this: ``dbmdz/bert-base-german-cased@main``
                      Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id,
                      since we use a git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any identifier allowed by git.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - :obj:`None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments ``config`` and ``state_dict``).
            cache_dir (:obj:`Union[str, os.PathLike]`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str], `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (:obj:`str` or `bool`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
            model_kwargs (:obj:`Dict`, `optional`)::
                model_kwargs will be passed to the model during initialization
        .. note::
            Passing :obj:`use_auth_token=True` is required when you want to use a private model.
        """

        model_id = pretrained_model_name_or_path
        map_location = torch.device(map_location)

        revision = None
        if len(model_id.split("@")) == 2:
            model_id, revision = model_id.split("@")

        if os.path.isdir(model_id) and CONFIG_NAME in os.listdir(model_id):
            config_file = os.path.join(model_id, CONFIG_NAME)
        else:
            try:
                config_url = hf_hub_url(
                    model_id, filename=CONFIG_NAME, revision=revision
                )
                config_file = cached_download(
                    config_url,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                )
            except requests.exceptions.RequestException:
                logger.warning("config.json NOT FOUND in HuggingFace Hub")
                config_file = None

        if os.path.isdir(model_id):
            print("LOADING weights from local directory")
            model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
        else:
            model_url = hf_hub_url(
                model_id, filename=PYTORCH_WEIGHTS_NAME, revision=revision
            )
            model_file = cached_download(
                model_url,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
            )

        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            model_kwargs.update({"config": config})

        model = cls(**model_kwargs)

        state_dict = torch.load(model_file, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()

        return model

    @staticmethod
    def push_to_hub(
        save_directory: Optional[str],
        model_id: Optional[str] = None,
        repo_url: Optional[str] = None,
        commit_message: Optional[str] = "add model",
        organization: Optional[str] = None,
        private: bool = None,
        api_endpoint=None,
        use_auth_token: Union[bool, str, None] = None,
        git_user: Optional[str] = None,
        git_email: Optional[str] = None,
    ) -> str:
        """
        Parameters:
            save_directory (:obj:`Union[str, os.PathLike]`):
                Directory having model weights & config.
            model_id (:obj:`str`, `optional`, defaults to :obj:`save_directory`):
                Repo name in huggingface_hub. If not specified, repo name will be same as `save_directory`
            repo_url (:obj:`str`, `optional`):
                Specify this in case you want to push to existing repo in hub.
            organization (:obj:`str`, `optional`):
                Organization in which you want to push your model.
            private (:obj:`bool`, `optional`):
                private: Whether the model repo should be private (requires a paid huggingface.co account)
            commit_message (:obj:`str`, `optional`, defaults to :obj:`add model`):
                Message to commit while pushing
            api_endpoint (:obj:`str`, `optional`):
                The API endpoint to use when pushing the model to the hub.
            use_auth_token (``str`` or ``bool``, `optional`, defaults ``None``):
                huggingface_token can be extract from ``HfApi().login(username, password)`` and is used to authenticate
                 against the hub (useful from Google Colab for instance).
            git_user (``str``, `optional`, defaults ``None``):
                will override the ``git config user.name`` for committing and pushing files to the hub.
            git_email (``str``, `optional`, defaults ``None``):
                will override the ``git config user.email`` for committing and pushing files to the hub.

        Returns:
            url to commit on remote repo.
        """
        if model_id is None:
            model_id = save_directory.split("/")[-1]

        # The auth token is necessary to create a repo
        if isinstance(use_auth_token, str):
            huggingface_token = use_auth_token
        elif use_auth_token is None and repo_url is not None:
            # If the repo url exists, then no need for a token
            huggingface_token = None
        else:
            huggingface_token = HfFolder.get_token()

        if repo_url is None:
            repo_url = HfApi(endpoint=api_endpoint).create_repo(
                huggingface_token,
                model_id,
                organization=organization,
                private=private,
                repo_type=None,
                exist_ok=True,
            )

        repo = Repository(
            save_directory,
            clone_from=repo_url,
            use_auth_token=use_auth_token,
            git_user=git_user,
            git_email=git_email,
        )

        return repo.push_to_hub(commit_message=commit_message)
