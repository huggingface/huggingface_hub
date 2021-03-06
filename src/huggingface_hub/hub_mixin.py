import json
import os

import torch

from .constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME
from .file_download import cached_download, hf_hub_url
from .hf_api import HfApi, HfFolder
from .repository import Repository


class ModelHubMixin(object):
    def __init__(self, *args, **kwargs):
        """
        Mix this class with your torch-model class for ease process of saving & loading from huggingface-hub

        Example::

            >>> from huggingface_hub import ModelHubMixin

            >>> class MyModel(nn.Module, ModelHubMixin):
            ...    def __init__(self):
            ...        super().__init__()
            ...        self.layer = ...
            ...    def forward(self, ...)
            ...        return ...

            >>> model = MyModel()
            >>> model.save_pretrained("mymodel") # Saving model weights in the directory
            >>> model.upload_to_hub("mymodel", "model-1") # Pushing model-weights to hf-hub

            >>> # Downloading weights from hf-hub & model will be initialized from those weights
            >>> model = MyModel.from_pretrained("username/mymodel")
        """

    def save_pretrained(
        self, save_directory: str, config: dict = None, push_to_hub=False, model_id=None
    ):
        """
        Saving weights in local directory.

        Parameters:
            save_directory (:obj:`str`):
                Directory in which you want to save weights.
            config (:obj:`dict`, `optional`):
                specify config incase you want to save config.
        """
        os.makedirs(save_directory, exist_ok=True)
        config_to_save = None

        # saving config
        if hasattr(self, "config"):
            if isinstance(self.config, dict):
                config_to_save = self.config

        if isinstance(config, dict):
            config_to_save = config

        if config_to_save is not None:
            path = os.path.join(save_directory, CONFIG_NAME)
            with open(path, "w") as f:
                json.dump(config_to_save, f)

        # saving model weights
        path = os.path.join(save_directory, PYTORCH_WEIGHTS_NAME)
        self._save_pretrained(path)

        if model_id is None:
            model_id = save_directory

        if push_to_hub:
            self.push_to_hub(save_directory, model_id)

    def _save_pretrained(self, path):
        """
        Overwrite this method in case you don't want to save complete model, rather some specific layers
        """
        model_to_save = self.module if hasattr(self, "module") else self
        torch.save(model_to_save.state_dict(), path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
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
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
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
        .. note::
            Passing :obj:`use_auth_token=True` is required when you want to use a private model.
        """

        model_id = pretrained_model_name_or_path
        strict = kwargs.pop("strict", True)
        map_location = kwargs.pop("map_location", torch.device("cpu"))
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", False)

        if len(model_id.split("/")) == 1:
            name = model_id
        else:
            _, name = model_id.split("/")

        revision = "main"
        if len(name.split("@")) > 1:
            name, revision = name.split("@")

        if name in os.listdir() and CONFIG_NAME in os.listdir(name):
            print("LOADING weights from local directory")
            config_file = os.path.join(name, CONFIG_NAME)
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
            except:
                config_file = None

        if name in os.listdir():
            print("LOADING weights from local directory")
            model_file = os.path.join(name, PYTORCH_WEIGHTS_NAME)
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
            model = cls(config, **kwargs)
        else:
            model = cls(**kwargs)

        state_dict = torch.load(model_file, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()

        return model

    @staticmethod
    def push_to_hub(weights_directory: str, model_id: str, **kwargs):
        """
        Parameters:
            weights_directory (:obj:`Union[str, os.PathLike]`):
                Directory having model-weights & config.
            model_id is like ``bert-base-uncased@main``
        """
        repo_url = kwargs.pop("repo_url", None)

        commit_message = kwargs.pop("commit_message", "add model")
        organization = kwargs.pop("organization", None)
        private = kwargs.pop("private", None)

        revision = "main"
        if len(model_id.split("@")) > 1:
            model_id, revision = model_id.split("@")

        token = HfFolder.get_token()
        if repo_url is None:
            repo_url = HfApi().create_repo(
                token,
                model_id,
                organization=organization,
                private=private,
                repo_type=None,
                exist_ok=True,
            )

        repo = Repository(weights_directory, clone_from=repo_url, use_auth_token=token)

        return repo.push_to_hub(commit_message=commit_message)
