import os
import json
from .file_download import (
    hf_hub_url, 
    cached_download,
    CONFIG_NAME,
    PYTORCH_WEIGHTS_NAME
    )

import torch
import subprocess
import os

PREFIX = "https://huggingface.co/"

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
            >>> # train your model using whatever trainer you link
            >>> # Saving model-weights in required format in specified directory
            >>> model.save_pretrained("mymodel")
            >>> # Pushing model-weights to hf-hub
            >>> model.upload_to_hub("username/mymodel")
            >>> # Downloading weights from hf-hub & model will be initialized from those weights
            >>> model = MyModel.from_pretrained("username/mymodel")
        """

    def save_pretrained(self, save_directory:str):
        """
            Saving weights in local directory that can be loaded pushed directly to huggingface-hub
        """
        os.makedirs(save_directory, exist_ok=True)

        self.wts_directory = save_directory

        # saving config
        if hasattr(self, 'config'):
            path = os.path.join(save_directory, CONFIG_NAME)
            with open(path, "w") as f:
                json.dump(self.config, f)

        # saving model weights
        path = os.path.join(save_directory, PYTORCH_WEIGHTS_NAME)
        self._save_pretrained(path)

    def _save_pretrained(self, path):
        """
            Overwrite this method in case you don't want to save complete model, rather some specific layers
        """
        model_to_save = self.module if hasattr(self, "module") else self
        torch.save(model_to_save.state_dict(), path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path:str, *model_args, **kwargs):
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
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
        .. note::
            Passing :obj:`use_auth_token=True` is required when you want to use a private model.
        """

        model_id = pretrained_model_name_or_path
        strict = kwargs.pop("strict", True)
        map_location = kwargs.pop("map_location", torch.device("cpu"))
        revision = kwargs.pop("revision", None)
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

        if name in os.listdir() and CONFIG_NAME in os.listdir(name):
            print("LOADING weights from local directory")
            config_file = os.path.join(name, CONFIG_NAME)
        else:
            try:
                config_url = hf_hub_url(model_id, filename=CONFIG_NAME, revision=revision)
                config_file = cached_download(config_url, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, use_auth_token=use_auth_token)
            except:
                config_file = None

        if name in os.listdir():
            print("LOADING weights from local directory")
            model_file = os.path.join(name, PYTORCH_WEIGHTS_NAME)
        else:
            model_url = hf_hub_url(model_id, filename=PYTORCH_WEIGHTS_NAME, revision=revision)
            model_file = cached_download(model_url, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, use_auth_token=use_auth_token)

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

    def upload_to_hub(self, model_id:str, **kwargs):
        """
            This method will upload your model weights to huggingface-hub.

            Parameters:
                model_id (:obj:`str`):
                    model_id which will be used later using from_pretrained (Generally: username/model_name)   
                wts_directory (:obj:`str`, `optional`):  
                    Directory having model-weights & config. Specify this only incase you are not saving using `.save_pretrained()` method
                branch (:obj:`str`, `optional`):
                    Since huggingface-hub is relying on git-lfs for versioning weights, you can specify branch in which you want to commit
                commit_message(:obj:`str`, `optional`):
                    your commit message to hub
                track_extensions(:obj:`list`, `optional`):
                    extensions of large files which should be tracked with git-lfs

            NOTE:
                - You need to have `git-lfs` installed
                    Ubuntu: `sudo apt-get install git-lfs`
                    Gcolab: `!sudo apt-get install git-lfs`
                    MAC-OS: `brew install git-lfs`
                - You need to create repository in huggingface-hub manually
        """

        wts_directory = kwargs.pop("wts_directory", None)
        branch = kwargs.pop("branch", "main")
        commit_message = kwargs.pop("commit_message", "add model")
        track_extensions = kwargs.pop("track_extensions", ["*.bin", "*pt", ".h5"])

        if wts_directory is None:
            wts_directory = self.wts_directory if hasattr(self, "wts_directory") else wts_directory

        if not os.path.isdir(wts_directory):
            raise FileExistsError(f"{wts_directory} doesn't exist")

        os.chdir(wts_directory)
        if not os.path.isdir(".git"):
            subprocess.run(["git", "init"], stdout=subprocess.PIPE)

        if not os.path.isdir(".git/lfs"):
            subprocess.run(["git-lfs", "install"], stdout=subprocess.PIPE)

        with open(".git/config", "r") as f:
            git_config = f.read().split()
        if "url" not in git_config:
            subprocess.run(["git", "remote", "add", "origin", PREFIX+model_id], stdout=subprocess.PIPE)

        if not branch in os.listdir(".git/refs/heads"):
            subprocess.run(["git", "checkout", "-b", branch], stdout=subprocess.PIPE)

        if (branch == "main") and ("url" not in git_config):
            subprocess.run(["git", "pull", "origin", "main"], stdout=subprocess.PIPE)

        subprocess.run(["git-lfs", "track"]+track_extensions, stdout=subprocess.PIPE)

        subprocess.run(["git", "add", "."], stdout=subprocess.PIPE)
        subprocess.run(["git", "commit", "-m", commit_message], stdout=subprocess.PIPE)

        subprocess.run(["git", "push", "origin", branch])
        os.chdir("../")
