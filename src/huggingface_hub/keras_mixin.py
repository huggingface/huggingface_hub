import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from huggingface_hub import ModelHubMixin
from huggingface_hub.file_download import is_tf_available
from huggingface_hub.snapshot_download import snapshot_download

from .constants import CONFIG_NAME
from .hf_api import HfApi, HfFolder
from .repository import Repository


logger = logging.getLogger(__name__)


def save_pretrained_keras(
    model, save_directory: str, config: Optional[Dict[str, Any]] = None
):
    """Saves a Keras model to save_directory in SavedModel format. Use this if you're using the Functional or Sequential APIs.

    model:
        The Keras model you'd like to save. The model must be compiled and built.
    save_directory (:obj:`str`):
        Specify directory in which you want to save the Keras model.
    config (:obj:`dict`, `optional`):
        Configuration object to be saved alongside the model weights.
    """
    if is_tf_available():
        import tensorflow as tf
    else:
        raise ImportError(
            "Called a Tensorflow-specific function but could not import it."
        )

    if not model.built:
        raise ValueError("Model should be built before trying to save")

    os.makedirs(save_directory, exist_ok=True)

    # saving config
    if config:
        if not isinstance(config, dict):
            raise RuntimeError(
                f"Provided config to save_pretrained_keras should be a dict. Got: '{type(config)}'"
            )
        path = os.path.join(save_directory, CONFIG_NAME)
        with open(path, "w") as f:
            json.dump(config, f)

    tf.keras.models.save_model(model, save_directory)


def from_pretrained_keras(*args, **kwargs):
    return KerasModelHubMixin.from_pretrained(*args, **kwargs)


def push_to_hub_keras(
    model,
    repo_path_or_name: Optional[str] = None,
    repo_url: Optional[str] = None,
    commit_message: Optional[str] = "Add model",
    organization: Optional[str] = None,
    private: Optional[bool] = None,
    api_endpoint: Optional[str] = None,
    use_auth_token: Optional[Union[bool, str]] = True,
    git_user: Optional[str] = None,
    git_email: Optional[str] = None,
    config: Optional[dict] = None,
):
    """
    Upload model checkpoint or tokenizer files to the 🤗 Model Hub while synchronizing a local clone of the repo in
    :obj:`repo_path_or_name`.

    Parameters:
        model:
            The Keras model you'd like to push to the hub. It model must be compiled and built.
        repo_path_or_name (:obj:`str`, `optional`):
            Can either be a repository name for your model or tokenizer in the Hub or a path to a local folder (in
            which case the repository will have the name of that local folder). If not specified, will default to
            the name given by :obj:`repo_url` and a local directory with that name will be created.
        repo_url (:obj:`str`, `optional`):
            Specify this in case you want to push to an existing repository in the hub. If unspecified, a new
            repository will be created in your namespace (unless you specify an :obj:`organization`) with
            :obj:`repo_name`.
        commit_message (:obj:`str`, `optional`):
            Message to commit while pushing. Will default to :obj:`"add model"`.
        organization (:obj:`str`, `optional`):
            Organization in which you want to push your model or tokenizer (you must be a member of this
            organization).
        private (:obj:`bool`, `optional`):
            Whether or not the repository created should be private (requires a paying subscription).
        api_endpoint (:obj:`str`, `optional`):
            The API endpoint to use when pushing the model to the hub.
        use_auth_token (:obj:`bool` or :obj:`str`, `optional`):
            The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
            generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`). Will default to
            :obj:`True`.
        git_user (``str``, `optional`):
            will override the ``git config user.name`` for committing and pushing files to the hub.
        git_email (``str``, `optional`):
            will override the ``git config user.email`` for committing and pushing files to the hub.
        config (:obj:`dict`, `optional`):
            Configuration object to be saved alongside the model weights.

    Returns:
        The url of the commit of your model in the given repository.
    """

    if repo_path_or_name is None and repo_url is None:
        raise ValueError("You need to specify a `repo_path_or_name` or a `repo_url`.")

    if isinstance(use_auth_token, bool) and use_auth_token:
        token = HfFolder.get_token()
    elif isinstance(use_auth_token, str):
        token = use_auth_token
    else:
        token = None

    if token is None:
        raise ValueError(
            "You must login to the Hugging Face hub on this computer by typing `huggingface-cli login` and "
            "entering your credentials to use `use_auth_token=True`. Alternatively, you can pass your own "
            "token as the `use_auth_token` argument."
        )

    if repo_path_or_name is None:
        repo_path_or_name = repo_url.split("/")[-1]

    # If no URL is passed and there's no path to a directory containing files, create a repo
    if repo_url is None and not os.path.exists(repo_path_or_name):
        repo_name = Path(repo_path_or_name).name
        repo_url = HfApi(endpoint=api_endpoint).create_repo(
            repo_name,
            token=token,
            organization=organization,
            private=private,
            repo_type=None,
            exist_ok=True,
        )

    repo = Repository(
        repo_path_or_name,
        clone_from=repo_url,
        use_auth_token=use_auth_token,
        git_user=git_user,
        git_email=git_email,
    )
    repo.git_pull(rebase=True)

    save_pretrained_keras(model, repo_path_or_name, config=config)

    # Commit and push!
    repo.git_add(auto_lfs_track=True)
    repo.git_commit(commit_message)
    return repo.git_push()


class KerasModelHubMixin(ModelHubMixin):
    def __init__(self, *args, **kwargs):
        """
        Mix this class with your keras-model class for ease process of saving & loading from huggingface-hub

        Example::

            >>> from huggingface_hub import KerasModelHubMixin

            >>> class MyModel(tf.keras.Model, KerasModelHubMixin):
            ...    def __init__(self, **kwargs):
            ...        super().__init__()
            ...        self.config = kwargs.pop("config", None)
            ...        self.dummy_inputs = ...
            ...        self.layer = ...
            ...    def call(self, ...)
            ...        return ...

            >>> # Init and compile the model as you normally would
            >>> model = MyModel()
            >>> model.compile(...)
            >>> # Build the graph by training it or passing dummy inputs
            >>> _ = model(model.dummy_inputs)
            >>> # You can save your model like this
            >>> model.save_pretrained("local_model_dir/", push_to_hub=False)
            >>> # Or, you can push to a new public model repo like this
            >>> model.push_to_hub("super-cool-model", git_user="your-hf-username", git_email="you@somesite.com")

            >>> # Downloading weights from hf-hub & model will be initialized from those weights
            >>> model = MyModel.from_pretrained("username/mymodel@main")
        """

    def _save_pretrained(self, save_directory):
        save_pretrained_keras(self, save_directory)

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
        use_auth_token,
        **model_kwargs,
    ):
        """Here we just call from_pretrained_keras function so both the mixin and functional APIs stay in sync.

        TODO - Some args above aren't used since we are calling snapshot_download instead of hf_hub_download.
        """
        if is_tf_available():
            import tensorflow as tf
        else:
            raise ImportError(
                "Called a Tensorflow-specific function but could not import it."
            )

        # TODO - Figure out what to do about these config values. Config is not going to be needed to load model
        cfg = model_kwargs.pop("config", None)

        # Root is either a local filepath matching model_id or a cached snapshot
        if not os.path.isdir(model_id):
            storage_folder = snapshot_download(
                repo_id=model_id, revision=revision, cache_dir=cache_dir
            )
        else:
            storage_folder = model_id

        model = tf.keras.models.load_model(storage_folder, **model_kwargs)

        # For now, we add a new attribute, config, to store the config loaded from the hub/a local dir.
        model.config = cfg

        return model
