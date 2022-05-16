import json
import os
import warnings
from pathlib import Path
from shutil import copytree, rmtree
from typing import Any, Dict, Optional, Union

import yaml
from huggingface_hub import ModelHubMixin
from huggingface_hub.file_download import (
    get_tf_version,
    is_graphviz_available,
    is_pydot_available,
    is_tf_available,
)
from huggingface_hub.snapshot_download import snapshot_download

from .constants import CONFIG_NAME
from .hf_api import HfApi, HfFolder
from .repository import Repository
from .utils import logging


logger = logging.get_logger(__name__)

if is_tf_available():
    import tensorflow as tf


def _extract_hyperparameters_from_keras(model):
    if model.optimizer is not None:
        hyperparameters = dict()
        hyperparameters["optimizer"] = model.optimizer.get_config()
        hyperparameters[
            "training_precision"
        ] = tf.keras.mixed_precision.global_policy().name
    else:
        hyperparameters = None
    return hyperparameters


def _parse_model_history(model, save_directory):
    lines = None
    if model.history is not None:
        if model.history.history != {}:
            path = os.path.join(save_directory, "history.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(model.history.history, f, indent=2, sort_keys=True)
            lines = []
            logs = model.history.history
            num_epochs = len(logs["loss"])

            for value in range(num_epochs):
                epoch_dict = {
                    log_key: log_value_list[value]
                    for log_key, log_value_list in logs.items()
                }
                values = dict()
                for k, v in epoch_dict.items():
                    if k.startswith("val_"):
                        k = "validation_" + k[4:]
                    elif k != "epoch":
                        k = "train_" + k
                    splits = k.split("_")
                    name = " ".join([part.capitalize() for part in splits])
                    values[name] = v
                lines.append(values)
    return lines


def _plot_network(model, save_directory):
    tf.keras.utils.plot_model(
        model,
        to_file=f"{save_directory}/model.png",
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
        layer_range=None,
    )


def _write_metrics(model, model_card, save_directory):
    lines = _parse_model_history(model, save_directory)
    if lines is not None:
        model_card += "\n| Epochs |"

        for i in lines[0].keys():
            model_card += f" {i} |"
        model_card += "\n |"
        for i in range(len(lines[0].keys()) + 1):
            model_card += "--- |"  # add header of table
        for line in lines:
            model_card += f"\n| {lines.index(line) + 1}|"  # add values
            for key in line:
                value = round(line[key], 3)
                model_card += f" {value}| "
    else:
        model_card += "Model history needed"
    return model_card


def _create_model_card(
    model,
    repo_dir: Path,
    plot_model: Optional[bool] = True,
    metadata: Optional[dict] = None,
):
    """
    Creates a model card for the repository.
    """
    hyperparameters = _extract_hyperparameters_from_keras(model)
    if plot_model and is_graphviz_available() and is_pydot_available():
        _plot_network(model, repo_dir)
    readme_path = f"{repo_dir}/README.md"
    metadata["library_name"] = "keras"
    model_card = "---\n"
    model_card += yaml.dump(metadata, default_flow_style=False)
    model_card += "---\n"
    model_card += "\n## Model description\n\nMore information needed\n"
    model_card += "\n## Intended uses & limitations\n\nMore information needed\n"
    model_card += "\n## Training and evaluation data\n\nMore information needed\n"
    if hyperparameters is not None:
        model_card += "\n## Training procedure\n"
        model_card += "\n### Training hyperparameters\n"
        model_card += "\nThe following hyperparameters were used during training:\n"
        model_card += "\n".join(
            [f"- {name}: {value}" for name, value in hyperparameters.items()]
        )
        model_card += "\n"
    model_card += "\n ## Training Metrics\n"
    model_card = _write_metrics(model, model_card, repo_dir)
    if plot_model and os.path.exists(f"{repo_dir}/model.png"):
        model_card += "\n ## Model Plot\n"
        model_card += "\n<details>"
        model_card += "\n<summary>View Model Plot</summary>\n"
        path_to_plot = "./model.png"
        model_card += f"\n![Model Image]({path_to_plot})\n"
        model_card += "\n</details>"

    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = model_card
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)


def save_pretrained_keras(
    model,
    save_directory: str,
    config: Optional[Dict[str, Any]] = None,
    include_optimizer: Optional[bool] = False,
    plot_model: Optional[bool] = True,
    tags: Optional[Union[list, str]] = None,
    **model_save_kwargs,
):
    """
    Saves a Keras model to save_directory in SavedModel format. Use this if
    you're using the Functional or Sequential APIs.

    Args:
        model (`Keras.Model`):
            The [Keras
            model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
            you'd like to save. The model must be compiled and built.
        save_directory (`str`):
            Specify directory in which you want to save the Keras model.
        config (`dict`, *optional*):
            Configuration object to be saved alongside the model weights.
        include_optimizer(`bool`, *optional*, defaults to `False`):
            Whether or not to include optimizer in serialization.
        plot_model (`bool`, *optional*, defaults to `True`):
            Setting this to `True` will plot the model and put it in the model
            card. Requires graphviz and pydot to be installed.
        tags (Union[`str`,`list`], *optional*):
            List of tags that are related to model or string of a single tag. See example tags
            [here](https://github.com/huggingface/hub-docs/blame/main/modelcard.md).
        model_save_kwargs(`dict`, *optional*):
            model_save_kwargs will be passed to
            [`tf.keras.models.save_model()`](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model).
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
                "Provided config to save_pretrained_keras should be a dict. Got:"
                f" '{type(config)}'"
            )
        path = os.path.join(save_directory, CONFIG_NAME)
        with open(path, "w") as f:
            json.dump(config, f)

    metadata = {}
    if isinstance(tags, list):
        metadata["tags"] = tags
    elif isinstance(tags, str):
        metadata["tags"] = [tags]

    task_name = model_save_kwargs.pop("task_name", None)
    if task_name is not None:
        warnings.warn(
            "`task_name` input argument is deprecated. Pass `tags` instead.",
            FutureWarning,
        )
        if "tags" in metadata:
            metadata["tags"].append(task_name)
        else:
            metadata["tags"] = [task_name]

    _create_model_card(model, save_directory, plot_model, metadata)
    tf.keras.models.save_model(
        model, save_directory, include_optimizer=include_optimizer, **model_save_kwargs
    )


def from_pretrained_keras(*args, **kwargs):
    r"""
    Instantiate a pretrained Keras model from a pre-trained model from the Hub.
    The model is expected to be in SavedModel format.```

    Parameters:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            Can be either:
                - A string, the `model id` of a pretrained model hosted inside a
                  model repo on huggingface.co. Valid model ids can be located
                  at the root-level, like `bert-base-uncased`, or namespaced
                  under a user or organization name, like
                  `dbmdz/bert-base-german-cased`.
                - You can add `revision` by appending `@` at the end of model_id
                  simply like this: `dbmdz/bert-base-german-cased@main` Revision
                  is the specific model version to use. It can be a branch name,
                  a tag name, or a commit id, since we use a git-based system
                  for storing models and other artifacts on huggingface.co, so
                  `revision` can be any identifier allowed by git.
                - A path to a `directory` containing model weights saved using
                  [`~transformers.PreTrainedModel.save_pretrained`], e.g.,
                  `./my_model_directory/`.
                - `None` if you are both providing the configuration and state
                  dictionary (resp. with keyword arguments `config` and
                  `state_dict`).
        force_download (`bool`, *optional*, defaults to `False`):
            Whether to force the (re-)download of the model weights and
            configuration files, overriding the cached versions if they exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether to delete incompletely received files. Will attempt to
            resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g.,
            `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The
            proxies are used on each request.
        use_auth_token (`str` or `bool`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If
            `True`, will use the token generated when running `transformers-cli
            login` (stored in `~/.huggingface`).
        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory in which a downloaded pretrained model
            configuration should be cached if the standard cache should not be
            used.
        local_files_only(`bool`, *optional*, defaults to `False`):
            Whether to only look at local files (i.e., do not try to download
            the model).
        model_kwargs (`Dict`, *optional*):
            model_kwargs will be passed to the model during initialization

    <Tip>

    Passing `use_auth_token=True` is required when you want to use a private
    model.

    </Tip>
    """
    return KerasModelHubMixin.from_pretrained(*args, **kwargs)


def push_to_hub_keras(
    model,
    repo_path_or_name: Optional[str] = None,
    repo_url: Optional[str] = None,
    log_dir: Optional[str] = None,
    commit_message: Optional[str] = "Add model",
    organization: Optional[str] = None,
    private: Optional[bool] = None,
    api_endpoint: Optional[str] = None,
    use_auth_token: Optional[Union[bool, str]] = True,
    git_user: Optional[str] = None,
    git_email: Optional[str] = None,
    config: Optional[dict] = None,
    include_optimizer: Optional[bool] = False,
    tags: Optional[Union[list, str]] = None,
    plot_model: Optional[bool] = True,
    **model_save_kwargs,
):
    """
    Upload model checkpoint or tokenizer files to the Hub while synchronizing a
    local clone of the repo in `repo_path_or_name`.

    Parameters:
        model (`Keras.Model`):
            The [Keras
            model](`https://www.tensorflow.org/api_docs/python/tf/keras/Model`)
            you'd like to push to the Hub. The model must be compiled and built.
        repo_path_or_name (`str`, *optional*):
            Can either be a repository name for your model or tokenizer in the
            Hub or a path to a local folder (in which case the repository will
            have the name of that local folder). If not specified, will default
            to the name given by `repo_url` and a local directory with that name
            will be created.
        repo_url (`str`, *optional*):
            Specify this in case you want to push to an existing repository in
            the Hub. If unspecified, a new repository will be created in your
            namespace (unless you specify an `organization`) with `repo_name`.
        log_dir (`str`, *optional*):
            TensorBoard logging directory to be pushed. The Hub automatically
            hosts and displays a TensorBoard instance if log files are included
            in the repository.
        commit_message (`str`, *optional*, defaults to "Add message"):
            Message to commit while pushing.
        organization (`str`, *optional*):
            Organization in which you want to push your model or tokenizer (you
            must be a member of this organization).
        private (`bool`, *optional*):
            Whether the repository created should be private.
        api_endpoint (`str`, *optional*):
            The API endpoint to use when pushing the model to the hub.
        use_auth_token (`bool` or `str`, *optional*, defaults to `True`):
            The token to use as HTTP bearer authorization for remote files. If
            `True`, will use the token generated when running `transformers-cli
            login` (stored in `~/.huggingface`). Will default to `True`.
        git_user (`str`, *optional*):
            will override the `git config user.name` for committing and pushing
            files to the Hub.
        git_email (`str`, *optional*):
            will override the `git config user.email` for committing and pushing
            files to the Hub.
        config (`dict`, *optional*):
            Configuration object to be saved alongside the model weights.
        include_optimizer (`bool`, *optional*, defaults to `False`):
            Whether or not to include optimizer during serialization.
        tags (Union[`list`, `str`], *optional*):
            List of tags that are related to model or string of a single tag. See example tags
            [here](https://github.com/huggingface/hub-docs/blame/main/modelcard.md).
        plot_model (`bool`, *optional*, defaults to `True`):
            Setting this to `True` will plot the model and put it in the model
            card. Requires graphviz and pydot to be installed.
        model_save_kwargs(`dict`, *optional*):
            model_save_kwargs will be passed to
            [`tf.keras.models.save_model()`](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model).

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
            "You must login to the Hugging Face hub on this computer by typing"
            " `huggingface-cli login` and entering your credentials to use"
            " `use_auth_token=True`. Alternatively, you can pass your own token as the"
            " `use_auth_token` argument."
        )

    if repo_path_or_name is None:
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
        use_auth_token=use_auth_token,
        git_user=git_user,
        git_email=git_email,
    )
    repo.git_pull(rebase=True)

    save_pretrained_keras(
        model,
        repo_path_or_name,
        config=config,
        include_optimizer=include_optimizer,
        tags=tags,
        plot_model=plot_model,
        **model_save_kwargs,
    )

    if log_dir is not None:
        if os.path.exists(f"{repo_path_or_name}/logs"):
            rmtree(f"{repo_path_or_name}/logs")
        copytree(log_dir, f"{repo_path_or_name}/logs")

    # Commit and push!
    repo.git_add(auto_lfs_track=True)
    repo.git_commit(commit_message)
    return repo.git_push()


class KerasModelHubMixin(ModelHubMixin):
    """
    Mixin to provide model Hub upload/download capabilities to Keras models.
    Override this class to obtain the following internal methods:
    - `_from_pretrained`, to load a model from the Hub or from local files.
    - `_save_pretrained`, to save a model in the `SavedModel` format.
    """

    def __init__(self, *args, **kwargs):
        """
        Mix this class with your keras-model class for ease process of saving &
        loading from huggingface-hub.


        ```python
        >>> from huggingface_hub import KerasModelHubMixin


        >>> class MyModel(tf.keras.Model, KerasModelHubMixin):
        ...     def __init__(self, **kwargs):
        ...         super().__init__()
        ...         self.config = kwargs.pop("config", None)
        ...         self.dummy_inputs = ...
        ...         self.layer = ...

        ...     def call(self, *args):
        ...         return ...


        >>> # Init and compile the model as you normally would
        >>> model = MyModel()
        >>> model.compile(...)
        >>> # Build the graph by training it or passing dummy inputs
        >>> _ = model(model.dummy_inputs)
        >>> # You can save your model like this
        >>> model.save_pretrained("local_model_dir/", push_to_hub=False)
        >>> # Or, you can push to a new public model repo like this
        >>> model.push_to_hub(
        ...     "super-cool-model",
        ...     git_user="your-hf-username",
        ...     git_email="you@somesite.com",
        ... )

        >>> # Downloading weights from hf-hub & model will be initialized from those weights
        >>> model = MyModel.from_pretrained("username/mymodel@main")
        ```
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
        """Here we just call from_pretrained_keras function so both the mixin and
        functional APIs stay in sync.

                TODO - Some args above aren't used since we are calling
                snapshot_download instead of hf_hub_download.
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
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                library_name="keras",
                library_version=get_tf_version(),
            )
        else:
            storage_folder = model_id

        model = tf.keras.models.load_model(storage_folder, **model_kwargs)

        # For now, we add a new attribute, config, to store the config loaded from the hub/a local dir.
        model.config = cfg

        return model
