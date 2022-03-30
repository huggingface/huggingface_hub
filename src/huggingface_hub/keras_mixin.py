import json
import logging
import os
from pathlib import Path
from shutil import copytree, rmtree
from typing import Any, Dict, Optional, Union

from huggingface_hub import ModelHubMixin
from huggingface_hub.file_download import (
    is_graphviz_available,
    is_pydot_available,
    is_tf_available,
)
from huggingface_hub.snapshot_download import snapshot_download

from .constants import CONFIG_NAME
from .hf_api import HfApi, HfFolder
from .repository import Repository


logger = logging.getLogger(__name__)

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


def _parse_model_history(model):
    lines = None
    if model.history is not None:
        if model.history.history != {}:
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
        show_layer_activations=True,
    )


def _write_metrics(model, model_card):
    lines = _parse_model_history(model)
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
    task_name: Optional[str] = None,
):
    """
    Creates a model card for the repository.
    """
    readme_path = f"{repo_dir}/README.md"
    model_card = "---\n"
    if task_name is not None:
        model_card += f"tags:\n- {task_name}\n"
    model_card += "library_name: keras\n---\n"
    model_card += "\n## Model description\n\nMore information needed\n"
    model_card += "\n## Intended uses & limitations\n\nMore information needed\n"
    model_card += "\n## Training and evaluation data\n\nMore information needed\n"
    if isinstance(model, dict):
        for model_name, tf_model in model.items():
            hyperparameters = _extract_hyperparameters_from_keras(tf_model)
            if plot_model and is_graphviz_available() and is_pydot_available():
                _plot_network(tf_model, f"{repo_dir}/{model_name}")
            if hyperparameters is not None:
                model_card += f"\n### Training hyperparameters for {model_name}\n"
                model_card += (
                    "\nThe following hyperparameters were used during training:\n"
                )
                model_card += "\n".join(
                    [f"- {name}: {value}" for name, value in hyperparameters.items()]
                )
            model_card += "\n"
            model_card += f"\n ## Training Metrics for {model_name}\n"
            model_card = _write_metrics(model[model_name], model_card)
            if plot_model and os.path.exists(f"{repo_dir}/{model_name}/model.png"):
                model_card += f"\n ## Model Plot for {model_name}\n"
                model_card += "\n<details>"
                model_card += "\n<summary>View Model Plot</summary>\n"
                model_card += f"\n![Model Image](./{model_name}/model.png)\n"
                model_card += "\n</details>\n\n"

    else:
        hyperparameters = _extract_hyperparameters_from_keras(model)
        if plot_model and is_graphviz_available() and is_pydot_available():
            _plot_network(model, repo_dir)
        if hyperparameters is not None:
            model_card += "\n### Training hyperparameters\n"
            model_card += "\nThe following hyperparameters were used during training:\n"
            model_card += "\n".join(
                [f"- {name}: {value}" for name, value in hyperparameters.items()]
            )
            model_card += "\n"
        model_card += "\n ## Training Metrics\n"
        model_card = _write_metrics(model, model_card)
        if plot_model and os.path.exists(f"{repo_dir}/model.png"):
            model_card += "\n ## Model Plot\n"
            model_card += "\n<details>"
            model_card += "\n<summary>View Model Plot</summary>\n"
            path_to_plot = "./model.png"
            model_card += f"\n![Model Image]({path_to_plot})\n"
            model_card += "\n</details>\n"

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
    task_name: Optional[str] = None,
    multiple_model_save_args: Optional[Dict] = None,
    **model_save_kwargs,
):
    """Saves a Keras model to save_directory in SavedModel format. Use this if you're using the Functional or Sequential APIs.

    model:
        The Keras model or dictionary of multiple models you'd like to push to the hub. The model(s) must be compiled and built. If multiple models are pushed, each key in dictionary should be an identifier of model and value should be model.
    save_directory (:obj:`str`):
        Specify directory in which you want to save the Keras model.
    config (:obj:`dict`, `optional`):
        Configuration object to be saved alongside the model weights. If multiple models are pushed, should be a dictionary of dictionaries with each key being model identifier and values being that model's config.
    include_optimizer(:obj:`bool`, `optional`):
        Whether or not to include optimizer in serialization.
    task_name (:obj:`str`, `optional`):
        Name of the task the model was trained on. See the available tasks at https://github.com/huggingface/huggingface_hub/blob/main/js/src/lib/interfaces/Types.ts.
    plot_model (:obj:`bool`):
        Setting this to `True` will plot the model and put it in the model card. Requires graphviz and pydot to be installed.
    multiple_model_save_args (:obj:`dict`, `optional`):
        Model save parameters as dictionary, where keys are model identifiers and values are dictionary of parameters.
    model_save_kwargs(:obj:`dict`, `optional`):
        model_save_kwargs will be passed to tf.keras.models.save_model() when saving only one model.
    """
    if is_tf_available():
        import tensorflow as tf
    else:
        raise ImportError(
            "Called a Tensorflow-specific function but could not import it."
        )
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

    # if multiple models are passed by a dictionary
    if isinstance(model, dict):
        for model_name in model.keys():

            if not model[model_name].built:
                raise ValueError("Model should be built before trying to save")

            os.makedirs(f"{save_directory}/{model_name}", exist_ok=True)
            if multiple_model_save_args is not None:
                if model_name in multiple_model_save_args:
                    model_args = multiple_model_save_args[model_name]
                tf.keras.models.save_model(
                    model[model_name],
                    f"{save_directory}/{model_name}",
                    include_optimizer=include_optimizer,
                    **model_args,
                )
            else:
                tf.keras.models.save_model(
                    model[model_name],
                    f"{save_directory}/{model_name}",
                    include_optimizer=include_optimizer,
                )
    else:
        if not model.built:
            raise ValueError("Model should be built before trying to save")

        os.makedirs(save_directory, exist_ok=True)
        tf.keras.models.save_model(
            model,
            save_directory,
            include_optimizer=include_optimizer,
            **model_save_kwargs,
        )
    _create_model_card(model, save_directory, plot_model, task_name)


def from_pretrained_keras(*args, **kwargs):
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
    task_name: Optional[str] = None,
    plot_model: Optional[bool] = True,
    multiple_model_save_args: Optional[Dict] = None,
    **model_save_kwargs,
):
    """
    Upload model checkpoint or tokenizer files to the 🤗 Model Hub while synchronizing a local clone of the repo in
    :obj:`repo_path_or_name`.

    Parameters:
        model:
            The Keras model or dictionary of multiple models you'd like to push to the hub. The model(s) must be compiled and built. If multiple models are pushed, each key in dictionary should be an identifier of model and value should be model.
        repo_path_or_name (:obj:`str`, `optional`):
            Can either be a repository name for your model or tokenizer in the Hub or a path to a local folder (in
            which case the repository will have the name of that local folder). If not specified, will default to
            the name given by :obj:`repo_url` and a local directory with that name will be created.
        repo_url (:obj:`str`, `optional`):
            Specify this in case you want to push to an existing repository in the hub. If unspecified, a new
            repository will be created in your namespace (unless you specify an :obj:`organization`) with
            :obj:`repo_name`.
        log_dir (:obj:`str`, `optional`):
            TensorBoard logging directory to be pushed. The Hub automatically hosts
            and displays a TensorBoard instance if log files are included in the repository.
        commit_message (:obj:`str`, `optional`):
            Message to commit while pushing. Will default to :obj:`"add model"`.
        organization (:obj:`str`, `optional`):
            Organization in which you want to push your model or tokenizer (you must be a member of this
            organization).
        private (:obj:`bool`, `optional`):
            Whether or not the repository created should be private.
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
            Configuration object to be saved alongside the model weights. If multiple models are pushed, should be a dictionary of dictionaries with each key being model identifier and values being that model's config.
        include_optimizer (:obj:`bool`, `optional`):
            Whether or not to include optimizer during serialization.
        task_name (:obj:`str`, `optional`):
            Name of the task the model was trained on. See the available tasks at https://github.com/huggingface/huggingface_hub/blob/main/js/src/lib/interfaces/Types.ts.
        plot_model (:obj:`bool`):
            Setting this to `True` will plot the model and put it in the model card. Requires graphviz and pydot to be installed.
        multiple_model_save_args (:obj:`dict`, `optional`):
            Model save parameters as dictionary, where keys are model identifiers and values are dictionary of parameters.
        model_save_kwargs (:obj:`dict`, `optional`):
            model_save_kwargs will be passed to tf.keras.models.save_model() when saving only one model.

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

    if isinstance(model, dict):
        save_pretrained_keras(
            model,
            repo_path_or_name,
            config=config,
            include_optimizer=include_optimizer,
            plot_model=plot_model,
            task_name=task_name,
            multiple_model_save_args=multiple_model_save_args,
        )
    else:
        save_pretrained_keras(
            model,
            repo_path_or_name,
            config=config,
            include_optimizer=include_optimizer,
            plot_model=plot_model,
            task_name=task_name,
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

        # Root is either a local filepath matching model_id or a cached snapshot
        if not os.path.isdir(model_id):
            storage_folder = snapshot_download(
                repo_id=model_id, revision=revision, cache_dir=cache_dir
            )
        else:
            storage_folder = model_id

        # there can be multiple models or single model
        list_dir = os.listdir(storage_folder)
        # if there's only one model
        if "saved_model.pb" in list_dir:
            # TODO - Figure out what to do about these config values. Config is not going to be needed to load model
            cfg = model_kwargs.pop("config", None)
            model = tf.keras.models.load_model(storage_folder, **model_kwargs)
            model.config = cfg
            return model
        else:
            models_dict = {}

            for folder in list_dir:
                if os.path.isdir(
                    f"{storage_folder}/{folder}"
                ) and not folder.startswith("."):
                    model_cfg = model_kwargs.pop("config", None)
                    if model_cfg:
                        model_cfg = model_cfg.pop(folder, None)
                    model = tf.keras.models.load_model(
                        f"{storage_folder}/{folder}", **model_kwargs
                    )
                    model.config = model_cfg
                    models_dict[folder] = model

            return models_dict
