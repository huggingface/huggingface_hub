import logging
import os
from pathlib import Path

from huggingface_hub.file_download import is_tf_available
from huggingface_hub.snapshot_download import snapshot_download
from huggingface_hub import ModelHubMixin, hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE


logger = logging.getLogger(__name__)

if is_tf_available():
    import tensorflow as tf


def save_pretrained_keras(model, save_dir):
    """Function mimicing save_pretrained. Use this if you're using the Functional or Sequential APIs."""
    if not model.built:
        raise ValueError("Model should be built before trying to save")

    tf.keras.models.save_model(model, save_dir)

def from_pretrained_keras(model_name_or_path, revision=None, cache_dir=None, **kwargs):
    """Function mimicing from_pretrained. Use this if you're using the Functional or Sequential APIs."""    

    # Root is either a local filepath matching model_id or a cached snapshot
    storage_folder = snapshot_download(
        repo_id=model_name_or_path,
        revision=revision,
        cache_dir=cache_dir
    ) if not os.path.isdir(model_name_or_path) else model_name_or_path

    return tf.keras.models.load_model(storage_folder, **kwargs)

def push_to_hub_keras(model, *args, **kwargs):
    """Here we simply call the mixin's push_to_hub function. Would it be better to do it the other way around?"""
    return ModelHubMixin.push_to_hub(model, *args, **kwargs)


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
            >>> model(model.dummy_inputs)
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
        return from_pretrained_keras(model_id, revision, cache_dir, **model_kwargs)
