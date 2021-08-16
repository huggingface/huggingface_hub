import logging
import os
from pathlib import Path

from huggingface_hub import ModelHubMixin, hf_hub_download


logger = logging.getLogger(__name__)


class KerasModelHubMixin(ModelHubMixin):

    _CONFIG_NAME = "config.json"
    _WEIGHTS_NAME = "tf_model.h5"

    def __init__(self, *args, **kwargs):
        """
        Mix this class with your keras-model class for ease process of saving & loading from huggingface-hub

        NOTE - Dummy Inputs are required to save/load models using this mixin. When saving, you are required to either:

            1. Assign an attribute to your class, self.dummy_inputs, that defines inputs to be passed to the model's call
            function to build the model.
            2. Pass the dummy_inputs kwarg to save_pretrained. We will save this along with the model (as if it were an attribute).

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

            >>> model = MyModel()
            >>> model.save_pretrained("mymodel", push_to_hub=False) # Saving model weights in the directory
            >>> model.push_to_hub("mymodel", "model-1") # Pushing model-weights to hf-hub

            >>> # Downloading weights from hf-hub & model will be initialized from those weights
            >>> model = MyModel.from_pretrained("username/mymodel@main")
        """

    def _save_pretrained(self, save_directory, dummy_inputs=None, **kwargs):

        dummy_inputs = (
            dummy_inputs
            if dummy_inputs is not None
            else getattr(self, "dummy_inputs", None)
        )

        if dummy_inputs is None:
            raise RuntimeError(
                "You must either provide dummy inputs or have them assigned as an attribute of this model"
            )

        _ = self(dummy_inputs, training=False)

        save_directory = Path(save_directory)
        model_file = save_directory / self._WEIGHTS_NAME
        self.save_weights(model_file)
        logger.info(f"Model weights saved in {model_file}")

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
        by_name=False,
        **model_kwargs,
    ):
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, cls._WEIGHTS_NAME)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=cls._WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only,
            )

        model = cls(**model_kwargs)

        if hasattr(model, "dummy_inputs") and model.dummy_inputs is not None:
            raise ValueError("Model must have a dummy_inputs attribute")

        _ = model(model.dummy_inputs, training=False)

        model.load_weights(model_file, by_name=by_name)

        _ = model(model.dummy_inputs, training=False)

        return model
