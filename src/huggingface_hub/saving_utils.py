import os
import json
from .file_download import (
    hf_hub_url, 
    cached_download,
    CONFIG_NAME,
    PYTORCH_WEIGHTS_NAME
    )

import torch


class SavingUtils(object):

    def __init__(self, **kwargs):
        """

        """

    def save_pretrained(self, save_directory:str):
        """
            Saving weights in local directory that can be loaded directly from huggingface hub
        """

        if save_directory not in os.listdir(): 
            os.makedirs(save_directory)

        # saving config
        if hasattr(self, 'config'):
            path = os.path.join(save_directory, CONFIG_NAME)
            with open(path, "w") as f:
                json.dump(self.config, f)

        # saving only the adapter weights and length embedding
        path = os.path.join(save_directory, PYTORCH_WEIGHTS_NAME)
        self._save_pretrained(path, verbose=False)

        return True

    def _save_pretrained(self, path, verbose):
        """
            Overwrite this method in case you don't want to save complete model, rather some specific layers
        """

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        torch.save(model_to_save.state_dict(), path)
        if verbose:
            print(f"saving model weights")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path:str, **kwargs):
        """
            Setting up this method will enable to load directly from huggingface hub just like other HF models are loaded
        """
        model_id = pretrained_model_name_or_path
        strict = kwargs.pop("strict", True)
        map_location = kwargs.pop("map_location", torch.device("cpu"))

        if len(model_id.split("/")) == 1:
            name = model_id
        else:
            _, name = model_id.split("/")

        if name in os.listdir() and CONFIG_NAME in os.listdir(name):
            print("LOADING weights from local directory")
            config_file = os.path.join(name, CONFIG_NAME)
        else:
            try:
                config_url = hf_hub_url(model_id, filename=CONFIG_NAME)
                config_file = cached_download(config_url)
            except:
                config_file = None

        if name in os.listdir():
            print("LOADING weights from local directory")
            model_file = os.path.join(name, PYTORCH_WEIGHTS_NAME)
        else:
            model_url = hf_hub_url(model_id, filename=PYTORCH_WEIGHTS_NAME)
            model_file = cached_download(model_url)

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
