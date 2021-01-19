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

    def __init__(self, **kwargs):
        """

        """

    def save_pretrained(self, save_directory:str):
        """
            Saving weights in local directory that can be loaded directly from huggingface-hub
        """
        self.wts_directory = save_directory

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

    def upload_to_hub(self, model_id:str, wts_directory:str=None, branch:str="main", commit_message:str="add model"):
        """
            This method will upload your model weights to huggingface-hub

            ARGUMENTS:
                model_id :       model_id which will be used later using from_pretrained (Generally: username/model_name)   
                wts_directory :  directory made using save_pretrained

            NOTE:
                - You need to create repository in huggingface hub manually
                - This may take some time depending on your directory size
        """

        if wts_directory is None:
            wts_directory = self.wts_directory if hasattr(self, wts_directory) else wts_directory

        if not os.path.isdir(wts_directory):
            raise FileExistsError(f"{wts_directory} doesn't exist")

        os.chdir(wts_directory)
        if not os.path.isdir(".git"):
            print(1)
            subprocess.run(["git", "init"], stdout=subprocess.PIPE)

        if not os.path.isdir(".git/lfs"):
            print(2)
            subprocess.run(["git-lfs", "install"], stdout=subprocess.PIPE)

        with open(".git/config", "r") as f:
            git_config = f.read().split()
        if "url" not in git_config:
            print(3)
            subprocess.run(["git", "remote", "add", "origin", PREFIX+model_id], stdout=subprocess.PIPE)
            subprocess.run(["git", "pull", "origin", branch, "--rebase"])

        if not branch in os.listdir(".git/refs/heads"):
            print(4)
            subprocess.run(["git", "checkout", "-b", branch], stdout=subprocess.PIPE)

        print(5)
        subprocess.run(["git", "add", "--all"], stdout=subprocess.PIPE)
        print(6)
        subprocess.run(["git", "commit", "-m", commit_message], stdout=subprocess.PIPE)

        print(7)
        subprocess.run(["git", "push", "origin", branch])
        print(8)


if __name__ == "__main__":

    mix = ModelHubMixin()
    mix.upload_to_hub("vasudevgupta/test", "test")
