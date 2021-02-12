import os


# Constants for file downloads

PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
CONFIG_NAME = "config.json"

HUGGINGFACE_CO_URL_HOME = "https://huggingface.co/"

HUGGINGFACE_CO_URL_TEMPLATE = (
    "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
)

REPO_TYPE_DATASET = "dataset"
REPO_TYPES = [None, REPO_TYPE_DATASET]

REPO_TYPE_DATASET_URL_PREFIX = "datasets/"


# default cache
hf_cache_home = os.path.expanduser(
    os.getenv(
        "HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface")
    )
)
default_cache_path = os.path.join(hf_cache_home, "hub")

HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", default_cache_path)
