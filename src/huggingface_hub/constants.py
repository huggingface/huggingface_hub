import os
import re
from typing import Optional


# Possible values for env variables

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})


def _is_true(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.upper() in ENV_VARS_TRUE_VALUES


def _is_true_or_auto(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.upper() in ENV_VARS_TRUE_AND_AUTO_VALUES


# Constants for file downloads

PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
CONFIG_NAME = "config.json"
REPOCARD_NAME = "README.md"

# Git-related constants

DEFAULT_REVISION = "main"
REGEX_COMMIT_OID = re.compile(r"[A-Fa-f0-9]{5,40}")

HUGGINGFACE_CO_URL_HOME = "https://huggingface.co/"

_staging_mode = _is_true(os.environ.get("HUGGINGFACE_CO_STAGING"))

ENDPOINT = os.getenv("HF_ENDPOINT") or (
    "https://hub-ci.huggingface.co" if _staging_mode else "https://huggingface.co"
)

HUGGINGFACE_CO_URL_TEMPLATE = ENDPOINT + "/{repo_id}/resolve/{revision}/{filename}"
HUGGINGFACE_HEADER_X_REPO_COMMIT = "X-Repo-Commit"
HUGGINGFACE_HEADER_X_LINKED_ETAG = "X-Linked-Etag"

REPO_ID_SEPARATOR = "--"
# ^ this substring is not allowed in repo_ids on hf.co
# and is the canonical one we use for serialization of repo ids elsewhere.


REPO_TYPE_DATASET = "dataset"
REPO_TYPE_SPACE = "space"
REPO_TYPE_MODEL = "model"
REPO_TYPES = [None, REPO_TYPE_MODEL, REPO_TYPE_DATASET, REPO_TYPE_SPACE]
SPACES_SDK_TYPES = ["gradio", "streamlit", "static"]

REPO_TYPES_URL_PREFIXES = {
    REPO_TYPE_DATASET: "datasets/",
    REPO_TYPE_SPACE: "spaces/",
}
REPO_TYPES_MAPPING = {
    "datasets": REPO_TYPE_DATASET,
    "spaces": REPO_TYPE_SPACE,
    "models": REPO_TYPE_MODEL,
}


# default cache
hf_cache_home = os.path.expanduser(
    os.getenv(
        "HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface")
    )
)
default_cache_path = os.path.join(hf_cache_home, "hub")

HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", default_cache_path)

HF_HUB_OFFLINE = _is_true(os.environ.get("HF_HUB_OFFLINE"))


# Here, `True` will disable progress bars globally without possibility of enabling it
# programatically. `False` will enable them without possibility of disabling them.
# If environement variable is not set (None), then the user is free to enable/disable
# them programmatically.
# TL;DR: env variable has priority over code
HF_HUB_DISABLE_PROGRESS_BARS: Optional[bool] = os.environ.get(
    "HF_HUB_DISABLE_PROGRESS_BARS"
)
if HF_HUB_DISABLE_PROGRESS_BARS is not None:
    HF_HUB_DISABLE_PROGRESS_BARS = _is_true(HF_HUB_DISABLE_PROGRESS_BARS)
