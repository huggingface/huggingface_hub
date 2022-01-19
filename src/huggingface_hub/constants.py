import os


# Possible values for env variables

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

# Constants for file downloads

PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
CONFIG_NAME = "config.json"
REPOCARD_NAME = "README.md"

DEFAULT_REVISION = "main"

HUGGINGFACE_CO_URL_HOME = "https://huggingface.co/"

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
_staging_mode = (
    os.environ.get("HUGGINGFACE_CO_STAGING", "NO").upper() in ENV_VARS_TRUE_VALUES
)

ENDPOINT = (
    "https://moon-staging.huggingface.co" if _staging_mode else "https://huggingface.co"
)


HUGGINGFACE_CO_URL_TEMPLATE = ENDPOINT + "/{repo_id}/resolve/{revision}/{filename}"

REPO_TYPE_DATASET = "dataset"
REPO_TYPE_SPACE = "space"
REPO_TYPES = [None, REPO_TYPE_DATASET, REPO_TYPE_SPACE]
SPACES_SDK_TYPES = ["gradio", "streamlit", "static"]

REPO_TYPES_URL_PREFIXES = {
    REPO_TYPE_DATASET: "datasets/",
    REPO_TYPE_SPACE: "spaces/",
}
REPO_TYPES_MAPPING = {"datasets": REPO_TYPE_DATASET, "spaces": REPO_TYPE_SPACE}


# default cache
hf_cache_home = os.path.expanduser(
    os.getenv(
        "HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface")
    )
)
default_cache_path = os.path.join(hf_cache_home, "hub")

HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", default_cache_path)

HF_HUB_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "AUTO").upper()
if HF_HUB_OFFLINE in ENV_VARS_TRUE_VALUES:
    HF_HUB_OFFLINE = True
else:
    HF_HUB_OFFLINE = False

# Only valid pipeline_tags for hosted models and datasets. Taken from interfaces/Types.ts
VALID_PIPELINE_TAGS = [
    "text-classification",
    "token-classification",
    "table-question-answering",
    "question-answering",
    "zero-shot-classification",
    "translation",
    "summarization",
    "conversational",
    "feature-extraction",
    "text-generation",
    "text2text-generation",
    "fill-mask",
    "sentence-similarity",
    "text-to-speech",
    "automatic-speech-recognition",
    "audio-to-audio",
    "audio-classification",
    "voice-activity-detection",
    "image-classification",
    "object-detection",
    "image-segmentation",
    "text-to-image",
    "image-to-text",
    "structured-data-classification",
    "named-entity-recognition",
    "part-of-speech-tagging",
    "audio-source-separation",
    "speech-enhancement",
]
