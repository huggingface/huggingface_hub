# TODO: remove in 0.11

import warnings


warnings.warn(
    "snapshot_download.py has been made private and will no longer be available from"
    " version 0.11. Please use `from huggingface_hub import snapshot_download` to"
    " import the only public function in this module. Other members of the file may be"
    " changed without a deprecation notice.",
    FutureWarning,
)

from ._snapshot_download import *  # noqa
from .constants import REPO_ID_SEPARATOR  # noqa
