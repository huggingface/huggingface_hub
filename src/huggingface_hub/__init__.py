# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ***********
# vendored from https://github.com/scientific-python/lazy_loader
import importlib
import importlib.util
import inspect
import os
import sys
import types
import warnings


class _LazyImportWarning(Warning):
    pass


def _attach(package_name, submodules=None, submod_attrs=None):
    """Attach lazily loaded submodules, functions, or other attributes.

    Typically, modules import submodules and attributes as follows:

    ```py
    import mysubmodule
    import anothersubmodule

    from .foo import someattr
    ```

    The idea is to replace a package's `__getattr__`, `__dir__`, and
    `__all__`, such that all imports work exactly the way they would
    with normal imports, except that the import occurs upon first use.

    The typical way to call this function, replacing the above imports, is:

    ```python
    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        ['mysubmodule', 'anothersubmodule'],
        {'foo': ['someattr']}
    )
    ```
    This functionality requires Python 3.7 or higher.

    Args:
        package_name (`str`):
            Typically use `__name__`.
        submodules (`set`):
            List of submodules to attach.
        submod_attrs (`dict`):
            Dictionary of submodule -> list of attributes / functions.
            These attributes are imported as they are used.

    Returns:
        __getattr__, __dir__, __all__

    """
    if submod_attrs is None:
        submod_attrs = {}

    if submodules is None:
        submodules = set()
    else:
        submodules = set(submodules)

    attr_to_modules = {
        attr: mod for mod, attrs in submod_attrs.items() for attr in attrs
    }

    __all__ = list(submodules | attr_to_modules.keys())

    def __getattr__(name):
        if name in submodules:
            return importlib.import_module(f"{package_name}.{name}")
        elif name in attr_to_modules:
            submod_path = f"{package_name}.{attr_to_modules[name]}"
            submod = importlib.import_module(submod_path)
            attr = getattr(submod, name)

            # If the attribute lives in a file (module) with the same
            # name as the attribute, ensure that the attribute and *not*
            # the module is accessible on the package.
            if name == attr_to_modules[name]:
                pkg = sys.modules[package_name]
                pkg.__dict__[name] = attr

            return attr
        else:
            raise AttributeError(f"No {package_name} attribute {name}")

    def __dir__():
        return __all__

    if os.environ.get("EAGER_IMPORT", ""):
        for attr in set(attr_to_modules.keys()) | submodules:
            __getattr__(attr)

    return __getattr__, __dir__, list(__all__)


# ************

__version__ = "0.9.0"


__getattr__, __dir__, __all__ = _attach(
    __name__,
    submodules=[],
    submod_attrs={
        "commands.user": ["notebook_login"],
        "constants": [
            "CONFIG_NAME",
            "FLAX_WEIGHTS_NAME",
            "HUGGINGFACE_CO_URL_HOME",
            "HUGGINGFACE_CO_URL_TEMPLATE",
            "PYTORCH_WEIGHTS_NAME",
            "REPO_TYPE_DATASET",
            "REPO_TYPE_MODEL",
            "REPO_TYPE_SPACE",
            "TF2_WEIGHTS_NAME",
            "TF_WEIGHTS_NAME",
        ],
        "fastai_utils": [
            "_save_pretrained_fastai",
            "from_pretrained_fastai",
            "push_to_hub_fastai",
        ],
        "file_download": [
            "cached_download",
            "hf_hub_download",
            "hf_hub_url",
            "try_to_load_from_cache",
        ],
        "hf_api": [
            "CommitOperation",
            "CommitOperationAdd",
            "CommitOperationDelete",
            "DatasetSearchArguments",
            "HfApi",
            "HfFolder",
            "ModelSearchArguments",
            "change_discussion_status",
            "comment_discussion",
            "create_commit",
            "create_discussion",
            "create_pull_request",
            "create_repo",
            "dataset_info",
            "delete_file",
            "delete_repo",
            "edit_discussion_comment",
            "get_dataset_tags",
            "get_discussion_details",
            "get_full_repo_name",
            "get_model_tags",
            "get_repo_discussions",
            "list_datasets",
            "list_metrics",
            "list_models",
            "list_repo_files",
            "login",
            "logout",
            "merge_pull_request",
            "model_info",
            "move_repo",
            "rename_discussion",
            "repo_type_and_id_from_hf_id",
            "set_access_token",
            "space_info",
            "unset_access_token",
            "update_repo_visibility",
            "upload_file",
            "upload_folder",
            "whoami",
        ],
        "hub_mixin": ["ModelHubMixin", "PyTorchModelHubMixin"],
        "inference_api": ["InferenceApi"],
        "keras_mixin": [
            "KerasModelHubMixin",
            "from_pretrained_keras",
            "push_to_hub_keras",
            "save_pretrained_keras",
        ],
        "repository": ["Repository"],
        "_snapshot_download": ["snapshot_download"],
        "utils": ["logging"],
        "utils.endpoint_helpers": ["DatasetFilter", "ModelFilter"],
        "repocard": [
            "metadata_eval_result",
            "metadata_load",
            "metadata_save",
            "metadata_update",
        ],
        "community": [
            "Discussion",
            "DiscussionWithDetails",
            "DiscussionEvent",
            "DiscussionComment",
            "DiscussionStatusChange",
            "DiscussionCommit",
            "DiscussionTitleChange",
        ],
    },
)
