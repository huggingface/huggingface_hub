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

__version__ = "0.4.0.dev0"

from .commands.user import notebook_login
from .constants import (
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    HUGGINGFACE_CO_URL_HOME,
    HUGGINGFACE_CO_URL_TEMPLATE,
    PYTORCH_WEIGHTS_NAME,
    REPO_TYPE_DATASET,
    REPO_TYPE_SPACE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
)
from .file_download import cached_download, hf_hub_download, hf_hub_url
from .hf_api import (
    DatasetSearchArguments,
    HfApi,
    HfFolder,
    ModelSearchArguments,
    create_repo,
    dataset_info,
    delete_file,
    delete_repo,
    get_dataset_tags,
    get_full_repo_name,
    get_model_tags,
    list_datasets,
    list_metrics,
    list_models,
    list_repo_files,
    list_repos_objs,
    login,
    logout,
    model_info,
    repo_type_and_id_from_hf_id,
    update_repo_visibility,
    upload_file,
    whoami,
)
from .hub_mixin import ModelHubMixin, PyTorchModelHubMixin
from .inference_api import InferenceApi
from .keras_mixin import (
    KerasModelHubMixin,
    from_pretrained_keras,
    push_to_hub_keras,
    save_pretrained_keras,
)
from .repository import Repository
from .snapshot_download import snapshot_download
from .utils import logging
from .utils.endpoint_helpers import DatasetFilter, ModelFilter
