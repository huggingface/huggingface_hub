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
import os
import unittest
from functools import partial

from huggingface_hub.hf_api import (
    HfApi,
)
from huggingface_hub.utils import (
    logging,
)

from .testing_constants import (
    ENDPOINT_STAGING,
    TOKEN,
)
from .testing_utils import (
    repo_name,
)


logger = logging.get_logger(__name__)

dataset_repo_name = partial(repo_name, prefix="my-dataset")
space_repo_name = partial(repo_name, prefix="my-space")
large_file_repo_name = partial(repo_name, prefix="my-model-largefiles")

WORKING_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/working_repo")
LARGE_FILE_14MB = "https://cdn-media.huggingface.co/lfs-largefiles/progit.epub"
LARGE_FILE_18MB = "https://cdn-media.huggingface.co/lfs-largefiles/progit.pdf"


class HfApiCommonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Share the valid token in all tests below."""
        cls._token = TOKEN
        cls._api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
