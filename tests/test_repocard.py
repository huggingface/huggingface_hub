# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import copy
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

import pytest

import yaml
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.hf_api import HfApi
from huggingface_hub.repocard import (
    metadata_eval_result,
    metadata_load,
    metadata_save,
    metadata_update,
)
from huggingface_hub.repository import Repository
from huggingface_hub.utils import logging

from .testing_constants import ENDPOINT_STAGING, TOKEN, USER
from .testing_utils import retry_endpoint, set_write_permission_and_retry


ROUND_TRIP_MODELCARD_CASE = """
---
language: no
datasets: CLUECorpusSmall
widget:
- text: 北京是[MASK]国的首都。
---

# Title
"""

DUMMY_MODELCARD = """

Hi

---
license: mit
datasets:
- foo
- bar
---

Hello
"""

DUMMY_MODELCARD_TARGET = """

Hi

---
meaning_of_life: 42
---

Hello
"""

DUMMY_MODELCARD_TARGET_NO_YAML = """---
meaning_of_life: 42
---
Hello
"""

DUMMY_NEW_MODELCARD_TARGET = """---
meaning_of_life: 42
---
"""

DUMMY_MODELCARD_TARGET_NO_TAGS = """
Hello
"""

DUMMY_MODELCARD_EVAL_RESULT = """---
model-index:
- name: RoBERTa fine-tuned on ReactionGIF
  results:
  - metrics:
    - type: accuracy
      value: 0.2662102282047272
      name: Accuracy
    task:
      type: text-classification
      name: Text Classification
    dataset:
      name: ReactionGIF
      type: julien-c/reactiongif
---
"""

logger = logging.get_logger(__name__)

REPOCARD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures/repocard"
)

REPO_NAME = f"dummy-hf-hub-{str(uuid.uuid4())}"


class RepocardTest(unittest.TestCase):
    def setUp(self):
        os.makedirs(REPOCARD_DIR, exist_ok=True)

    def tearDown(self) -> None:
        if os.path.exists(REPOCARD_DIR):
            shutil.rmtree(REPOCARD_DIR, onerror=set_write_permission_and_retry)
        logger.info(f"Does {REPOCARD_DIR} exist: {os.path.exists(REPOCARD_DIR)}")

    def test_metadata_load(self):
        filepath = Path(REPOCARD_DIR) / REPOCARD_NAME
        filepath.write_text(DUMMY_MODELCARD)
        data = metadata_load(filepath)
        self.assertDictEqual(data, {"license": "mit", "datasets": ["foo", "bar"]})

    def test_metadata_save(self):
        filename = "dummy_target.md"
        filepath = Path(REPOCARD_DIR) / filename
        filepath.write_text(DUMMY_MODELCARD)
        metadata_save(filepath, {"meaning_of_life": 42})
        content = filepath.read_text()
        self.assertEqual(content, DUMMY_MODELCARD_TARGET)

    def test_metadata_save_from_file_no_yaml(self):
        filename = "dummy_target_2.md"
        filepath = Path(REPOCARD_DIR) / filename
        filepath.write_text("Hello\n")
        metadata_save(filepath, {"meaning_of_life": 42})
        content = filepath.read_text()
        self.assertEqual(content, DUMMY_MODELCARD_TARGET_NO_YAML)

    def test_metadata_save_new_file(self):
        filename = "new_dummy_target.md"
        filepath = Path(REPOCARD_DIR) / filename
        metadata_save(filepath, {"meaning_of_life": 42})
        content = filepath.read_text()
        self.assertEqual(content, DUMMY_NEW_MODELCARD_TARGET)

    def test_no_metadata_returns_none(self):
        filename = "dummy_target_3.md"
        filepath = Path(REPOCARD_DIR) / filename
        filepath.write_text(DUMMY_MODELCARD_TARGET_NO_TAGS)
        data = metadata_load(filepath)
        self.assertEqual(data, None)

    def test_metadata_eval_result(self):
        data = metadata_eval_result(
            model_pretty_name="RoBERTa fine-tuned on ReactionGIF",
            task_pretty_name="Text Classification",
            task_id="text-classification",
            metrics_pretty_name="Accuracy",
            metrics_id="accuracy",
            metrics_value=0.2662102282047272,
            dataset_pretty_name="ReactionGIF",
            dataset_id="julien-c/reactiongif",
        )
        filename = "eval_results.md"
        filepath = Path(REPOCARD_DIR) / filename
        metadata_save(filepath, data)
        content = filepath.read_text().splitlines()
        self.assertEqual(content, DUMMY_MODELCARD_EVAL_RESULT.splitlines())


class RepocardUpdateTest(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING)

    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = TOKEN
        cls._api.set_access_token(TOKEN)

    @retry_endpoint
    def setUp(self) -> None:
        self.repo_path = Path(tempfile.mkdtemp())
        self.repo = Repository(
            self.repo_path / REPO_NAME,
            clone_from=f"{USER}/{REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with self.repo.commit("Add README to main branch"):
            with open("README.md", "w+") as f:
                f.write(DUMMY_MODELCARD_EVAL_RESULT)

        self.existing_metadata = yaml.safe_load(
            DUMMY_MODELCARD_EVAL_RESULT.strip().strip("---")
        )

    def tearDown(self) -> None:
        self._api.delete_repo(repo_id=f"{REPO_NAME}", token=self._token)
        shutil.rmtree(self.repo_path)

    def test_update_dataset_name(self):
        new_datasets_data = {"datasets": "['test/test_dataset']"}
        metadata_update(f"{USER}/{REPO_NAME}", new_datasets_data, token=self._token)

        self.repo.git_pull()
        updated_metadata = metadata_load(self.repo_path / REPO_NAME / "README.md")
        expected_metadata = copy.deepcopy(self.existing_metadata)
        expected_metadata.update(new_datasets_data)
        self.assertDictEqual(updated_metadata, expected_metadata)

    def test_update_existing_result_with_overwrite(self):
        new_metadata = copy.deepcopy(self.existing_metadata)
        new_metadata["model-index"][0]["results"][0]["metrics"][0][
            "value"
        ] = 0.2862102282047272
        metadata_update(
            f"{USER}/{REPO_NAME}", new_metadata, token=self._token, overwrite=True
        )

        self.repo.git_pull()
        updated_metadata = metadata_load(self.repo_path / REPO_NAME / "README.md")
        self.assertDictEqual(updated_metadata, new_metadata)

    def test_update_existing_result_without_overwrite(self):
        new_metadata = copy.deepcopy(self.existing_metadata)
        new_metadata["model-index"][0]["results"][0]["metrics"][0][
            "value"
        ] = 0.2862102282047272

        with pytest.raises(
            ValueError,
            match=(
                "You passed a new value for the existing metric 'name: Accuracy, type:"
                " accuracy'. Set `overwrite=True` to overwrite existing metrics."
            ),
        ):
            metadata_update(
                f"{USER}/{REPO_NAME}", new_metadata, token=self._token, overwrite=False
            )

    def test_update_existing_field_without_overwrite(self):
        new_datasets_data = {"datasets": "['test/test_dataset']"}
        metadata_update(f"{USER}/{REPO_NAME}", new_datasets_data, token=self._token)

        with pytest.raises(
            ValueError,
            match=(
                "You passed a new value for the existing meta data field 'datasets'."
                " Set `overwrite=True` to overwrite existing metadata."
            ),
        ):
            new_datasets_data = {"datasets": "['test/test_dataset_2']"}
            metadata_update(
                f"{USER}/{REPO_NAME}",
                new_datasets_data,
                token=self._token,
                overwrite=False,
            )

    def test_update_new_result_existing_dataset(self):
        new_result = metadata_eval_result(
            model_pretty_name="RoBERTa fine-tuned on ReactionGIF",
            task_pretty_name="Text Classification",
            task_id="text-classification",
            metrics_pretty_name="Recall",
            metrics_id="recall",
            metrics_value=0.7762102282047272,
            dataset_pretty_name="ReactionGIF",
            dataset_id="julien-c/reactiongif",
        )

        metadata_update(
            f"{USER}/{REPO_NAME}", new_result, token=self._token, overwrite=False
        )

        expected_metadata = copy.deepcopy(self.existing_metadata)
        expected_metadata["model-index"][0]["results"][0]["metrics"].append(
            new_result["model-index"][0]["results"][0]["metrics"][0]
        )

        self.repo.git_pull()
        updated_metadata = metadata_load(self.repo_path / REPO_NAME / "README.md")
        self.assertDictEqual(updated_metadata, expected_metadata)

    def test_update_new_result_new_dataset(self):
        new_result = metadata_eval_result(
            model_pretty_name="RoBERTa fine-tuned on ReactionGIF",
            task_pretty_name="Text Classification",
            task_id="text-classification",
            metrics_pretty_name="Accuracy",
            metrics_id="accuracy",
            metrics_value=0.2662102282047272,
            dataset_pretty_name="ReactionJPEG",
            dataset_id="julien-c/reactionjpeg",
        )

        metadata_update(
            f"{USER}/{REPO_NAME}", new_result, token=self._token, overwrite=False
        )

        expected_metadata = copy.deepcopy(self.existing_metadata)
        expected_metadata["model-index"][0]["results"].append(
            new_result["model-index"][0]["results"][0]
        )
        self.repo.git_pull()
        updated_metadata = metadata_load(self.repo_path / REPO_NAME / "README.md")
        self.assertDictEqual(updated_metadata, expected_metadata)
