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
import logging
import os
import shutil
import tempfile
import unittest
from functools import partial
from pathlib import Path

import pytest

import requests
import yaml
from huggingface_hub import (
    ModelCard,
    ModelCardData,
    metadata_eval_result,
    metadata_load,
    metadata_save,
    metadata_update,
)
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.file_download import hf_hub_download, is_jinja_available
from huggingface_hub.hf_api import HfApi
from huggingface_hub.repository import Repository
from huggingface_hub.utils import logging

from .testing_constants import (
    ENDPOINT_STAGING,
    ENDPOINT_STAGING_BASIC_AUTH,
    TOKEN,
    USER,
)
from .testing_utils import (
    expect_deprecation,
    repo_name,
    retry_endpoint,
    set_write_permission_and_retry,
)


SAMPLE_CARDS_DIR = Path(__file__).parent / "fixtures/cards"

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
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: ReactionGIF
      type: julien-c/reactiongif
      config: default
      split: test
    metrics:
    - type: accuracy
      value: 0.2662102282047272
      name: Accuracy
      config: default
      verified: false
---
"""

logger = logging.get_logger(__name__)

REPOCARD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures/repocard"
)

repo_name = partial(repo_name, prefix="dummy-hf-hub")


def require_jinja(test_case):
    """
    Decorator marking a test that requires Jinja2.

    These tests are skipped when Jinja2 is not installed.

    """
    if not is_jinja_available():
        return unittest.skip("test requires Jinja2.")(test_case)
    else:
        return test_case


class RepocardMetadataTest(unittest.TestCase):
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
            metrics_config="default",
            metrics_verified=False,
            dataset_pretty_name="ReactionGIF",
            dataset_id="julien-c/reactiongif",
            dataset_config="default",
            dataset_split="test",
        )
        filename = "eval_results.md"
        filepath = Path(REPOCARD_DIR) / filename
        metadata_save(filepath, data)
        content = filepath.read_text().splitlines()
        self.assertEqual(content, DUMMY_MODELCARD_EVAL_RESULT.splitlines())


class RepocardMetadataUpdateTest(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING)

    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = TOKEN
        cls._api.set_access_token(TOKEN)

    @retry_endpoint
    @expect_deprecation("clone_from")
    def setUp(self) -> None:
        self.repo_path = Path(tempfile.mkdtemp())
        self.REPO_NAME = repo_name()
        self.repo = Repository(
            self.repo_path / self.REPO_NAME,
            clone_from=f"{USER}/{self.REPO_NAME}",
            use_auth_token=self._token,
            git_user="ci",
            git_email="ci@dummy.com",
        )

        with self.repo.commit("Add README to main branch"):
            with open("README.md", "w+") as f:
                f.write(DUMMY_MODELCARD_EVAL_RESULT)

        self.existing_metadata = yaml.safe_load(
            DUMMY_MODELCARD_EVAL_RESULT.strip().strip("-")
        )

    def tearDown(self) -> None:
        self._api.delete_repo(repo_id=f"{self.REPO_NAME}", token=self._token)
        shutil.rmtree(self.repo_path)

    def test_update_dataset_name(self):
        new_datasets_data = {"datasets": ["test/test_dataset"]}
        metadata_update(
            f"{USER}/{self.REPO_NAME}", new_datasets_data, token=self._token
        )

        self.repo.git_pull()
        updated_metadata = metadata_load(self.repo_path / self.REPO_NAME / "README.md")
        expected_metadata = copy.deepcopy(self.existing_metadata)
        expected_metadata.update(new_datasets_data)
        self.assertDictEqual(updated_metadata, expected_metadata)

    def test_update_existing_result_with_overwrite(self):
        new_metadata = copy.deepcopy(self.existing_metadata)
        new_metadata["model-index"][0]["results"][0]["metrics"][0][
            "value"
        ] = 0.2862102282047272
        metadata_update(
            f"{USER}/{self.REPO_NAME}", new_metadata, token=self._token, overwrite=True
        )

        self.repo.git_pull()
        updated_metadata = metadata_load(self.repo_path / self.REPO_NAME / "README.md")
        self.assertDictEqual(updated_metadata, new_metadata)

    def test_metadata_update_upstream(self):
        new_metadata = copy.deepcopy(self.existing_metadata)
        new_metadata["model-index"][0]["results"][0]["metrics"][0]["value"] = 0.1

        path = hf_hub_download(
            f"{USER}/{self.REPO_NAME}",
            filename=REPOCARD_NAME,
            use_auth_token=self._token,
        )

        metadata_update(
            f"{USER}/{self.REPO_NAME}", new_metadata, token=self._token, overwrite=True
        )

        self.assertNotEqual(metadata_load(path), new_metadata)
        self.assertEqual(metadata_load(path), self.existing_metadata)

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
                f"{USER}/{self.REPO_NAME}",
                new_metadata,
                token=self._token,
                overwrite=False,
            )

    def test_update_existing_field_without_overwrite(self):
        new_datasets_data = {"datasets": "['test/test_dataset']"}
        metadata_update(
            f"{USER}/{self.REPO_NAME}", new_datasets_data, token=self._token
        )

        with pytest.raises(
            ValueError,
            match=(
                "You passed a new value for the existing meta data field 'datasets'."
                " Set `overwrite=True` to overwrite existing metadata."
            ),
        ):
            new_datasets_data = {"datasets": "['test/test_dataset_2']"}
            metadata_update(
                f"{USER}/{self.REPO_NAME}",
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
            metrics_config="default",
            metrics_verified=False,
            dataset_pretty_name="ReactionGIF",
            dataset_id="julien-c/reactiongif",
            dataset_config="default",
            dataset_split="test",
        )

        metadata_update(
            f"{USER}/{self.REPO_NAME}", new_result, token=self._token, overwrite=False
        )

        expected_metadata = copy.deepcopy(self.existing_metadata)
        expected_metadata["model-index"][0]["results"][0]["metrics"].append(
            new_result["model-index"][0]["results"][0]["metrics"][0]
        )

        self.repo.git_pull()
        updated_metadata = metadata_load(self.repo_path / self.REPO_NAME / "README.md")
        self.assertDictEqual(updated_metadata, expected_metadata)

    def test_update_new_result_new_dataset(self):
        new_result = metadata_eval_result(
            model_pretty_name="RoBERTa fine-tuned on ReactionGIF",
            task_pretty_name="Text Classification",
            task_id="text-classification",
            metrics_pretty_name="Accuracy",
            metrics_id="accuracy",
            metrics_value=0.2662102282047272,
            metrics_config="default",
            metrics_verified=False,
            dataset_pretty_name="ReactionJPEG",
            dataset_id="julien-c/reactionjpeg",
            dataset_config="default",
            dataset_split="test",
        )

        metadata_update(
            f"{USER}/{self.REPO_NAME}", new_result, token=self._token, overwrite=False
        )

        expected_metadata = copy.deepcopy(self.existing_metadata)
        expected_metadata["model-index"][0]["results"].append(
            new_result["model-index"][0]["results"][0]
        )
        self.repo.git_pull()
        updated_metadata = metadata_load(self.repo_path / self.REPO_NAME / "README.md")
        self.assertDictEqual(updated_metadata, expected_metadata)


class ModelCardTest(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING)

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        """Assign pytest caplog as attribute so we can use captured log messages in tests below."""
        self.caplog = caplog

    def test_load_modelcard_from_file(self):
        sample_path = SAMPLE_CARDS_DIR / "sample_simple.md"
        card = ModelCard.load(sample_path)
        self.assertEqual(
            card.data.to_dict(),
            {
                "language": ["en"],
                "license": "mit",
                "library_name": "pytorch-lightning",
                "tags": ["pytorch", "image-classification"],
                "datasets": ["beans"],
                "metrics": ["acc"],
            },
        )
        self.assertTrue(
            card.text.strip().startswith("# my-cool-model"),
            "Card text not loaded properly",
        )

    def test_change_modelcard_data(self):
        sample_path = SAMPLE_CARDS_DIR / "sample_simple.md"
        card = ModelCard.load(sample_path)
        card.data.language = ["fr"]

        with tempfile.TemporaryDirectory() as tempdir:
            updated_card_path = Path(tempdir) / "updated.md"
            card.save(updated_card_path)

            updated_card = ModelCard.load(updated_card_path)
            self.assertEqual(
                updated_card.data.language, ["fr"], "Card data not updated properly"
            )

    @require_jinja
    def test_model_card_from_default_template(self):
        card = ModelCard.from_template(
            card_data=ModelCardData(
                language="en",
                license="mit",
                library_name="pytorch",
                tags=["image-classification", "resnet"],
                datasets="imagenet",
                metrics=["acc", "f1"],
            ),
            model_id=None,
        )
        self.assertTrue(
            card.text.strip().startswith("# Model Card for Model ID"),
            "Default model name not set correctly",
        )

    @require_jinja
    def test_model_card_from_default_template_with_model_id(self):
        card = ModelCard.from_template(
            card_data=ModelCardData(
                language="en",
                license="mit",
                library_name="pytorch",
                tags=["image-classification", "resnet"],
                datasets="imagenet",
                metrics=["acc", "f1"],
            ),
            model_id="my-cool-model",
        )
        self.assertTrue(
            card.text.strip().startswith("# Model Card for my-cool-model"),
            "model_id not properly set in card template",
        )

    @require_jinja
    def test_model_card_from_custom_template(self):
        template_path = SAMPLE_CARDS_DIR / "sample_template.md"
        card = ModelCard.from_template(
            card_data=ModelCardData(
                language="en",
                license="mit",
                library_name="pytorch",
                tags="text-classification",
                datasets="glue",
                metrics="acc",
            ),
            template_path=template_path,
            some_data="asdf",
        )
        self.assertTrue(
            card.text.endswith("asdf"),
            "Custom template didn't set jinja variable correctly",
        )

    def test_model_card_data_must_be_dict(self):
        sample_path = SAMPLE_CARDS_DIR / "sample_invalid_card_data.md"
        with pytest.raises(
            ValueError, match="repo card metadata block should be a dict"
        ):
            ModelCard.load(sample_path)

    def test_model_card_without_metadata(self):
        sample_path = SAMPLE_CARDS_DIR / "sample_no_metadata.md"

        with self.caplog.at_level(logging.WARNING):
            card = ModelCard.load(sample_path)
        self.assertIn(
            "Repo card metadata block was not found. Setting CardData to empty.",
            self.caplog.text,
        )
        self.assertEqual(card.data, ModelCardData())

    def test_model_card_with_invalid_model_index(self):
        """
        Test that when loading a card that has invalid model-index, no eval_results are added + it logs a warning
        """
        sample_path = SAMPLE_CARDS_DIR / "sample_invalid_model_index.md"
        with self.caplog.at_level(logging.WARNING):
            card = ModelCard.load(sample_path)
        self.assertIn(
            "Invalid model-index. Not loading eval results into CardData.",
            self.caplog.text,
        )
        self.assertIsNone(card.data.eval_results)

    def test_validate_modelcard(self):
        sample_path = SAMPLE_CARDS_DIR / "sample_simple.md"
        card = ModelCard.load(sample_path)
        card.validate()

        card.data.license = "asdf"
        with pytest.raises(RuntimeError, match='- Error: "license" must be one of'):
            card.validate()

    def test_push_to_hub(self):
        repo_id = f"{USER}/{repo_name('push-card')}"
        self._api.create_repo(repo_id, token=TOKEN)

        card_data = ModelCardData(
            language="en",
            license="mit",
            library_name="pytorch",
            tags=["text-classification"],
            datasets="glue",
            metrics="acc",
        )
        # Mock what RepoCard.from_template does so we can test w/o Jinja2
        content = f"{card_data.to_yaml()}\n\n# MyModel\n\nHello, world!"
        card = ModelCard(content)

        url = f"{ENDPOINT_STAGING_BASIC_AUTH}/{repo_id}/resolve/main/README.md"

        # Check this file doesn't exist (sanity check)
        with pytest.raises(requests.exceptions.HTTPError):
            r = requests.get(url)
            r.raise_for_status()

        # Push the card up to README.md in the repo
        card.push_to_hub(repo_id, token=TOKEN)

        # No error should occur now, as README.md should exist
        r = requests.get(url)
        r.raise_for_status()

        self._api.delete_repo(repo_id=repo_id, token=TOKEN)

    def test_push_and_create_pr(self):
        repo_id = f"{USER}/{repo_name('pr-card')}"
        self._api.create_repo(repo_id, token=TOKEN)
        card_data = ModelCardData(
            language="en",
            license="mit",
            library_name="pytorch",
            tags=["text-classification"],
            datasets="glue",
            metrics="acc",
        )
        # Mock what RepoCard.from_template does so we can test w/o Jinja2
        content = f"{card_data.to_yaml()}\n\n# MyModel\n\nHello, world!"
        card = ModelCard(content)

        url = f"{ENDPOINT_STAGING_BASIC_AUTH}/api/models/{repo_id}/discussions"
        r = requests.get(url)
        data = r.json()
        self.assertEqual(data["count"], 0)
        card.push_to_hub(repo_id, token=TOKEN, create_pr=True)
        r = requests.get(url)
        data = r.json()
        self.assertEqual(data["count"], 1)

        self._api.delete_repo(repo_id=repo_id, token=TOKEN)

    def test_preserve_windows_linebreaks(self):
        card_path = SAMPLE_CARDS_DIR / "sample_windows_line_breaks.md"
        card = ModelCard.load(card_path)
        self.assertIn("\r\n", str(card))


class DatasetCardTest(unittest.TestCase):
    pass
