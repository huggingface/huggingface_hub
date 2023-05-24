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
import re
import unittest
from pathlib import Path

import pytest
import requests
import yaml

from huggingface_hub import (
    DatasetCard,
    DatasetCardData,
    EvalResult,
    ModelCard,
    ModelCardData,
    RepoCard,
    SpaceCard,
    SpaceCardData,
    metadata_eval_result,
    metadata_load,
    metadata_save,
    metadata_update,
)
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import HfApi
from huggingface_hub.repocard import REGEX_YAML_BLOCK
from huggingface_hub.repocard_data import CardData
from huggingface_hub.utils import SoftTemporaryDirectory, is_jinja_available, logging

from .testing_constants import (
    ENDPOINT_STAGING,
    ENDPOINT_STAGING_BASIC_AUTH,
    TOKEN,
    USER,
)
from .testing_utils import (
    repo_name,
    retry_endpoint,
    with_production_testing,
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
---
license: mit
datasets:
- foo
- bar
---

Hello
"""

DUMMY_MODELCARD_TARGET = """---
meaning_of_life: 42
---

Hello
"""

DUMMY_MODELCARD_TARGET_WITH_EMOJI = """---
emoji: 🎁
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
      verified: true
---
"""

DUMMY_MODELCARD_NO_TEXT_CONTENT = """---
license: cc-by-sa-4.0
---
"""

DUMMY_MODELCARD_EVAL_RESULT_BOTH_VERIFIED_AND_UNVERIFIED = """---
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
      value: 0.6666666666666666
      name: Accuracy
      config: default
      verified: true
---

This is a test model card.
"""


def require_jinja(test_case):
    """
    Decorator marking a test that requires Jinja2.

    These tests are skipped when Jinja2 is not installed.
    """
    if not is_jinja_available():
        return unittest.skip("test requires Jinja2.")(test_case)
    else:
        return test_case


@pytest.mark.usefixtures("fx_cache_dir")
class RepocardMetadataTest(unittest.TestCase):
    cache_dir: Path

    def setUp(self) -> None:
        self.filepath = self.cache_dir / REPOCARD_NAME

    def test_metadata_load(self):
        self.filepath.write_text(DUMMY_MODELCARD)
        data = metadata_load(self.filepath)
        self.assertDictEqual(data, {"license": "mit", "datasets": ["foo", "bar"]})

    def test_metadata_save(self):
        self.filepath.write_text(DUMMY_MODELCARD)
        metadata_save(self.filepath, {"meaning_of_life": 42})
        content = self.filepath.read_text()
        self.assertEqual(content, DUMMY_MODELCARD_TARGET)

    def test_metadata_save_with_emoji_character(self):
        self.filepath.write_text(DUMMY_MODELCARD)
        metadata_save(self.filepath, {"emoji": "🎁"})
        content = self.filepath.read_text(encoding="utf-8")
        self.assertEqual(content, DUMMY_MODELCARD_TARGET_WITH_EMOJI)

    def test_metadata_save_from_file_no_yaml(self):
        self.filepath.write_text("Hello\n")
        metadata_save(self.filepath, {"meaning_of_life": 42})
        content = self.filepath.read_text()
        self.assertEqual(content, DUMMY_MODELCARD_TARGET_NO_YAML)

    def test_metadata_save_new_file(self):
        metadata_save(self.filepath, {"meaning_of_life": 42})
        content = self.filepath.read_text()
        self.assertEqual(content, DUMMY_NEW_MODELCARD_TARGET)

    def test_no_metadata_returns_none(self):
        self.filepath.write_text(DUMMY_MODELCARD_TARGET_NO_TAGS)
        data = metadata_load(self.filepath)
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
            metrics_verified=True,
            dataset_pretty_name="ReactionGIF",
            dataset_id="julien-c/reactiongif",
            dataset_config="default",
            dataset_split="test",
        )
        metadata_save(self.filepath, data)
        content = self.filepath.read_text().splitlines()
        self.assertEqual(content, DUMMY_MODELCARD_EVAL_RESULT.splitlines())


class RepocardMetadataUpdateTest(unittest.TestCase):
    @retry_endpoint
    def setUp(self) -> None:
        self.token = TOKEN
        self.api = HfApi(token=TOKEN)

        self.repo_id = self.api.create_repo(repo_name()).repo_id
        self.api.upload_file(
            path_or_fileobj=DUMMY_MODELCARD_EVAL_RESULT.encode(), repo_id=self.repo_id, path_in_repo=REPOCARD_NAME
        )
        self.existing_metadata = yaml.safe_load(DUMMY_MODELCARD_EVAL_RESULT.strip().strip("-"))

    def tearDown(self) -> None:
        self.api.delete_repo(repo_id=self.repo_id)

    def _get_remote_card(self) -> str:
        return hf_hub_download(repo_id=self.repo_id, filename=REPOCARD_NAME)

    def test_update_dataset_name(self):
        new_datasets_data = {"datasets": ["test/test_dataset"]}
        metadata_update(self.repo_id, new_datasets_data, token=self.token)

        hf_hub_download(repo_id=self.repo_id, filename=REPOCARD_NAME)
        updated_metadata = metadata_load(self._get_remote_card())
        expected_metadata = copy.deepcopy(self.existing_metadata)
        expected_metadata.update(new_datasets_data)
        self.assertDictEqual(updated_metadata, expected_metadata)

    def test_update_existing_result_with_overwrite(self):
        new_metadata = copy.deepcopy(self.existing_metadata)
        new_metadata["model-index"][0]["results"][0]["metrics"][0]["value"] = 0.2862102282047272
        metadata_update(self.repo_id, new_metadata, token=self.token, overwrite=True)

        updated_metadata = metadata_load(self._get_remote_card())
        self.assertDictEqual(updated_metadata, new_metadata)

    def test_update_verify_token(self):
        """Tests whether updating the verification token updates in-place.

        Regression test for https://github.com/huggingface/huggingface_hub/issues/1210
        """
        new_metadata = copy.deepcopy(self.existing_metadata)
        new_metadata["model-index"][0]["results"][0]["metrics"][0]["verifyToken"] = "1234"
        metadata_update(self.repo_id, new_metadata, token=self.token, overwrite=True)

        updated_metadata = metadata_load(self._get_remote_card())
        self.assertDictEqual(updated_metadata, new_metadata)

    def test_metadata_update_upstream(self):
        new_metadata = copy.deepcopy(self.existing_metadata)
        new_metadata["model-index"][0]["results"][0]["metrics"][0]["value"] = 0.1

        # download first, then update
        path = self._get_remote_card()
        metadata_update(self.repo_id, new_metadata, token=self.token, overwrite=True)

        self.assertNotEqual(metadata_load(path), new_metadata)
        self.assertEqual(metadata_load(path), self.existing_metadata)

    def test_update_existing_result_without_overwrite(self):
        new_metadata = copy.deepcopy(self.existing_metadata)
        new_metadata["model-index"][0]["results"][0]["metrics"][0]["value"] = 0.2862102282047272

        with pytest.raises(
            ValueError,
            match=(
                "You passed a new value for the existing metric 'name: Accuracy, type:"
                " accuracy'. Set `overwrite=True` to overwrite existing metrics."
            ),
        ):
            metadata_update(self.repo_id, new_metadata, token=self.token, overwrite=False)

    def test_update_existing_field_without_overwrite(self):
        new_datasets_data = {"datasets": "['test/test_dataset']"}
        metadata_update(self.repo_id, new_datasets_data, token=self.token)

        with pytest.raises(
            ValueError,
            match=(
                "You passed a new value for the existing meta data field 'datasets'."
                " Set `overwrite=True` to overwrite existing metadata."
            ),
        ):
            new_datasets_data = {"datasets": "['test/test_dataset_2']"}
            metadata_update(self.repo_id, new_datasets_data, token=self.token, overwrite=False)

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

        metadata_update(self.repo_id, new_result, token=self.token, overwrite=False)

        expected_metadata = copy.deepcopy(self.existing_metadata)
        expected_metadata["model-index"][0]["results"][0]["metrics"].append(
            new_result["model-index"][0]["results"][0]["metrics"][0]
        )

        updated_metadata = metadata_load(self._get_remote_card())
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

        metadata_update(self.repo_id, new_result, token=self.token, overwrite=False)

        expected_metadata = copy.deepcopy(self.existing_metadata)
        expected_metadata["model-index"][0]["results"].append(new_result["model-index"][0]["results"][0])

        updated_metadata = metadata_load(self._get_remote_card())
        self.assertDictEqual(updated_metadata, expected_metadata)

    def test_update_metadata_on_empty_text_content(self) -> None:
        """Test `update_metadata` on a model card that has metadata but no text content

        Regression test for https://github.com/huggingface/huggingface_hub/issues/1010
        """
        # Create modelcard with metadata but empty text content
        self.api.upload_file(
            path_or_fileobj=DUMMY_MODELCARD_NO_TEXT_CONTENT.encode(), path_in_repo=REPOCARD_NAME, repo_id=self.repo_id
        )
        metadata_update(self.repo_id, {"tag": "test"}, token=self.token)

        # Check update went fine
        updated_metadata = metadata_load(self._get_remote_card())
        expected_metadata = {"license": "cc-by-sa-4.0", "tag": "test"}
        self.assertDictEqual(updated_metadata, expected_metadata)

    def test_update_with_existing_name(self):
        new_metadata = copy.deepcopy(self.existing_metadata)
        new_metadata["model-index"][0].pop("name")
        new_metadata["model-index"][0]["results"][0]["metrics"][0]["value"] = 0.2862102282047272
        metadata_update(self.repo_id, new_metadata, token=self.token, overwrite=True)

        card_data = ModelCard.load(self.repo_id)
        self.assertEqual(card_data.data.model_name, self.existing_metadata["model-index"][0]["name"])

    def test_update_without_existing_name(self):
        # delete existing metadata
        self.api.upload_file(path_or_fileobj="# Test".encode(), repo_id=self.repo_id, path_in_repo="README.md")

        new_metadata = copy.deepcopy(self.existing_metadata)
        new_metadata["model-index"][0].pop("name")

        metadata_update(self.repo_id, new_metadata, token=self.token, overwrite=True)

        card_data = ModelCard.load(self.repo_id)
        self.assertEqual(card_data.data.model_name, self.repo_id)

    def test_update_with_both_verified_and_unverified_metric(self):
        """Regression test for #1185.

        See https://github.com/huggingface/huggingface_hub/issues/1185.
        """
        self.api.upload_file(
            path_or_fileobj=DUMMY_MODELCARD_EVAL_RESULT_BOTH_VERIFIED_AND_UNVERIFIED.encode(),
            repo_id=self.repo_id,
            path_in_repo="README.md",
        )
        card = ModelCard.load(self.repo_id)
        metadata = card.data.to_dict()
        metadata_update(self.repo_id, metadata=metadata, overwrite=True, token=self.token)

        new_card = ModelCard.load(self.repo_id)
        self.assertEqual(len(new_card.data.eval_results), 2)
        first_result = new_card.data.eval_results[0]
        second_result = new_card.data.eval_results[1]

        # One is verified, the other not
        self.assertFalse(first_result.verified)
        self.assertTrue(second_result.verified)

        # Result values are different
        self.assertEqual(first_result.metric_value, 0.2662102282047272)
        self.assertEqual(second_result.metric_value, 0.6666666666666666)


class TestMetadataUpdateOnMissingCard(unittest.TestCase):
    def setUp(self) -> None:
        """
        Share this valid token in all tests below.
        """
        self._token = TOKEN
        self._api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)
        self._repo_id = f"{USER}/{repo_name()}"

    def test_metadata_update_missing_readme_on_model(self) -> None:
        self._api.create_repo(self._repo_id)
        metadata_update(self._repo_id, {"tag": "this_is_a_test"}, token=self._token)
        model_card = ModelCard.load(self._repo_id, token=self._token)

        # Created a card with default template + metadata
        self.assertIn("# Model Card for Model ID", str(model_card))
        self.assertEqual(model_card.data.to_dict(), {"tag": "this_is_a_test"})

        self._api.delete_repo(self._repo_id)

    def test_metadata_update_missing_readme_on_dataset(self) -> None:
        self._api.create_repo(self._repo_id, repo_type="dataset")
        metadata_update(
            self._repo_id,
            {"tag": "this is a dataset test"},
            token=self._token,
            repo_type="dataset",
        )
        dataset_card = DatasetCard.load(self._repo_id, token=self._token)

        # Created a card with default template + metadata
        self.assertIn("# Dataset Card for Dataset Name", str(dataset_card))
        self.assertEqual(dataset_card.data.to_dict(), {"tag": "this is a dataset test"})

        self._api.delete_repo(self._repo_id, repo_type="dataset")

    def test_metadata_update_missing_readme_on_space(self) -> None:
        self._api.create_repo(self._repo_id, repo_type="space", space_sdk="static")
        self._api.delete_file("README.md", self._repo_id, repo_type="space")
        with self.assertRaises(ValueError):
            # Cannot create a default readme on a space repo (should be automatically
            # created on the Hub).
            metadata_update(
                self._repo_id,
                {"tag": "this is a space test"},
                token=self._token,
                repo_type="space",
            )
        self._api.delete_repo(self._repo_id, repo_type="space")


class TestCaseWithCapLog(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING, token=TOKEN)

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        """Assign pytest caplog as attribute so we can use captured log messages in tests below."""
        self.caplog = caplog


class RepoCardTest(TestCaseWithCapLog):
    def test_load_repocard_from_file(self):
        sample_path = SAMPLE_CARDS_DIR / "sample_simple.md"
        card = RepoCard.load(sample_path)
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

    def test_change_repocard_data(self):
        sample_path = SAMPLE_CARDS_DIR / "sample_simple.md"
        card = RepoCard.load(sample_path)
        card.data.language = ["fr"]

        with SoftTemporaryDirectory() as tempdir:
            updated_card_path = Path(tempdir) / "updated.md"
            card.save(updated_card_path)

            updated_card = RepoCard.load(updated_card_path)
            self.assertEqual(updated_card.data.language, ["fr"], "Card data not updated properly")

    @require_jinja
    def test_repo_card_from_default_template(self):
        card = RepoCard.from_template(
            card_data=CardData(
                language="en",
                license="mit",
                library_name="pytorch",
                tags=["image-classification", "resnet"],
                datasets="imagenet",
                metrics=["acc", "f1"],
            ),
            model_id=None,
        )
        self.assertIsInstance(card, RepoCard)
        self.assertTrue(
            card.text.strip().startswith("# Model Card for Model ID"),
            "Default model name not set correctly",
        )

    @require_jinja
    def test_repo_card_from_default_template_with_model_id(self):
        card = RepoCard.from_template(
            card_data=CardData(
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
    def test_repo_card_from_custom_template(self):
        template_path = SAMPLE_CARDS_DIR / "sample_template.md"
        card = RepoCard.from_template(
            card_data=CardData(
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

    def test_repo_card_data_must_be_dict(self):
        sample_path = SAMPLE_CARDS_DIR / "sample_invalid_card_data.md"
        with pytest.raises(ValueError, match="repo card metadata block should be a dict"):
            RepoCard(sample_path.read_text())

    def test_repo_card_without_metadata(self):
        sample_path = SAMPLE_CARDS_DIR / "sample_no_metadata.md"

        with self.caplog.at_level(logging.WARNING):
            card = RepoCard(sample_path.read_text())
        self.assertIn(
            "Repo card metadata block was not found. Setting CardData to empty.",
            self.caplog.text,
        )
        self.assertEqual(card.data, CardData())

    def test_validate_repocard(self):
        sample_path = SAMPLE_CARDS_DIR / "sample_simple.md"
        card = RepoCard.load(sample_path)
        card.validate()

        card.data.license = "asdf"
        with pytest.raises(ValueError, match='- Error: "license" must be one of'):
            card.validate()

    def test_push_to_hub(self):
        repo_id = f"{USER}/{repo_name('push-card')}"
        self._api.create_repo(repo_id)

        card_data = CardData(
            language="en",
            license="mit",
            library_name="pytorch",
            tags=["text-classification"],
            datasets="glue",
            metrics="acc",
        )
        # Mock what RepoCard.from_template does so we can test w/o Jinja2
        content = f"---\n{card_data.to_yaml()}\n---\n\n# MyModel\n\nHello, world!"
        card = RepoCard(content)

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

        self._api.delete_repo(repo_id=repo_id)

    def test_push_and_create_pr(self):
        repo_id = f"{USER}/{repo_name('pr-card')}"
        self._api.create_repo(repo_id)
        card_data = CardData(
            language="en",
            license="mit",
            library_name="pytorch",
            tags=["text-classification"],
            datasets="glue",
            metrics="acc",
        )
        # Mock what RepoCard.from_template does so we can test w/o Jinja2
        content = f"---\n{card_data.to_yaml()}\n---\n\n# MyModel\n\nHello, world!"
        card = RepoCard(content)

        url = f"{ENDPOINT_STAGING_BASIC_AUTH}/api/models/{repo_id}/discussions"
        r = requests.get(url)
        data = r.json()
        self.assertEqual(data["count"], 0)
        card.push_to_hub(repo_id, token=TOKEN, create_pr=True)
        r = requests.get(url)
        data = r.json()
        self.assertEqual(data["count"], 1)

        self._api.delete_repo(repo_id=repo_id)

    def test_preserve_windows_linebreaks(self):
        card_path = SAMPLE_CARDS_DIR / "sample_windows_line_breaks.md"
        card = RepoCard.load(card_path)
        self.assertIn("\r\n", str(card))

    def test_preserve_linebreaks_when_saving(self):
        card_path = SAMPLE_CARDS_DIR / "sample_simple.md"
        card = RepoCard.load(card_path)
        with SoftTemporaryDirectory() as tmpdir:
            tmpfile = os.path.join(tmpdir, "readme.md")
            card.save(tmpfile)
            card2 = RepoCard.load(tmpfile)
        self.assertEqual(str(card), str(card2))

    def test_updating_text_updates_content(self):
        sample_path = SAMPLE_CARDS_DIR / "sample_simple.md"
        card = RepoCard.load(sample_path)
        card.text = "Hello, world!"
        line_break = "\r\n" if os.name == "nt" else "\n"
        self.assertEqual(
            card.content,
            # line_break depends on platform. Correctly set when using RepoCard.save(...) to avoid diffs
            f"---\n{card.data.to_yaml()}\n---\nHello, world!".replace("\n", line_break),
        )


class TestRegexYamlBlock(unittest.TestCase):
    def test_match_with_leading_whitespace(self):
        self.assertIsNotNone(REGEX_YAML_BLOCK.search("   \n---\nmetadata: 1\n---"))

    def test_match_without_leading_whitespace(self):
        self.assertIsNotNone(REGEX_YAML_BLOCK.search("---\nmetadata: 1\n---"))

    def test_does_not_match_with_leading_text(self):
        self.assertIsNone(REGEX_YAML_BLOCK.search("something\n---\nmetadata: 1\n---"))


class ModelCardTest(TestCaseWithCapLog):
    def test_model_card_with_invalid_model_index(self):
        """Test raise an error when loading a card that has invalid model-index."""
        sample_path = SAMPLE_CARDS_DIR / "sample_invalid_model_index.md"
        with self.assertRaises(ValueError):
            ModelCard.load(sample_path)

    def test_model_card_with_invalid_model_index_and_ignore_error(self):
        """Test trigger a warning when loading a card that has invalid model-index and `ignore_metadata_errors=True`

        Some information is lost.
        """
        sample_path = SAMPLE_CARDS_DIR / "sample_invalid_model_index.md"
        with self.caplog.at_level(logging.WARNING):
            card = ModelCard.load(sample_path, ignore_metadata_errors=True)
        self.assertIn(
            "Invalid model-index. Not loading eval results into CardData.",
            self.caplog.text,
        )
        self.assertIsNone(card.data.eval_results)

    def test_model_card_with_model_index(self):
        """Test that loading a model card with multiple evaluations is consistent with `metadata_load`.

        Regression test for https://github.com/huggingface/huggingface_hub/issues/1208
        """
        sample_path = SAMPLE_CARDS_DIR / "sample_simple_model_index.md"
        card = ModelCard.load(sample_path)
        metadata = metadata_load(sample_path)
        self.assertDictEqual(card.data.to_dict(), metadata)

    def test_load_model_card_from_file(self):
        sample_path = SAMPLE_CARDS_DIR / "sample_simple.md"
        card = ModelCard.load(sample_path)
        self.assertIsInstance(card, ModelCard)
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
        self.assertIsInstance(card, ModelCard)
        self.assertTrue(
            card.text.endswith("asdf"),
            "Custom template didn't set jinja variable correctly",
        )

    @require_jinja
    def test_model_card_from_template_eval_results(self):
        template_path = SAMPLE_CARDS_DIR / "sample_template.md"
        card = ModelCard.from_template(
            card_data=ModelCardData(
                eval_results=[
                    EvalResult(
                        task_type="text-classification",
                        task_name="Text Classification",
                        dataset_type="julien-c/reactiongif",
                        dataset_name="ReactionGIF",
                        dataset_config="default",
                        dataset_split="test",
                        metric_type="accuracy",
                        metric_value=0.2662102282047272,
                        metric_name="Accuracy",
                        metric_config="default",
                        verified=True,
                    ),
                ],
                model_name="RoBERTa fine-tuned on ReactionGIF",
            ),
            template_path=template_path,
            some_data="asdf",
        )
        self.assertIsInstance(card, ModelCard)
        self.assertTrue(card.text.endswith("asdf"))
        self.assertTrue(card.data.to_dict().get("eval_results") is None)
        self.assertEqual(str(card)[: len(DUMMY_MODELCARD_EVAL_RESULT)], DUMMY_MODELCARD_EVAL_RESULT)


class DatasetCardTest(TestCaseWithCapLog):
    def test_load_datasetcard_from_file(self):
        sample_path = SAMPLE_CARDS_DIR / "sample_datasetcard_simple.md"
        card = DatasetCard.load(sample_path)
        self.assertEqual(
            card.data.to_dict(),
            {
                "annotations_creators": ["crowdsourced", "expert-generated"],
                "language_creators": ["found"],
                "language": ["en"],
                "license": ["bsd-3-clause"],
                "multilinguality": ["monolingual"],
                "size_categories": ["n<1K"],
                "task_categories": ["image-segmentation"],
                "task_ids": ["semantic-segmentation"],
                "pretty_name": "Sample Segmentation",
            },
        )
        self.assertIsInstance(card, DatasetCard)
        self.assertIsInstance(card.data, DatasetCardData)
        self.assertTrue(card.text.strip().startswith("# Dataset Card for"))

    @require_jinja
    def test_dataset_card_from_default_template(self):
        card_data = DatasetCardData(
            language="en",
            license="mit",
        )

        # Here we check default title when pretty_name not provided.
        card = DatasetCard.from_template(card_data)
        self.assertTrue(card.text.strip().startswith("# Dataset Card for Dataset Name"))

        card_data = DatasetCardData(
            language="en",
            license="mit",
            pretty_name="My Cool Dataset",
        )

        # Here we pass the card data as kwargs as well so template picks up pretty_name.
        card = DatasetCard.from_template(card_data, **card_data.to_dict())
        self.assertTrue(card.text.strip().startswith("# Dataset Card for My Cool Dataset"))

        self.assertIsInstance(card, DatasetCard)

    @require_jinja
    def test_dataset_card_from_default_template_with_template_variables(self):
        card_data = DatasetCardData(
            language="en",
            license="mit",
            pretty_name="My Cool Dataset",
        )

        # Here we pass the card data as kwargs as well so template picks up pretty_name.
        card = DatasetCard.from_template(
            card_data,
            homepage_url="https://huggingface.co",
            repo_url="https://github.com/huggingface/huggingface_hub",
            paper_url="https://arxiv.org/pdf/1910.03771.pdf",
            point_of_contact="https://huggingface.co/nateraw",
            dataset_summary=(
                "This is a test dataset card to check if the template variables "
                "in the dataset card template are working."
            ),
        )
        self.assertTrue(card.text.strip().startswith("# Dataset Card for My Cool Dataset"))
        self.assertIsInstance(card, DatasetCard)

        matches = re.findall(r"Homepage:\*\* https:\/\/huggingface\.co", str(card))
        self.assertEqual(matches[0], "Homepage:** https://huggingface.co")

    @require_jinja
    def test_dataset_card_from_custom_template(self):
        card = DatasetCard.from_template(
            card_data=DatasetCardData(
                language="en",
                license="mit",
                pretty_name="My Cool Dataset",
            ),
            template_path=SAMPLE_CARDS_DIR / "sample_datasetcard_template.md",
            pretty_name="My Cool Dataset",
            some_data="asdf",
        )
        self.assertIsInstance(card, DatasetCard)

        # Title this time is just # {{ pretty_name }}
        self.assertTrue(card.text.strip().startswith("# My Cool Dataset"))

        # some_data is at the bottom of the template, so should end with whatever we passed to it
        self.assertTrue(card.text.strip().endswith("asdf"))


@with_production_testing
class SpaceCardTest(TestCaseWithCapLog):
    def test_load_spacecard_from_hub(self) -> None:
        card = SpaceCard.load("multimodalart/dreambooth-training")
        self.assertIsInstance(card, SpaceCard)
        self.assertIsInstance(card.data, SpaceCardData)
        self.assertEqual(card.data.title, "Dreambooth Training")
        self.assertIsNone(card.data.app_port)
