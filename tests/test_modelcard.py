import logging
import tempfile
import unittest
from pathlib import Path

import pytest

import requests
from huggingface_hub import HfApi, ModelCard, ModelCardData
from huggingface_hub.file_download import is_jinja_available

from .testing_constants import (
    ENDPOINT_STAGING,
    ENDPOINT_STAGING_BASIC_AUTH,
    TOKEN,
    USER,
)
from .testing_utils import repo_name


SAMPLE_CARDS_DIR = Path(__file__).parent / "fixtures/cards"


def require_jinja(test_case):
    """
    Decorator marking a test that requires Jinja2.

    These tests are skipped when Jinja2 is not installed.

    """
    if not is_jinja_available():
        return unittest.skip("test requires Jinja2.")(test_case)
    else:
        return test_case


class ModelCardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._api = HfApi(endpoint=ENDPOINT_STAGING)
        cls._api.set_access_token(TOKEN)

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
