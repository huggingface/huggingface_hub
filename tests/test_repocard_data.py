import unittest

import pytest
import yaml

from huggingface_hub import SpaceCardData
from huggingface_hub.repocard_data import (
    CardData,
    DatasetCardData,
    EvalResult,
    ModelCardData,
    eval_results_to_model_index,
    model_index_to_eval_results,
)


OPEN_LLM_LEADERBOARD_URL = "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard"
DUMMY_METADATA_WITH_MODEL_INDEX = """
language: en
license: mit
library_name: timm
tags:
- pytorch
- image-classification
datasets:
- beans
metrics:
- acc
model-index:
- name: my-cool-model
  results:
  - task:
      type: image-classification
    dataset:
      type: beans
      name: Beans
    metrics:
    - type: acc
      value: 0.9
    source:
      name: Open LLM Leaderboard
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
"""


class BaseCardDataTest(unittest.TestCase):
    def test_metadata_behave_as_dict(self):
        metadata = CardData(foo="bar")

        # .get and __getitem__
        self.assertEqual(metadata.get("foo"), "bar")
        self.assertEqual(metadata.get("FOO"), None)  # case sensitive
        self.assertEqual(metadata["foo"], "bar")
        with self.assertRaises(KeyError):  # case sensitive
            _ = metadata["FOO"]

        # __setitem__
        metadata["foo"] = "BAR"
        self.assertEqual(metadata.get("foo"), "BAR")
        self.assertEqual(metadata["foo"], "BAR")

        # __contains__
        self.assertTrue("foo" in metadata)
        self.assertFalse("FOO" in metadata)

        # export
        self.assertEqual(str(metadata), "foo: BAR")

        # .pop
        self.assertEqual(metadata.pop("foo"), "BAR")


class ModelCardDataTest(unittest.TestCase):
    def test_eval_results_to_model_index(self):
        expected_results = yaml.safe_load(DUMMY_METADATA_WITH_MODEL_INDEX)

        eval_results = [
            EvalResult(
                task_type="image-classification",
                dataset_type="beans",
                dataset_name="Beans",
                metric_type="acc",
                metric_value=0.9,
                source_name="Open LLM Leaderboard",
                source_url=OPEN_LLM_LEADERBOARD_URL,
            ),
        ]

        model_index = eval_results_to_model_index("my-cool-model", eval_results)

        self.assertEqual(model_index, expected_results["model-index"])

    def test_model_index_to_eval_results(self):
        model_index = [
            {
                "name": "my-cool-model",
                "results": [
                    {
                        "task": {
                            "type": "image-classification",
                        },
                        "dataset": {
                            "type": "cats_vs_dogs",
                            "name": "Cats vs. Dogs",
                        },
                        "metrics": [
                            {
                                "type": "acc",
                                "value": 0.85,
                            },
                            {
                                "type": "f1",
                                "value": 0.9,
                            },
                        ],
                    },
                    {
                        "task": {
                            "type": "image-classification",
                        },
                        "dataset": {
                            "type": "beans",
                            "name": "Beans",
                        },
                        "metrics": [
                            {
                                "type": "acc",
                                "value": 0.9,
                                "verified": True,
                                "verifyToken": 1234,
                            }
                        ],
                        "source": {
                            "name": "Open LLM Leaderboard",
                            "url": OPEN_LLM_LEADERBOARD_URL,
                        },
                    },
                ],
            }
        ]
        model_name, eval_results = model_index_to_eval_results(model_index)

        self.assertEqual(len(eval_results), 3)
        self.assertEqual(model_name, "my-cool-model")

        self.assertEqual(eval_results[0].dataset_type, "cats_vs_dogs")
        self.assertIsNone(eval_results[0].source_name)
        self.assertIsNone(eval_results[0].source_url)

        self.assertEqual(eval_results[1].metric_type, "f1")
        self.assertEqual(eval_results[1].metric_value, 0.9)
        self.assertIsNone(eval_results[1].source_name)
        self.assertIsNone(eval_results[1].source_url)

        self.assertEqual(eval_results[2].task_type, "image-classification")
        self.assertEqual(eval_results[2].dataset_type, "beans")
        self.assertEqual(eval_results[2].verified, True)
        self.assertEqual(eval_results[2].verify_token, 1234)
        self.assertEqual(eval_results[2].source_name, "Open LLM Leaderboard")
        self.assertEqual(eval_results[2].source_url, OPEN_LLM_LEADERBOARD_URL)

    def test_card_data_requires_model_name_for_eval_results(self):
        with pytest.raises(ValueError, match="`eval_results` requires `model_name` to be set."):
            ModelCardData(
                eval_results=[
                    EvalResult(
                        task_type="image-classification",
                        dataset_type="beans",
                        dataset_name="Beans",
                        metric_type="acc",
                        metric_value=0.9,
                    ),
                ],
            )

        data = ModelCardData(
            model_name="my-cool-model",
            eval_results=[
                EvalResult(
                    task_type="image-classification",
                    dataset_type="beans",
                    dataset_name="Beans",
                    metric_type="acc",
                    metric_value=0.9,
                ),
            ],
        )

        model_index = eval_results_to_model_index(data.model_name, data.eval_results)

        self.assertEqual(model_index[0]["name"], "my-cool-model")
        self.assertEqual(model_index[0]["results"][0]["task"]["type"], "image-classification")

    def test_arbitrary_incoming_card_data(self):
        data = ModelCardData(
            model_name="my-cool-model",
            eval_results=[
                EvalResult(
                    task_type="image-classification",
                    dataset_type="beans",
                    dataset_name="Beans",
                    metric_type="acc",
                    metric_value=0.9,
                ),
            ],
            some_arbitrary_kwarg="some_value",
        )

        self.assertEqual(data.some_arbitrary_kwarg, "some_value")

        data_dict = data.to_dict()
        self.assertEqual(data_dict["some_arbitrary_kwarg"], "some_value")

    def test_eval_result_with_incomplete_source(self):
        # Source url without name: ok
        EvalResult(
            task_type="image-classification",
            dataset_type="beans",
            dataset_name="Beans",
            metric_type="acc",
            metric_value=0.9,
            source_url=OPEN_LLM_LEADERBOARD_URL,
        )

        # Source name without url: not ok
        with self.assertRaises(ValueError):
            EvalResult(
                task_type="image-classification",
                dataset_type="beans",
                dataset_name="Beans",
                metric_type="acc",
                metric_value=0.9,
                source_name="Open LLM Leaderboard",
            )

    def test_model_card_unique_tags(self):
        data = ModelCardData(tags=["tag2", "tag1", "tag2", "tag3"])
        assert data.tags == ["tag2", "tag1", "tag3"]


class DatasetCardDataTest(unittest.TestCase):
    def test_train_eval_index_keys_updated(self):
        train_eval_index = [
            {
                "config": "plain_text",
                "task": "text-classification",
                "task_id": "binary_classification",
                "splits": {"train_split": "train", "eval_split": "test"},
                "col_mapping": {"text": "text", "label": "target"},
                "metrics": [
                    {
                        "type": "accuracy",
                        "name": "Accuracy",
                    },
                    {"type": "f1", "name": "F1 macro", "args": {"average": "macro"}},
                ],
            }
        ]
        card_data = DatasetCardData(
            language="en",
            license="mit",
            pretty_name="My Cool Dataset",
            train_eval_index=train_eval_index,
        )
        # The init should have popped this out of kwargs and into train_eval_index attr
        self.assertEqual(card_data.train_eval_index, train_eval_index)
        # Underlying train_eval_index gets converted to train-eval-index in DatasetCardData._to_dict.
        # So train_eval_index should be None in the dict
        self.assertTrue(card_data.to_dict().get("train_eval_index") is None)
        # And train-eval-index should be in the dict
        self.assertEqual(card_data.to_dict()["train-eval-index"], train_eval_index)


class SpaceCardDataTest(unittest.TestCase):
    def test_space_card_data(self) -> None:
        card_data = SpaceCardData(
            title="Dreambooth Training",
            license="mit",
            sdk="gradio",
            duplicated_from="multimodalart/dreambooth-training",
        )
        self.assertEqual(
            card_data.to_dict(),
            {
                "title": "Dreambooth Training",
                "sdk": "gradio",
                "license": "mit",
                "duplicated_from": "multimodalart/dreambooth-training",
            },
        )
        self.assertIsNone(card_data.tags)  # SpaceCardData has some default attributes
