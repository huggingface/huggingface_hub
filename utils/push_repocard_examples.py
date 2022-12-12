# coding=utf-8
# Copyright 2022-present, the HuggingFace Inc. team.
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
"""Generate and push an empty ModelCard and DatasetCard to the Hub as examples."""
import argparse

from huggingface_hub import DatasetCard, DatasetCardData, ModelCard, ModelCardData


MODEL_CARD_REPO_ID = "huggingface/model-card-example"
DATASET_CARD_REPO_ID = "huggingface/dataset-card-example"


def push_model_card_example(overwrite: bool) -> None:
    """Generate an empty model card from template for documentation purposes.

    Do not push if content has not changed. Script is triggered in CI on main branch.
    Card is pushed to https://huggingface.co/huggingface/model-card-example.
    """

    card = ModelCard.from_template(ModelCardData())
    if not overwrite:
        existing_card = ModelCard.load(MODEL_CARD_REPO_ID)
        if str(existing_card) == str(card):
            print("Model Card not pushed: did not change.")
            return
    print(f"Pushing empty Model Card to Hub: {MODEL_CARD_REPO_ID}")
    card.push_to_hub(MODEL_CARD_REPO_ID)


def push_dataset_card_example(overwrite: bool) -> None:
    """Generate an empty dataset card from template for documentation purposes.

    Do not push if content has not changed. Script is triggered in CI on main branch.
    Card is pushed to https://huggingface.co/datasets/huggingface/dataset-card-example.
    """
    card = DatasetCard.from_template(DatasetCardData())
    if not overwrite:
        existing_card = DatasetCard.load(DATASET_CARD_REPO_ID)
        if str(existing_card) == str(card):
            print("Dataset Card not pushed: did not change.")
            return
    print(f"Pushing empty Dataset Card to Hub: {DATASET_CARD_REPO_ID}")
    card.push_to_hub(DATASET_CARD_REPO_ID)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Whether to force updating examples. By default, push to hub only if card"
            " is updated."
        ),
    )
    args = parser.parse_args()

    push_model_card_example(args.overwrite)
    push_dataset_card_example(args.overwrite)
