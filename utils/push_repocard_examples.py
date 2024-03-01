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
from pathlib import Path

import jinja2

from huggingface_hub import DatasetCard, ModelCard, hf_hub_download, upload_file, whoami
from huggingface_hub.constants import REPOCARD_NAME


ORG_NAME = "templates"
MODEL_CARD_REPO_ID = "templates/model-card-example"
DATASET_CARD_REPO_ID = "templates/dataset-card-example"


def check_can_push():
    """Check the user can push to the `templates/` folder with its credentials."""
    try:
        me = whoami()
    except EnvironmentError:
        print("You must be logged in to push repo card examples.")

    if all(org["name"] != ORG_NAME for org in me.get("orgs", [])):
        print(f"❌ You must have access to organization '{ORG_NAME}' to push repo card examples.")
        exit(1)


def push_model_card_example(overwrite: bool) -> None:
    """Generate an empty model card from template for documentation purposes.

    Do not push if content has not changed. Script is triggered in CI on main branch.
    Card is pushed to https://huggingface.co/templates/model-card-example.
    """
    # Not using ModelCard directly to preserve comments in metadata part
    template = jinja2.Template(ModelCard.default_template_path.read_text())
    content = template.render(
        card_data="{}",
        model_summary=(
            "This modelcard aims to be a base template for new models. "
            "It has been generated using [this raw template]"
            "(https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md?plain=1)."
        ),
    )
    if not overwrite:
        existing_content = Path(hf_hub_download(MODEL_CARD_REPO_ID, REPOCARD_NAME, repo_type="model")).read_text()
        if content == existing_content:
            print("Model Card not pushed: did not change.")
            return
    print(f"Pushing empty Model Card to Hub: {MODEL_CARD_REPO_ID}")
    upload_file(
        path_or_fileobj=content.encode(),
        path_in_repo=REPOCARD_NAME,
        repo_id=MODEL_CARD_REPO_ID,
        repo_type="model",
    )


def push_dataset_card_example(overwrite: bool) -> None:
    """Generate an empty dataset card from template for documentation purposes.

    Do not push if content has not changed. Script is triggered in CI on main branch.
    Card is pushed to https://huggingface.co/datasets/templates/dataset-card-example.
    """
    # Not using DatasetCard directly to preserve comments in metadata part
    template = jinja2.Template(DatasetCard.default_template_path.read_text())
    content = template.render(
        card_data="{}",
        dataset_summary=(
            "This dataset card aims to be a base template for new datasets. "
            "It has been generated using [this raw template]"
            "(https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md?plain=1)."
        ),
    )
    if not overwrite:
        existing_content = Path(hf_hub_download(DATASET_CARD_REPO_ID, REPOCARD_NAME, repo_type="dataset")).read_text()
        if content == existing_content:
            print("Dataset Card not pushed: did not change.")
            return
    print(f"Pushing empty Dataset Card to Hub: {DATASET_CARD_REPO_ID}")
    upload_file(
        path_or_fileobj=content.encode(),
        path_in_repo=REPOCARD_NAME,
        repo_id=DATASET_CARD_REPO_ID,
        repo_type="dataset",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to force updating examples. By default, push to hub only if card is updated.",
    )
    args = parser.parse_args()

    check_can_push()
    push_model_card_example(args.overwrite)
    push_dataset_card_example(args.overwrite)
