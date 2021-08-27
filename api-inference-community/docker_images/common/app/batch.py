#!/usr/bin/env python

import os

from api_inference_community.batch import batch
from app.main import get_pipeline


DATASET_NAME = os.getenv("DATASET_NAME")
DATASET_CONFIG = os.getenv("DATASET_CONFIG", None)
DATASET_SPLIT = os.getenv("DATASET_SPLIT")
DATASET_COLUMN = os.getenv("DATASET_COLUMN")
USE_GPU = os.getenv("USE_GPU", "0").lower() in {"1", "true"}
TOKEN = os.getenv("TOKEN")
REPO_ID = os.getenv("REPO_ID")
TASK = os.getenv("TASK")

if __name__ == "__main__":
    batch(
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        dataset_split=DATASET_SPLIT,
        dataset_column=DATASET_COLUMN,
        token=TOKEN,
        repo_id=REPO_ID,
        use_gpu=USE_GPU,
        pipeline=get_pipeline(),
        task=TASK,
    )
