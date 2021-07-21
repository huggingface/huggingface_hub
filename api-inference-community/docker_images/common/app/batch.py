#!/usr/bin/env python
import io
import os

import datasets
import tqdm
from app.main import get_pipeline
from huggingface_hub import HfApi


DATASET_NAME = os.getenv("DATASET_NAME")
DATASET_CONFIG = os.getenv("DATASET_CONFIG", None)
DATASET_SPLIT = os.getenv("DATASET_SPLIT")
DATASET_COLUMN = os.getenv("DATASET_COLUMN")
USE_GPU = os.getenv("USE_GPU", "0").lower() in {"1", "true"}
TOKEN = os.getenv("TOKEN")
REPO_ID = os.getenv("REPO_ID")


def iterate(pipe, dataset, f):
    for i, result in enumerate(tqdm.tqdm(pipe.iter(dataset))):
        f.write(str(result).encode("utf-8"))
        f.write(b"\n")


def iterate_slow(pipe, dataset, f):
    for i, item in enumerate(tqdm.tqdm(dataset)):
        try:
            result = pipe(item)
        except Exception as e:
            result = {"error": str(e)}

        f.write(str(result).encode("utf-8"))
        f.write(b"\n")


def main():
    dset = datasets.load_dataset(DATASET_NAME, name=DATASET_CONFIG, split=DATASET_SPLIT)

    pipe = get_pipeline()

    f = io.BytesIO()
    filename = f"data_{DATASET_CONFIG}_{DATASET_SPLIT}_{DATASET_COLUMN}.txt"
    # TODO change to .iter(...) to get max performance on GPUs
    print("Start batch")

    if hasattr(pipe, "iter"):
        iterate(pipe, dset[DATASET_COLUMN], f)
    else:
        iterate_slow(pipe, dset[DATASET_COLUMN], f)

    f.seek(0)

    api = HfApi()
    repo_id = REPO_ID
    try:
        api.upload_file(TOKEN, f, filename, repo_id, repo_type="dataset")
    except Exception:
        pass


if __name__ == "__main__":
    main()
