import pytest

from huggingface_hub._datasets_parquet import (
    fetch_dataset_parquet_status,
    list_dataset_parquet_entries,
)
from huggingface_hub.errors import EntryNotFoundError


def test_fetch_dataset_parquet_status_parses_incomplete_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub._datasets_parquet._fetch_json",
        lambda url, token: {
            "partial": True,
            "pending": ["models"],
            "failed": [{"config": "datasets"}],
        },
    )

    status = fetch_dataset_parquet_status(repo_id="user/repo", token="token")

    assert status.partial is True
    assert status.pending == ("models",)
    assert status.failed == ("datasets",)


def test_list_dataset_parquet_entries_from_root_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub._datasets_parquet._fetch_json",
        lambda url, token: {
            "datasets": {
                "train": ["https://example.com/datasets-train-0000.parquet"],
            },
            "models": {
                "train": ["https://example.com/models-train-0000.parquet"],
                "test": ["https://example.com/models-test-0000.parquet"],
            },
        },
    )

    entries = list_dataset_parquet_entries(repo_id="cfahlgren1/hub-stats", token="token")

    assert [(entry.config, entry.split) for entry in entries] == [
        ("datasets", "train"),
        ("models", "test"),
        ("models", "train"),
    ]
    assert (
        entries[0].parquet_file_path
        == "hf://datasets/cfahlgren1/hub-stats@~parquet/datasets/train/datasets-train-0000.parquet"
    )


def test_list_dataset_parquet_entries_returns_all_parquet_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub._datasets_parquet._fetch_json",
        lambda url, token: {
            "datasets": {
                "train": [
                    "https://example.com/datasets-train-0000.parquet",
                    "https://example.com/datasets-train-0001.parquet",
                ],
            },
        },
    )

    entries = list_dataset_parquet_entries(
        repo_id="cfahlgren1/hub-stats", token="token", config="datasets", split="train"
    )

    assert [entry.parquet_file_path for entry in entries] == [
        "hf://datasets/cfahlgren1/hub-stats@~parquet/datasets/train/datasets-train-0000.parquet",
        "hf://datasets/cfahlgren1/hub-stats@~parquet/datasets/train/datasets-train-0001.parquet",
    ]


def test_list_dataset_parquet_entries_filtered_by_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub._datasets_parquet._fetch_json",
        lambda url, token: {
            "datasets": {"train": ["https://example.com/datasets-train.parquet"]},
            "models": {"train": ["https://example.com/models-train.parquet"]},
        },
    )

    entries = list_dataset_parquet_entries(repo_id="cfahlgren1/hub-stats", token="token", config="models")

    assert [(entry.config, entry.split) for entry in entries] == [("models", "train")]


def test_list_dataset_parquet_entries_filtered_by_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub._datasets_parquet._fetch_json",
        lambda url, token: {
            "datasets": {
                "train": ["https://example.com/datasets-train.parquet"],
                "validation": ["https://example.com/datasets-validation.parquet"],
            },
            "models": {
                "train": ["https://example.com/models-train.parquet"],
                "test": ["https://example.com/models-test.parquet"],
            },
        },
    )

    entries = list_dataset_parquet_entries(repo_id="cfahlgren1/hub-stats", token="token", split="train")

    assert [(entry.config, entry.split) for entry in entries] == [
        ("datasets", "train"),
        ("models", "train"),
    ]


def test_list_dataset_parquet_entries_no_match_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub._datasets_parquet._fetch_json",
        lambda url, token: {
            "datasets": {"train": ["https://example.com/datasets-train.parquet"]},
        },
    )

    with pytest.raises(EntryNotFoundError, match="No parquet entries found"):
        list_dataset_parquet_entries(repo_id="cfahlgren1/hub-stats", token="token", config="models")
