import pytest

from huggingface_hub.utils._parquet import list_dataset_parquet_entries


def test_list_dataset_parquet_entries_from_root_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub.utils._parquet._fetch_json",
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
        ("models", "train"),
        ("models", "test"),
    ]
    assert entries[0].url == "https://example.com/datasets-train-0000.parquet"


def test_list_dataset_parquet_entries_returns_all_parquet_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub.utils._parquet._fetch_json",
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

    assert [entry.url for entry in entries] == [
        "https://example.com/datasets-train-0000.parquet",
        "https://example.com/datasets-train-0001.parquet",
    ]


def test_list_dataset_parquet_entries_filtered_by_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub.utils._parquet._fetch_json",
        lambda url, token: {
            "datasets": {"train": ["https://example.com/datasets-train.parquet"]},
            "models": {"train": ["https://example.com/models-train.parquet"]},
        },
    )

    entries = list_dataset_parquet_entries(repo_id="cfahlgren1/hub-stats", token="token", config="models")

    assert [(entry.config, entry.split) for entry in entries] == [("models", "train")]


def test_list_dataset_parquet_entries_filtered_by_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub.utils._parquet._fetch_json",
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
        "huggingface_hub.utils._parquet._fetch_json",
        lambda url, token: {
            "datasets": {"train": ["https://example.com/datasets-train.parquet"]},
        },
    )

    with pytest.raises(ValueError, match="No parquet entries found"):
        list_dataset_parquet_entries(repo_id="cfahlgren1/hub-stats", token="token", config="models")
