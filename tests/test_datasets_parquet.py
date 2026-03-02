import pytest

from huggingface_hub._parquet_dataset import list_dataset_parquet_files_internal


def _patch_parquet_api(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    """Patch get_session and hf_raise_for_status so list_dataset_parquet_files_internal returns *payload*."""

    class FakeResponse:
        status_code = 200
        headers = {}

        def json(self):
            return payload

        def raise_for_status(self):
            pass

    class FakeSession:
        def get(self, url, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("huggingface_hub._parquet_dataset.get_session", lambda: FakeSession())
    monkeypatch.setattr("huggingface_hub._parquet_dataset.hf_raise_for_status", lambda response: None)


def test_list_dataset_parquet_files_from_root_api(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_parquet_api(
        monkeypatch,
        {
            "datasets": {
                "train": ["https://example.com/datasets-train-0000.parquet"],
            },
            "models": {
                "train": ["https://example.com/models-train-0000.parquet"],
                "test": ["https://example.com/models-test-0000.parquet"],
            },
        },
    )

    entries = list_dataset_parquet_files_internal(repo_id="cfahlgren1/hub-stats", token="token")

    assert [(entry.config, entry.split) for entry in entries] == [
        ("datasets", "train"),
        ("models", "train"),
        ("models", "test"),
    ]
    assert entries[0].url == "https://example.com/datasets-train-0000.parquet"


def test_list_dataset_parquet_files_returns_all_parquet_files(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_parquet_api(
        monkeypatch,
        {
            "datasets": {
                "train": [
                    "https://example.com/datasets-train-0000.parquet",
                    "https://example.com/datasets-train-0001.parquet",
                ],
            },
        },
    )

    entries = list_dataset_parquet_files_internal(
        repo_id="cfahlgren1/hub-stats", token="token", config="datasets", split="train"
    )

    assert [entry.url for entry in entries] == [
        "https://example.com/datasets-train-0000.parquet",
        "https://example.com/datasets-train-0001.parquet",
    ]


def test_list_dataset_parquet_files_filtered_by_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_parquet_api(
        monkeypatch,
        {
            "datasets": {"train": ["https://example.com/datasets-train.parquet"]},
            "models": {"train": ["https://example.com/models-train.parquet"]},
        },
    )

    entries = list_dataset_parquet_files_internal(repo_id="cfahlgren1/hub-stats", token="token", config="models")

    assert [(entry.config, entry.split) for entry in entries] == [("models", "train")]


def test_list_dataset_parquet_files_filtered_by_split(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_parquet_api(
        monkeypatch,
        {
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

    entries = list_dataset_parquet_files_internal(repo_id="cfahlgren1/hub-stats", token="token", split="train")

    assert [(entry.config, entry.split) for entry in entries] == [
        ("datasets", "train"),
        ("models", "train"),
    ]


def test_list_dataset_parquet_files_no_match_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_parquet_api(
        monkeypatch,
        {"datasets": {"train": ["https://example.com/datasets-train.parquet"]}},
    )

    with pytest.raises(ValueError, match="No parquet entries found"):
        list_dataset_parquet_files_internal(repo_id="cfahlgren1/hub-stats", token="token", config="models")


def test_list_dataset_parquet_files_no_match_with_split_filter_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_parquet_api(
        monkeypatch,
        {"datasets": {"train": ["https://example.com/datasets-train.parquet"]}},
    )

    with pytest.raises(
        ValueError, match="No parquet entries found for dataset 'cfahlgren1/hub-stats' with split='test'."
    ):
        list_dataset_parquet_files_internal(repo_id="cfahlgren1/hub-stats", token="token", split="test")


def test_list_dataset_parquet_files_no_match_with_empty_config_filter_includes_filter_in_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_parquet_api(
        monkeypatch,
        {"datasets": {"train": ["https://example.com/datasets-train.parquet"]}},
    )

    with pytest.raises(
        ValueError, match="No parquet entries found for dataset 'cfahlgren1/hub-stats' with config=''."
    ):
        list_dataset_parquet_files_internal(repo_id="cfahlgren1/hub-stats", token="token", config="")
