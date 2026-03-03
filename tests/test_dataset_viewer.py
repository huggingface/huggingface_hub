import types

import pytest

from huggingface_hub._dataset_viewer import (
    DatasetParquetEntry,
    _build_duckdb_cli_input,
    _build_duckdb_secret_statements,
    _DuckDBCliConnection,
    _get_duckdb_connection,
    _normalize_query,
    execute_raw_sql_query,
)
from huggingface_hub.constants import _HF_DEFAULT_ENDPOINT
from huggingface_hub.hf_api import HfApi


# ---------------------------------------------------------------------------
# Parquet tests
# ---------------------------------------------------------------------------


def _patch_datasets_server(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    """Patch get_session so that HfApi.list_dataset_parquet_files returns *payload*."""

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

    monkeypatch.setattr("huggingface_hub.hf_api.get_session", lambda: FakeSession())


def _make_api() -> HfApi:
    """Create an HfApi with the default production endpoint (tests run in staging mode)."""
    return HfApi(endpoint=_HF_DEFAULT_ENDPOINT)


def test_list_dataset_parquet_files_from_datasets_server(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_datasets_server(
        monkeypatch,
        {
            "parquet_files": [
                {
                    "dataset": "cfahlgren1/hub-stats",
                    "config": "datasets",
                    "split": "train",
                    "url": "https://example.com/datasets-train-0000.parquet",
                    "filename": "0000.parquet",
                    "size": 1234,
                },
                {
                    "dataset": "cfahlgren1/hub-stats",
                    "config": "models",
                    "split": "train",
                    "url": "https://example.com/models-train-0000.parquet",
                    "filename": "0000.parquet",
                    "size": 5678,
                },
                {
                    "dataset": "cfahlgren1/hub-stats",
                    "config": "models",
                    "split": "test",
                    "url": "https://example.com/models-test-0000.parquet",
                    "filename": "0000.parquet",
                    "size": 9012,
                },
            ],
        },
    )

    entries = _make_api().list_dataset_parquet_files(repo_id="cfahlgren1/hub-stats")

    assert [(entry.config, entry.split) for entry in entries] == [
        ("datasets", "train"),
        ("models", "train"),
        ("models", "test"),
    ]
    assert entries[0].url == "https://example.com/datasets-train-0000.parquet"
    assert entries[0].size == 1234


def test_list_dataset_parquet_files_returns_all_parquet_files(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_datasets_server(
        monkeypatch,
        {
            "parquet_files": [
                {
                    "dataset": "cfahlgren1/hub-stats",
                    "config": "datasets",
                    "split": "train",
                    "url": "https://example.com/datasets-train-0000.parquet",
                    "filename": "0000.parquet",
                    "size": 1000,
                },
                {
                    "dataset": "cfahlgren1/hub-stats",
                    "config": "datasets",
                    "split": "train",
                    "url": "https://example.com/datasets-train-0001.parquet",
                    "filename": "0001.parquet",
                    "size": 2000,
                },
            ],
        },
    )

    entries = _make_api().list_dataset_parquet_files(repo_id="cfahlgren1/hub-stats")

    assert [entry.url for entry in entries] == [
        "https://example.com/datasets-train-0000.parquet",
        "https://example.com/datasets-train-0001.parquet",
    ]


def test_list_dataset_parquet_files_includes_size(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_datasets_server(
        monkeypatch,
        {
            "parquet_files": [
                {
                    "dataset": "d",
                    "config": "default",
                    "split": "train",
                    "url": "https://example.com/0.parquet",
                    "filename": "0.parquet",
                    "size": 4415,
                },
            ],
        },
    )

    entries = _make_api().list_dataset_parquet_files(repo_id="d")

    assert len(entries) == 1
    assert entries[0].size == 4415


def test_list_dataset_parquet_files_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_datasets_server(monkeypatch, {"parquet_files": []})

    entries = _make_api().list_dataset_parquet_files(repo_id="empty/dataset")

    assert entries == []


def test_list_dataset_parquet_files_raises_on_non_default_endpoint() -> None:
    api = HfApi(endpoint="https://custom-hub.example.com")
    with pytest.raises(ValueError, match="only available on the Hugging Face Hub"):
        api.list_dataset_parquet_files(repo_id="some/dataset")


def test_dataset_parquet_entry_is_frozen() -> None:
    entry = DatasetParquetEntry(config="default", split="train", url="https://example.com/0.parquet", size=100)
    with pytest.raises(AttributeError):
        entry.config = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SQL / DuckDB tests
# ---------------------------------------------------------------------------


def test_normalize_query_strips_whitespace_and_semicolons() -> None:
    assert _normalize_query("  SELECT 1;  ") == "SELECT 1"
    assert _normalize_query("SELECT 1;;") == "SELECT 1"


def test_normalize_query_empty_raises() -> None:
    with pytest.raises(ValueError, match="SQL query cannot be empty"):
        _normalize_query(" ; ")


def test_build_duckdb_cli_input_rejects_cli_meta_commands() -> None:
    with pytest.raises(ValueError, match="meta-commands are not allowed"):
        _build_duckdb_cli_input(setup_statements=[], query="SELECT 1;\n.shell id")


def test_build_duckdb_cli_input_allows_decimal_literal_at_line_start() -> None:
    query_input = _build_duckdb_cli_input(setup_statements=[], query="SELECT\n  .5 AS x")
    assert query_input == "SELECT\n  .5 AS x;"


def test_execute_raw_sql_query_runs_normalized_query(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRelation:
        description = [("count",)]

        def __str__(self):
            return "┌───────┐\n│ count │\n├───────┤\n│   123 │\n└───────┘"

        def fetchall(self):
            return [(123,)]

    class FakeConnection:
        def __init__(self):
            self.executed_queries = []
            self.closed = False

        def sql(self, query):
            self.executed_queries.append(query)
            return FakeRelation()

        def close(self):
            self.closed = True

    fake_connection = FakeConnection()
    monkeypatch.setattr(
        "huggingface_hub._dataset_viewer._get_duckdb_connection",
        lambda token, endpoint: fake_connection,
    )

    result = execute_raw_sql_query(
        sql_query="SELECT COUNT(*) AS count;",
        token="token",
        output_format="json",
    )

    assert result.columns == ("count",)
    assert result.rows == ((123,),)
    assert result.table == ""
    assert "SELECT COUNT(*) AS count" in fake_connection.executed_queries
    assert fake_connection.closed is True


def test_execute_raw_sql_query_table_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRelation:
        def __str__(self):
            return "┌───┐\n│ a │\n├───┤\n│ 1 │\n└───┘"

        def fetchall(self):
            raise AssertionError("fetchall should not be called for table output")

    class FakeConnection:
        def __init__(self):
            self.executed_queries = []
            self.closed = False

        def sql(self, query):
            self.executed_queries.append(query)
            return FakeRelation()

        def close(self):
            self.closed = True

    fake_connection = FakeConnection()
    monkeypatch.setattr(
        "huggingface_hub._dataset_viewer._get_duckdb_connection", lambda token, endpoint: fake_connection
    )

    result = execute_raw_sql_query(sql_query=" SELECT 1; ", token="token", output_format="table")

    assert result.columns == ()
    assert result.rows == ()
    assert result.table == "┌───┐\n│ a │\n├───┤\n│ 1 │\n└───┘"
    assert result.raw_json is None
    assert fake_connection.executed_queries == ["SELECT 1"]
    assert fake_connection.closed is True


def test_get_duckdb_connection_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def import_module(name):
        if name == "duckdb":
            raise ModuleNotFoundError("No module named 'duckdb'")
        return None

    monkeypatch.setattr("huggingface_hub._dataset_viewer.importlib.import_module", import_module)
    monkeypatch.setattr("huggingface_hub._dataset_viewer.shutil.which", lambda _: None)
    with pytest.raises(ImportError, match="pip install duckdb") as exc_info:
        _get_duckdb_connection(token=None)
    assert "brew install duckdb" in str(exc_info.value)


def test_get_duckdb_connection_creates_huggingface_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeConnection:
        def __init__(self):
            self.executed = []

        def execute(self, statement):
            self.executed.append(statement)

    fake_connection = FakeConnection()
    fake_duckdb = types.SimpleNamespace(connect=lambda: fake_connection)

    monkeypatch.setattr("huggingface_hub._dataset_viewer.importlib.import_module", lambda name: fake_duckdb)

    connection = _get_duckdb_connection(token="abc")

    assert connection is fake_connection
    assert any("BEARER_TOKEN 'abc'" in s for s in fake_connection.executed)
    assert any("TYPE HUGGINGFACE" in s for s in fake_connection.executed)


def test_get_duckdb_connection_skips_secrets_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeConnection:
        def __init__(self):
            self.executed = []

        def execute(self, statement):
            self.executed.append(statement)

    fake_connection = FakeConnection()
    fake_duckdb = types.SimpleNamespace(connect=lambda: fake_connection)

    monkeypatch.setattr("huggingface_hub._dataset_viewer.importlib.import_module", lambda name: fake_duckdb)

    _get_duckdb_connection(token=None)

    assert fake_connection.executed == []


def test_get_duckdb_connection_uses_cli_fallback_when_python_package_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def import_module(name):
        if name == "duckdb":
            raise ModuleNotFoundError("No module named 'duckdb'")
        return None

    monkeypatch.setattr("huggingface_hub._dataset_viewer.importlib.import_module", import_module)
    monkeypatch.setattr("huggingface_hub._dataset_viewer.shutil.which", lambda _: "/usr/local/bin/duckdb")

    connection = _get_duckdb_connection(token="abc")

    assert isinstance(connection, _DuckDBCliConnection)
    assert connection.binary_path == "/usr/local/bin/duckdb"
    relation = connection.sql("SELECT 1")
    assert relation.__class__.__name__ == "_DuckDBCliRelation"
    assert relation.setup_statements == _build_duckdb_secret_statements("abc")
