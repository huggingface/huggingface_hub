import types

import pytest

from huggingface_hub.utils._duckdb import (
    DatasetSqlQueryResult,
    _build_duckdb_secret_statements,
    _DuckDBCliConnection,
    _get_duckdb_connection,
    _normalize_query,
    execute_raw_sql_query,
    format_sql_result,
)


def test_normalize_query_strips_whitespace_and_semicolons() -> None:
    assert _normalize_query("  SELECT 1;  ") == "SELECT 1"
    assert _normalize_query("SELECT 1;;") == "SELECT 1"


def test_normalize_query_empty_raises() -> None:
    with pytest.raises(ValueError, match="SQL query cannot be empty"):
        _normalize_query(" ; ")


def test_format_sql_result_table_returns_duckdb_table() -> None:
    result = DatasetSqlQueryResult(
        columns=("name",),
        rows=(("models",),),
        table="┌─────────┐\n│  name   │\n├─────────┤\n│ models  │\n└─────────┘",
    )
    output = format_sql_result(result=result, output_format="table")
    assert output == result.table


def test_format_sql_result_json_output() -> None:
    result = DatasetSqlQueryResult(columns=("subset", "rows"), rows=(("models", 10),), table="")
    payload = format_sql_result(result=result, output_format="json")
    assert payload == '[\n  {\n    "subset": "models",\n    "rows": 10\n  }\n]'


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
        "huggingface_hub.utils._duckdb._get_duckdb_connection",
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
        "huggingface_hub.utils._duckdb._get_duckdb_connection", lambda token, endpoint: fake_connection
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

    monkeypatch.setattr("huggingface_hub.utils._duckdb.importlib.import_module", import_module)
    monkeypatch.setattr("huggingface_hub.utils._duckdb.shutil.which", lambda _: None)
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

    monkeypatch.setattr("huggingface_hub.utils._duckdb.importlib.import_module", lambda name: fake_duckdb)

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

    monkeypatch.setattr("huggingface_hub.utils._duckdb.importlib.import_module", lambda name: fake_duckdb)

    _get_duckdb_connection(token=None)

    assert fake_connection.executed == []


def test_get_duckdb_connection_uses_cli_fallback_when_python_package_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def import_module(name):
        if name == "duckdb":
            raise ModuleNotFoundError("No module named 'duckdb'")
        return None

    monkeypatch.setattr("huggingface_hub.utils._duckdb.importlib.import_module", import_module)
    monkeypatch.setattr("huggingface_hub.utils._duckdb.shutil.which", lambda _: "/usr/local/bin/duckdb")

    connection = _get_duckdb_connection(token="abc")

    assert isinstance(connection, _DuckDBCliConnection)
    assert connection.binary_path == "/usr/local/bin/duckdb"
    relation = connection.sql("SELECT 1")
    assert relation.__class__.__name__ == "_DuckDBCliRelation"
    assert relation.setup_statements == _build_duckdb_secret_statements("abc")
