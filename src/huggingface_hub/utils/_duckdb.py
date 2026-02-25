# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.
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


import importlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Optional, Union

from .. import constants
from ._headers import get_token_to_send


@dataclass(frozen=True)
class DatasetSqlQueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    table: str
    raw_json: Optional[str] = None


def execute_raw_sql_query(
    sql_query: str,
    token: Union[str, bool, None],
    output_format: str = "table",
    *,
    endpoint: str = constants.ENDPOINT,
) -> DatasetSqlQueryResult:
    normalized_query = _normalize_query(sql_query)
    if output_format not in {"table", "json"}:
        raise ValueError(f"Unsupported SQL output format: {output_format!r}")

    effective_token = get_token_to_send(token)
    connection = None
    try:
        connection = _get_duckdb_connection(token=effective_token, endpoint=endpoint)
        relation = connection.sql(normalized_query)
        if relation is None:
            raise ValueError("SQL query must return rows.")
        table: str
        columns: tuple[str, ...]
        rows: tuple[tuple[Any, ...], ...]
        if output_format == "table":
            table = str(relation)
            if not table:
                raise ValueError("SQL query must return rows.")
            columns = ()
            rows = ()
            raw_json = None
        else:
            to_json = getattr(relation, "to_json", None)
            if callable(to_json):
                return DatasetSqlQueryResult(columns=(), rows=(), table="", raw_json=to_json())
            if relation.description is None:
                raise ValueError("SQL query must return rows.")
            columns = tuple(column[0] for column in relation.description)
            table = ""
            rows = tuple(tuple(row) for row in relation.fetchall())
            raw_json = None
        return DatasetSqlQueryResult(columns=columns, rows=rows, table=table, raw_json=raw_json)
    except (ImportError, ValueError):
        raise
    except Exception as e:
        raise ValueError(str(e)) from e
    finally:
        if connection is not None:
            connection.close()


def format_sql_result(result: DatasetSqlQueryResult, output_format: str) -> str:
    if output_format == "table":
        return result.table
    if result.raw_json is not None:
        return result.raw_json
    return json.dumps(
        [{column: value for column, value in zip(result.columns, row)} for row in result.rows],
        indent=2,
        default=str,
    )


def _normalize_query(sql_query: str) -> str:
    normalized_query = sql_query.strip().rstrip(";").strip()
    if not normalized_query:
        raise ValueError("SQL query cannot be empty.")
    return normalized_query


def _get_duckdb_connection(token: Union[str, bool, None], endpoint: str = constants.ENDPOINT):
    try:
        duckdb = importlib.import_module("duckdb")
    except ModuleNotFoundError as error:
        duckdb_binary = shutil.which("duckdb")
        if duckdb_binary is None:
            raise ImportError(
                "DuckDB is required for `hf datasets sql`. Install the Python package with `pip install duckdb` or "
                "install the DuckDB CLI binary (for example `brew install duckdb`)."
            ) from error
        return _DuckDBCliConnection(binary_path=duckdb_binary, token=token, endpoint=endpoint)

    connection = duckdb.connect()
    try:
        for statement in _build_duckdb_secret_statements(token, endpoint=endpoint):
            connection.execute(statement)
        return connection
    except Exception:
        connection.close()
        raise


@dataclass
class _DuckDBCliConnection:
    binary_path: str
    token: Union[str, bool, None]
    endpoint: str

    def __post_init__(self) -> None:
        self._setup_statements = _build_duckdb_secret_statements(self.token, endpoint=self.endpoint)

    def sql(self, query: str) -> "_DuckDBCliRelation":
        return _DuckDBCliRelation(
            binary_path=self.binary_path,
            setup_statements=self._setup_statements,
            query=query,
        )

    def close(self) -> None:
        pass


@dataclass
class _DuckDBCliRelation:
    binary_path: str
    setup_statements: list[str]
    query: str

    def __post_init__(self) -> None:
        self._table: Optional[str] = None
        self._json: Optional[str] = None

    def to_json(self) -> str:
        if self._json is None:
            self._json = _run_duckdb_cli(
                binary_path=self.binary_path,
                setup_statements=self.setup_statements,
                query=self.query,
                output_mode="json",
            )
        return self._json

    def __str__(self) -> str:
        if self._table is None:
            self._table = _run_duckdb_cli(
                binary_path=self.binary_path,
                setup_statements=self.setup_statements,
                query=self.query,
                output_mode="table",
            )
        return self._table


def _build_duckdb_secret_statements(token: Union[str, bool, None], endpoint: str = constants.ENDPOINT) -> list[str]:
    if not isinstance(token, str) or not token:
        return []

    escaped_token = token.replace("'", "''")
    escaped_endpoint = endpoint.replace("'", "''")
    return [
        f"CREATE OR REPLACE SECRET hf_hub_token (TYPE HTTP, BEARER_TOKEN '{escaped_token}', SCOPE '{escaped_endpoint}')",
        f"CREATE OR REPLACE SECRET hf_token (TYPE HUGGINGFACE, TOKEN '{escaped_token}')",
    ]


def _run_duckdb_cli(binary_path: str, setup_statements: list[str], query: str, output_mode: str) -> str:
    command = [binary_path]
    if output_mode != "table":
        command.append(f"-{output_mode}")
    query_input = _build_duckdb_cli_input(setup_statements=setup_statements, query=query)
    result = subprocess.run(command, input=query_input, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        error_message = result.stderr.strip() or result.stdout.strip() or "DuckDB CLI command failed."
        raise ValueError(error_message)
    return result.stdout.strip()


def _build_duckdb_cli_input(setup_statements: list[str], query: str) -> str:
    statements: list[str] = []
    if setup_statements:
        statements.append(f".output {os.devnull}")
        statements.extend(f"{statement};" for statement in setup_statements)
        statements.append(".output stdout")
    statements.append(f"{query};")
    return "\n".join(statements)
