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

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Any, Union

from .hf_file_system import HfFileSystem


@dataclass(frozen=True)
class DatasetSqlQueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    table: str


def execute_raw_sql_query(
    sql_query: str,
    token: Union[str, bool, None],
) -> DatasetSqlQueryResult:
    normalized_query = _normalize_query(sql_query)

    connection = _get_duckdb_connection(token=token)
    try:
        relation = connection.sql(normalized_query)
        if relation.description is None:
            raise ValueError("SQL query must return rows.")
        table = str(relation)
        columns = tuple(column[0] for column in relation.description)
        rows = tuple(tuple(row) for row in relation.fetchall())
        return DatasetSqlQueryResult(columns=columns, rows=rows, table=table)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(str(e)) from e
    finally:
        connection.close()


def format_sql_result(result: DatasetSqlQueryResult, output_format: str) -> str:
    if output_format == "table":
        return result.table
    if output_format == "json":
        return json.dumps(
            [{column: value for column, value in zip(result.columns, row)} for row in result.rows],
            indent=2,
            default=str,
        )
    raise ValueError(f"Unsupported SQL output format: {output_format!r}")


def _normalize_query(sql_query: str) -> str:
    normalized_query = sql_query.strip()
    while normalized_query.endswith(";"):
        normalized_query = normalized_query[:-1].rstrip()
    if not normalized_query:
        raise ValueError("SQL query cannot be empty.")
    return normalized_query


def _get_duckdb_connection(token: Union[str, bool, None]):
    try:
        duckdb = importlib.import_module("duckdb")
    except ModuleNotFoundError as error:
        raise ImportError(
            "The 'duckdb' package is required for `hf datasets sql`. Install it with `pip install duckdb`."
        ) from error

    connection = duckdb.connect()
    connection.register_filesystem(HfFileSystem(token=token))
    return connection
