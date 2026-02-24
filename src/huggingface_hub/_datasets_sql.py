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

from . import constants
from .utils import get_token_to_send


@dataclass(frozen=True)
class DatasetSqlQueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    table: str


def execute_raw_sql_query(
    sql_query: str,
    token: Union[str, bool, None],
    output_format: str = "json",
) -> DatasetSqlQueryResult:
    normalized_query = _normalize_query(sql_query)
    if output_format not in {"table", "json"}:
        raise ValueError(f"Unsupported SQL output format: {output_format!r}")

    effective_token = get_token_to_send(token)
    connection = _get_duckdb_connection(token=effective_token)
    try:
        relation = connection.sql(normalized_query)
        if relation is None or relation.description is None:
            raise ValueError("SQL query must return rows.")
        table = str(relation)
        columns = tuple(column[0] for column in relation.description)
        rows = tuple(tuple(row) for row in relation.fetchall()) if output_format == "json" else ()
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


def _get_duckdb_connection(token: Union[str, bool, None]):
    try:
        duckdb = importlib.import_module("duckdb")
    except ModuleNotFoundError as error:
        raise ImportError(
            "The 'duckdb' package is required for `hf datasets sql`. Install it with `pip install duckdb`."
        ) from error

    connection = duckdb.connect()
    if isinstance(token, str) and token:
        escaped_token = token.replace("'", "''")
        escaped_endpoint = constants.ENDPOINT.replace("'", "''")
        connection.execute(
            f"CREATE OR REPLACE SECRET hf_hub_token (TYPE HTTP, BEARER_TOKEN '{escaped_token}', "
            f"SCOPE '{escaped_endpoint}')"
        )
        connection.execute(f"CREATE OR REPLACE SECRET hf_token (TYPE HUGGINGFACE, TOKEN '{escaped_token}')")
    return connection
