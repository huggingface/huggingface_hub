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
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

from . import constants
from .utils import get_token, logging


if TYPE_CHECKING:
    import duckdb


logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class DatasetParquetEntry:
    """Represents a single parquet file available for a dataset on the Hub."""

    config: str
    split: str
    url: str
    size: int


def execute_raw_sql_query(sql_query: str, *, token: str | bool | None = None) -> list[dict[str, Any]]:
    normalized_query = sql_query.strip().rstrip(";").strip()
    _raise_on_forbidden_query(normalized_query)

    connection = None
    try:
        connection = _get_duckdb_connection(token=token)
        relation = connection.sql(normalized_query)
        if relation is None:
            raise ValueError("SQL query must return rows.")

        if isinstance(relation, _DuckDBCliRelation):
            # DuckDB binary => run CLI => parse JSON
            return relation.execute()
        else:
            # DuckDB Python API => fetch columns + rows => convert to dicts
            columns = tuple(column[0] for column in relation.description)
            rows = tuple(tuple(row) for row in relation.fetchall())
            return [dict(zip(columns, row)) for row in rows]
    finally:
        if connection is not None:
            connection.close()


def _raise_on_forbidden_query(query: str) -> None:
    if len(query) == 0:
        raise ValueError("SQL query cannot be empty.")

    # DuckDB CLI meta-commands are dot-prefixed words (e.g. `.shell`, `.output`).
    # Let's forbid them for now but allow SQL expressions like `.5` that can legitimately start a line.
    for line in query.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(".") and stripped[1:2].isalpha():
            raise ValueError("DuckDB CLI meta-commands are not allowed in SQL queries.")


def _get_duckdb_connection(
    token: str | bool | None,
) -> Union["duckdb.DuckDBPyConnection", "_DuckDBCliConnection"]:
    try:
        # If DuckDB is installed as a Python package, use it!
        import duckdb
    except ImportError as error:
        # Otherwise, use the DuckDB CLI binary.
        duckdb_binary = shutil.which("duckdb")
        if duckdb_binary is None:
            raise ImportError(
                "DuckDB is required for `hf datasets sql`. Install the Python package with `pip install duckdb` or "
                "install the DuckDB CLI binary (for example `brew install duckdb`)."
            ) from error
        has_hf_extension = _install_hf_extension_cli(duckdb_binary)
        return _DuckDBCliConnection(binary_path=duckdb_binary, token=token, has_hf_extension=has_hf_extension)

    # Create a new connection (Python API).
    has_hf_extension = _install_hf_extension_python()
    connection = duckdb.connect()
    try:
        for statement in _build_duckdb_secret_statements(token):
            connection.execute(statement)
        if has_hf_extension:
            connection.execute("LOAD hf FROM hf_repo;")
        return connection
    except Exception:
        connection.close()
        raise


@dataclass
class _DuckDBCliConnection:
    """DuckDB connection.

    Mimics the DuckDB Python API, but runs the queries via the DuckDB CLI binary.
    """

    binary_path: str
    token: str | bool | None
    has_hf_extension: bool

    def __post_init__(self) -> None:
        self._setup_statements = _build_duckdb_secret_statements(self.token)
        if self.has_hf_extension:
            self._setup_statements += ["LOAD hf FROM hf_repo;"]

    def sql(self, query: str) -> "_DuckDBCliRelation":
        return _DuckDBCliRelation(binary_path=self.binary_path, setup_statements=self._setup_statements, query=query)

    def close(self) -> None:
        pass


@dataclass
class _DuckDBCliRelation:
    """DuckDB relation.

    Mimics the DuckDB Python API, but runs the queries via the DuckDB CLI binary.
    """

    binary_path: str
    setup_statements: list[str]
    query: str

    def execute(self) -> list[dict[str, Any]]:
        # Build the DuckDB CLI input.
        setup = []
        if self.setup_statements:
            setup = [
                f".output {os.devnull}",
                *(f"{stmt};" for stmt in self.setup_statements),
                ".output",
            ]
        full_query = "\n".join(setup + [self.query + ";"])

        # Run DuckDB binary
        result = subprocess.run(
            [self.binary_path, "-json"],
            input=full_query,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode != 0:
            error_message = result.stderr.strip() or result.stdout.strip() or "DuckDB CLI command failed."
            raise RuntimeError(error_message)

        # Parse JSON output and return
        return json.loads(result.stdout.strip())


def _install_hf_extension_cli(duckdb_binary: str) -> bool:
    result = subprocess.run(
        [
            duckdb_binary,
            "-c",
            """SET allow_extension_repositories = 'allowed'; CREATE OR REPLACE EXTENSION REPOSITORY hf_repo WITH PREFIX 'https://hf.co/buckets/duckdb-extensions/builds/resolve'; INSTALL hf FROM hf_repo;""",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        version = subprocess.run([duckdb_binary, "-version"], capture_output=True).stdout.decode().strip()
        platform = (
            subprocess.run([duckdb_binary, "-line", "-c", "PRAGMA platform;"], capture_output=True)
            .stdout.decode()
            .strip()
        )
        logger.warning(
            f"hf extension is not enabled on DuckDB {version} ({platform}), see https://hf.co/duckdb-extensions."
        )
        return False
    return True


def _install_hf_extension_python() -> bool:
    import duckdb

    try:
        with duckdb.connect(config={"allow_extension_repositories": "allowed"}) as con:
            con.execute(
                "CREATE OR REPLACE EXTENSION REPOSITORY hf_repo WITH PREFIX 'https://hf.co/buckets/duckdb-extensions/builds/resolve';"
            )
            con.execute("INSTALL hf FROM hf_repo;")
        return True
    except duckdb.Error:
        platform = (duckdb.execute("PRAGMA platform;").fetchone() or ("unknown platform",))[0]
        logger.warning(
            f"hf extension is not enabled on DuckDB {duckdb.__version__} ({platform}), see https://hf.co/duckdb-extensions."
        )
        return False


def _build_duckdb_secret_statements(token: str | bool | None) -> list[str]:
    if token is None or token is True:
        token = get_token()

    if not token:
        return []

    escaped_token = token.replace("'", "''")
    escaped_endpoint = constants.ENDPOINT.replace("'", "''")
    return [
        f"CREATE OR REPLACE SECRET hf_hub_token (TYPE HTTP, BEARER_TOKEN '{escaped_token}', SCOPE '{escaped_endpoint}')",
        f"CREATE OR REPLACE SECRET hf_token (TYPE HUGGINGFACE, TOKEN '{escaped_token}')",
    ]
