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

from dataclasses import dataclass
from typing import Any, Optional, Union
from urllib.parse import quote, unquote, urlparse

from . import constants
from .errors import EntryNotFoundError
from .utils import build_hf_headers, get_session, hf_raise_for_status


@dataclass
class DatasetParquetStatus:
    partial: bool
    pending: tuple[str, ...]
    failed: tuple[str, ...]


@dataclass(frozen=True)
class DatasetParquetEntry:
    config: str
    split: str
    parquet_file_path: str


def list_dataset_parquet_entries(
    repo_id: str,
    token: Union[str, bool, None],
    config: Optional[str] = None,
    split: Optional[str] = None,
) -> list[DatasetParquetEntry]:
    root_url = _hub_parquet_url(repo_id=repo_id, endpoint=constants.ENDPOINT)
    root_payload = _fetch_json(url=root_url, token=token)

    parquet_by_config = _parse_hub_parquet_root_payload(root_payload)

    if not parquet_by_config and config is not None:
        config_url = f"{root_url}/{quote(config, safe='')}"
        config_payload = _fetch_json(url=config_url, token=token)
        parsed_config_payload = _parse_hub_parquet_config_payload(config_payload)
        if parsed_config_payload:
            parquet_by_config = {config: parsed_config_payload}

    entries = [
        DatasetParquetEntry(
            config=config_name,
            split=split_name,
            parquet_file_path=_build_parquet_file_path(
                repo_id=repo_id,
                config=config_name,
                split=split_name,
                parquet_file_name=parquet_file_name,
            ),
        )
        for config_name, splits in sorted(parquet_by_config.items())
        for split_name, parquet_file_names in sorted(splits.items())
        for parquet_file_name in sorted(parquet_file_names)
    ]

    # Root endpoint returns all configs; filter when user passed --subset.
    if config is not None:
        entries = [entry for entry in entries if entry.config == config]
    if split is not None:
        entries = [entry for entry in entries if entry.split == split]

    if not entries:
        raise EntryNotFoundError(
            f"No parquet entries found for dataset '{repo_id}' with filters config={config!r}, split={split!r}."
        )
    return entries


def fetch_dataset_parquet_status(repo_id: str, token: Union[str, bool, None]) -> DatasetParquetStatus:
    payload = _fetch_json(url=_hub_parquet_url(repo_id=repo_id, endpoint=constants.ENDPOINT), token=token)
    return DatasetParquetStatus(
        partial=bool(payload.get("partial")),
        pending=_normalize_status_entries(payload, "pending"),
        failed=_normalize_status_entries(payload, "failed"),
    )


def _fetch_json(url: str, token: Union[str, bool, None]) -> Any:
    response = get_session().get(url, headers=build_hf_headers(token=token), timeout=constants.DEFAULT_REQUEST_TIMEOUT)
    hf_raise_for_status(response)
    return response.json()


def _hub_parquet_url(repo_id: str, endpoint: str) -> str:
    return f"{endpoint.rstrip('/')}/api/datasets/{repo_id}/parquet"


def _parse_hub_parquet_root_payload(payload: Any) -> dict[str, dict[str, set[str]]]:
    parsed: dict[str, dict[str, set[str]]] = {}
    if not isinstance(payload, dict):
        return parsed

    parquet_files = payload.get("parquet_files")
    if isinstance(parquet_files, list):
        for item in parquet_files:
            if not isinstance(item, dict):
                continue
            config = item.get("config")
            split = item.get("split")
            parquet_file_names = _normalize_parquet_file_names(item.get("url"), item.get("urls"))
            if isinstance(config, str) and isinstance(split, str) and parquet_file_names:
                parsed.setdefault(config, {}).setdefault(split, set()).update(parquet_file_names)
        if parsed:
            return parsed

    for config_name, config_payload in payload.items():
        if config_name in {"partial", "pending", "failed", "parquet_files"}:
            continue
        parsed_config = _parse_hub_parquet_config_payload(config_payload)
        if parsed_config:
            parsed[config_name] = parsed_config
    return parsed


def _parse_hub_parquet_config_payload(payload: Any) -> dict[str, set[str]]:
    parsed: dict[str, set[str]] = {}
    if not isinstance(payload, dict):
        return parsed

    parquet_files = payload.get("parquet_files")
    if isinstance(parquet_files, list):
        for item in parquet_files:
            if not isinstance(item, dict):
                continue
            split = item.get("split")
            parquet_file_names = _normalize_parquet_file_names(item.get("url"), item.get("urls"))
            if isinstance(split, str) and parquet_file_names:
                parsed.setdefault(split, set()).update(parquet_file_names)
        if parsed:
            return parsed

    for split_name, urls in payload.items():
        if split_name in {"partial", "pending", "failed", "parquet_files"}:
            continue
        parquet_file_names = _normalize_parquet_file_names(urls)
        if parquet_file_names:
            parsed.setdefault(split_name, set()).update(parquet_file_names)
    return parsed


def _normalize_parquet_file_names(*values: Any) -> tuple[str, ...]:
    file_names: set[str] = set()
    for value in values:
        if isinstance(value, str):
            file_name = _extract_parquet_file_name(value)
            if file_name is not None:
                file_names.add(file_name)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    file_name = _extract_parquet_file_name(item)
                    if file_name is not None:
                        file_names.add(file_name)
    return tuple(sorted(file_names))


def _extract_parquet_file_name(url_or_name: str) -> Optional[str]:
    parsed_url = urlparse(url_or_name)
    candidate = url_or_name if parsed_url.scheme == "" else parsed_url.path.rsplit("/", 1)[-1]
    candidate = unquote(candidate)
    if candidate.endswith(".parquet"):
        return candidate
    return None


def _build_parquet_file_path(repo_id: str, config: str, split: str, parquet_file_name: str) -> str:
    return (
        f"hf://datasets/{repo_id}@~parquet/{quote(config, safe='')}/{quote(split, safe='')}/"
        f"{quote(parquet_file_name, safe='')}"
    )


def _normalize_status_entries(payload: Any, key: str) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return ()

    values = payload.get(key)
    items: list[str] = []

    if isinstance(values, str):
        items.append(values)
    elif isinstance(values, list):
        for value in values:
            if isinstance(value, str):
                items.append(value)
                continue
            if isinstance(value, dict):
                config = value.get("config")
                if isinstance(config, str):
                    items.append(config)

    return tuple(sorted(set(items)))
