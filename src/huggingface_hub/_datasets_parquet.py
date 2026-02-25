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


from dataclasses import dataclass
from typing import Any, Optional, Union

from . import constants
from .errors import EntryNotFoundError
from .utils import build_hf_headers, get_session, hf_raise_for_status


@dataclass(frozen=True)
class DatasetParquetEntry:
    config: str
    split: str
    url: str


def list_dataset_parquet_entries(
    repo_id: str,
    token: Union[str, bool, None],
    config: Optional[str] = None,
    split: Optional[str] = None,
) -> list[DatasetParquetEntry]:
    payload = _fetch_json(url=_hub_parquet_url(repo_id=repo_id, endpoint=constants.ENDPOINT), token=token)

    entries: list[DatasetParquetEntry] = []
    if isinstance(payload, dict):
        for config_name, splits in payload.items():
            if not isinstance(splits, dict):
                continue
            if config is not None and config_name != config:
                continue
            for split_name, urls in splits.items():
                if not isinstance(urls, list):
                    continue
                if split is not None and split_name != split:
                    continue
                for url in urls:
                    if isinstance(url, str):
                        entries.append(DatasetParquetEntry(config=config_name, split=split_name, url=url))

    if not entries:
        raise EntryNotFoundError(
            f"No parquet entries found for dataset '{repo_id}' with filters config={config!r}, split={split!r}."
        )
    return entries


def _fetch_json(url: str, token: Union[str, bool, None]) -> Any:
    response = get_session().get(url, headers=build_hf_headers(token=token), timeout=constants.DEFAULT_REQUEST_TIMEOUT)
    hf_raise_for_status(response)
    return response.json()


def _hub_parquet_url(repo_id: str, endpoint: str) -> str:
    return f"{endpoint.rstrip('/')}/api/datasets/{repo_id}/parquet"
