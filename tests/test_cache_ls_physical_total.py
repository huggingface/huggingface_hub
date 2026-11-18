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

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from huggingface_hub.cli.hf import app
from huggingface_hub.utils import CachedFileInfo, CachedRepoInfo, CachedRevisionInfo, HFCacheInfo
from huggingface_hub.utils._cache_manager import _format_size


def test_revision_summary_counts_shared_blobs_once() -> None:
    def file(name: str, revision: str, blob: str, size: int) -> CachedFileInfo:
        return CachedFileInfo(
            file_name=name,
            file_path=Path(f"/cache/snapshots/{revision}/{name}"),
            blob_path=Path(f"/cache/blobs/{blob}"),
            size_on_disk=size,
            blob_last_accessed=0.0,
            blob_last_modified=0.0,
        )

    shared_a = file("weights.bin", "a", "shared", 10)
    shared_b = file("weights.bin", "b", "shared", 10)
    first = CachedRevisionInfo(
        commit_hash="a" * 40,
        snapshot_path=Path("/cache/snapshots/a"),
        size_on_disk=12,
        files=frozenset({shared_a, file("README.md", "a", "readme-a", 2)}),
        refs=frozenset(),
        last_modified=0.0,
    )
    second = CachedRevisionInfo(
        commit_hash="b" * 40,
        snapshot_path=Path("/cache/snapshots/b"),
        size_on_disk=13,
        files=frozenset({shared_b, file("README.md", "b", "readme-b", 3)}),
        refs=frozenset({"main"}),
        last_modified=0.0,
    )
    repo = CachedRepoInfo(
        repo_id="user/model",
        repo_type="model",
        repo_path=Path("/cache"),
        size_on_disk=15,
        nb_files=3,
        revisions=frozenset({first, second}),
        last_accessed=0.0,
        last_modified=0.0,
    )
    cache = HFCacheInfo(size_on_disk=15, repos=frozenset({repo}), incomplete_files=frozenset(), warnings=[])
    entries = [(repo, first), (repo, second)]
    refs = {repo: frozenset({"main"})}

    with (
        patch("huggingface_hub.cli.cache.scan_cache_dir", return_value=cache),
        patch("huggingface_hub.cli.cache.collect_cache_entries", return_value=(entries, refs)),
    ):
        all_result = CliRunner().invoke(app, ["cache", "ls", "--revisions"])
        filtered_result = CliRunner().invoke(app, ["cache", "ls", "--revisions", "--filter", "type=model"])
        limited_result = CliRunner().invoke(app, ["cache", "ls", "--revisions", "--limit", "1"])

    for result in (all_result, filtered_result, limited_result):
        assert result.exit_code == 0, result.output
    assert f"2 revision(s) and {_format_size(15)} on disk" in all_result.output
    assert f"2 revision(s) and {_format_size(15)} on disk" in filtered_result.output
    assert f"1 revision(s) and {_format_size(12)} on disk" in limited_result.output
