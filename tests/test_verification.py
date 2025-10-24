import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils._verification import (
    collect_local_files,
    compute_file_hash,
    resolve_expected_hash,
    resolve_local_root,
    verify_maps,
)
from huggingface_hub.utils.sha import git_hash


def _write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_collect_local_files_lists_all(tmp_path: Path) -> None:
    base = tmp_path
    (_ := base / "a" / "b.txt").parent.mkdir(parents=True, exist_ok=True)
    (base / "a" / "b.txt").write_text("x")
    (base / "c.bin").write_bytes(b"y")

    mapping = collect_local_files(base)
    assert mapping["a/b.txt"].read_text() == "x"
    assert mapping["c.bin"].read_bytes() == b"y"


def test_resolve_local_root_cache_single_snapshot(tmp_path: Path) -> None:
    cache_dir = tmp_path
    storage = cache_dir / "models--user--model"
    (storage / "blobs").mkdir(parents=True)
    commit = "a" * 40
    snapshot = storage / "snapshots" / commit
    snapshot.mkdir(parents=True)
    _write(snapshot / "config.json", b"{}")
    _write(snapshot / "nested" / "file.txt", b"hello")

    root, resolved_revision = resolve_local_root(
        repo_id="user/model", repo_type="model", revision=commit, cache_dir=cache_dir, local_dir=None
    )
    assert resolved_revision == commit
    mapping = collect_local_files(root)
    assert sorted(mapping.keys()) == ["config.json", "nested/file.txt"]


def test_resolve_expected_hash_prefers_lfs_sha256() -> None:
    entry = SimpleNamespace(path="x", blob_id="deadbeef", lfs={"sha256": "cafebabe"})
    algo, expected = resolve_expected_hash(entry)
    assert algo == "sha256" and expected == "cafebabe"


def test_compute_file_hash_git_sha1_stream(tmp_path: Path) -> None:
    data = b"content-xyz"
    p = tmp_path / "f.bin"
    _write(p, data)
    # expected git-sha1 (with header)
    expected = git_hash(data)
    actual = compute_file_hash(p, "git-sha1", git_hash_cache={})
    assert actual == expected


def test_verify_maps_success_local_dir(tmp_path: Path) -> None:
    # local
    loc = tmp_path / "loc"
    loc.mkdir()
    _write(loc / "a.txt", b"aa")
    _write(loc / "b.txt", b"bb")
    local_by_path = collect_local_files(loc)
    # remote entries (non-LFS for a.txt; LFS for b.txt)
    remote_by_path = {
        "a.txt": SimpleNamespace(path="a.txt", blob_id=git_hash(b"aa"), lfs=None),
        "b.txt": SimpleNamespace(path="b.txt", blob_id="unused", lfs={"sha256": hashlib.sha256(b"bb").hexdigest()}),
    }
    res = verify_maps(remote_by_path=remote_by_path, local_by_path=local_by_path, revision="abc")
    assert res.checked_count == 2 and not res.mismatches and not res.missing_paths and not res.extra_paths


def test_verify_maps_reports_mismatch(tmp_path: Path) -> None:
    loc = tmp_path / "loc2"
    loc.mkdir()
    _write(loc / "a.txt", b"wrong")
    local_by_path = collect_local_files(loc)
    remote_by_path = {"a.txt": SimpleNamespace(path="a.txt", blob_id=git_hash(b"right"), lfs=None)}
    res = verify_maps(remote_by_path=remote_by_path, local_by_path=local_by_path, revision="r")
    assert len(res.mismatches) == 1
    m = res.mismatches[0]
    assert m["path"] == "a.txt" and m["algorithm"] == "git-sha1"


def test_api_verify_repo_checksums_cache_mode(tmp_path: Path) -> None:
    # minimal dummy cache structure
    cache_dir = tmp_path
    commit = "b" * 40
    storage = cache_dir / "models--user--model"
    snapshot = storage / "snapshots" / commit
    snapshot.mkdir(parents=True)
    content = b"hello-world"
    _write(snapshot / "file.txt", content)

    with patch.object(
        HfApi,
        "list_repo_tree",
        return_value=[SimpleNamespace(path="file.txt", blob_id=git_hash(content), lfs=None)],
    ):
        res = HfApi().verify_repo_checksums(
            repo_id="user/model", repo_type="model", revision=commit, cache_dir=cache_dir, token=None
        )
        assert res.revision == commit and res.checked_count == 1 and not res.mismatches
