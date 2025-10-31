import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import huggingface_hub.utils._verification as verification_module
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils._verification import (
    HashAlgo,
    collect_local_files,
    compute_file_hash,
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


@pytest.mark.parametrize(
    "algorithm,data,expected_fn",
    [
        ("sha256", b"hello", lambda d: hashlib.sha256(d).hexdigest()),
        ("git-sha1", b"hello", lambda d: git_hash(d)),
    ],
)
def test_compute_file_hash_algorithms(tmp_path: Path, algorithm: HashAlgo, data: bytes, expected_fn) -> None:
    fp = tmp_path / "x.bin"
    _write(fp, data)

    actual = compute_file_hash(fp, algorithm)
    assert actual == expected_fn(data)


def test_compute_file_hash_git_sha1_computes_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fp = tmp_path / "x.txt"
    data = b"cached!"
    _write(fp, data)

    calls = {"count": 0}

    def fake_git_hash(b: bytes) -> str:
        calls["count"] += 1
        return git_hash(b)

    monkeypatch.setattr(verification_module, "git_hash", fake_git_hash, raising=False)

    h1 = compute_file_hash(fp, "git-sha1")
    h2 = compute_file_hash(fp, "git-sha1")

    assert h1 == h2 == git_hash(data)
    # Each call computes the hash independently (no cache)
    assert calls["count"] == 2


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


def test_verify_maps_success_local_dir(tmp_path: Path) -> None:
    # local
    loc = tmp_path / "loc"
    _write(loc / "a.txt", b"aa")
    _write(loc / "b.txt", b"bb")
    local_by_path = collect_local_files(loc)

    # remote entries (non-LFS for a.txt; LFS for b.txt)
    remote_by_path = {
        "a.txt": SimpleNamespace(path="a.txt", blob_id=git_hash(b"aa"), lfs=None),
        "b.txt": SimpleNamespace(
            path="b.txt",
            blob_id="unused",
            lfs={"sha256": hashlib.sha256(b"bb").hexdigest()},
        ),
    }
    res = verify_maps(
        remote_by_path=remote_by_path,
        local_by_path=local_by_path,
        revision="abc",
        verified_path=loc,
    )
    assert res.checked_count == 2
    assert res.mismatches == []
    assert res.missing_paths == []
    assert res.extra_paths == []
    assert res.verified_path == loc


def test_verify_maps_reports_mismatch(tmp_path: Path) -> None:
    loc = tmp_path / "loc2"
    loc.mkdir()
    _write(loc / "a.txt", b"wrong")
    local_by_path = collect_local_files(loc)
    remote_by_path = {"a.txt": SimpleNamespace(path="a.txt", blob_id=git_hash(b"right"), lfs=None)}
    res = verify_maps(
        remote_by_path=remote_by_path,
        local_by_path=local_by_path,
        revision="r",
        verified_path=loc,
    )
    assert len(res.mismatches) == 1
    m = res.mismatches[0]
    assert m["path"] == "a.txt" and m["algorithm"] == "git-sha1"
    assert res.verified_path == loc


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
