"""Regression tests for https://github.com/huggingface/huggingface_hub/issues/4768.

`hf download` (i.e. `snapshot_download` / `hf_hub_download`) could report success while returning a
snapshot whose files were empty or truncated. Two failure modes:

- Case A - a *new* download whose reconstruction/stream produced fewer bytes than expected was promoted
  into the cache without any size check (Xet path only; the HTTP path already had one).
- Case B - a cache *already poisoned* by an older buggy run was trusted forever: the commit-hash
  shortcut in `_hf_hub_download_to_cache_dir` returned the cached file as soon as it existed, never
  comparing its size to the expected size.

The invariant under test: a successful return implies every requested file exists **and** matches the
size the Hub reports for that commit - while a legitimate zero-byte file stays valid.
"""

import os
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub._local_folder import get_local_download_paths, write_download_metadata
from huggingface_hub._snapshot_download import _raise_if_snapshot_incomplete
from huggingface_hub._tree_cache import TreeCacheEntry, tree_cache_folder_for_local_dir, write_tree_cache
from huggingface_hub.errors import IncompleteSnapshotError
from huggingface_hub.file_download import HfFileMetadata, _create_symlink, _get_pointer_path, repo_folder_name
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import XetFileData
from huggingface_hub.utils._xet import XetConnectionInfo


COMMIT = "1111111111111111111111111111111111111111"
ETAG = "e" * 64  # sha256-shaped etag => treated as an LFS/Xet blob


def _storage(cache_dir, repo_id="user/repo"):
    return Path(cache_dir) / repo_folder_name(repo_id=repo_id, repo_type="model")


def _seed_cache_entry(cache_dir, filename, *, blob_bytes, expected_size, etag=ETAG, xet=False, write_tree=True):
    """Recreate on disk exactly what a previous (possibly interrupted) download would have left."""
    storage = _storage(cache_dir)
    (storage / "blobs").mkdir(parents=True, exist_ok=True)
    (storage / "refs").mkdir(parents=True, exist_ok=True)
    blob_path = storage / "blobs" / etag
    blob_path.write_bytes(blob_bytes)

    pointer = Path(_get_pointer_path(str(storage), COMMIT, filename))
    pointer.parent.mkdir(parents=True, exist_ok=True)
    _create_symlink(str(blob_path), str(pointer), new_blob=False)

    if write_tree:
        write_tree_cache(
            str(storage),
            COMMIT,
            {
                filename: TreeCacheEntry(
                    size=expected_size,
                    blob_id="blob-" + filename,
                    lfs_sha256=etag if xet else None,
                    lfs_size=expected_size if xet else None,
                    xet_hash=("c" * 64) if xet else None,
                )
            },
        )
    return storage, blob_path, pointer


def _http_metadata(size, etag=ETAG):
    return HfFileMetadata(
        commit_hash=COMMIT,
        etag=etag,
        location=f"https://hub.example/resolve/{COMMIT}/file",
        size=size,
        xet_file_data=None,
    )


def _fake_http_get(url, temp_file, *, expected_size=None, **kwargs):
    """Stand-in for `http_get` that writes exactly `expected_size` bytes (a correct download)."""
    temp_file.write(b"\0" * (expected_size or 0))


def _fake_http_get_writing(nbytes):
    def _inner(url, temp_file, *, expected_size=None, **kwargs):
        temp_file.write(b"\0" * nbytes)

    return _inner


def _fake_xet_session(nbytes):
    """A mocked hf_xet session whose reconstruction writes `nbytes` to the path it is given."""
    group = Mock()

    def _write(xet_file_info, path, *a, **k):
        Path(path).write_bytes(b"\0" * nbytes)

    group.__enter__ = Mock(return_value=group)
    group.__exit__ = Mock(return_value=False)
    group.start_download_file.side_effect = _write
    session = Mock()
    session.new_file_download_group.return_value = group
    conn = XetConnectionInfo(access_token="t", expiration_unix_epoch=9999999999, endpoint="https://cas")
    return patch.multiple(
        "huggingface_hub.utils._xet",
        get_xet_session=Mock(return_value=session),
        refresh_xet_connection_info=Mock(return_value=conn),
    )


# --------------------------------------------------------------------------------------------------
# Case B - poisoned cache is not trusted (commit-hash shortcut + post-metadata shortcuts)
# --------------------------------------------------------------------------------------------------


class TestPoisonedCacheRecovery:
    def test_complete_cache_is_returned_without_any_download(self, tmp_path):
        """Test 1: expected == actual => fast path, no network."""
        _seed_cache_entry(tmp_path, "model.bin", blob_bytes=b"\0" * 100, expected_size=100)
        with (
            patch("huggingface_hub.file_download.http_get") as http_get,
            patch("huggingface_hub.file_download.get_hf_file_metadata") as meta,
        ):
            path = hf_hub_download("user/repo", "model.bin", revision=COMMIT, cache_dir=tmp_path)
        assert os.path.getsize(path) == 100
        http_get.assert_not_called()
        meta.assert_not_called()  # commit-hash shortcut skipped the HEAD call entirely

    def test_poisoned_blob_repaired_through_commit_hash_shortcut(self, tmp_path):
        """Test 2 + 6: 400 cached bytes vs 1000 expected (tree cache present) => must re-download."""
        _, blob_path, pointer = _seed_cache_entry(tmp_path, "model.bin", blob_bytes=b"\0" * 400, expected_size=1000)
        assert os.path.getsize(pointer) == 400  # poisoned

        with (
            patch("huggingface_hub.file_download.get_hf_file_metadata", return_value=_http_metadata(1000)),
            patch("huggingface_hub.file_download.http_get", side_effect=_fake_http_get) as http_get,
        ):
            path = hf_hub_download("user/repo", "model.bin", revision=COMMIT, cache_dir=tmp_path)

        http_get.assert_called_once()
        assert os.path.getsize(path) == 1000
        assert os.path.getsize(blob_path) == 1000  # blob repaired in place

    def test_poisoned_blob_repaired_through_post_metadata_shortcut(self, tmp_path):
        """Branch revision (commit-hash shortcut not taken) => the post-metadata size check must catch it."""
        storage, blob_path, _ = _seed_cache_entry(
            tmp_path, "model.bin", blob_bytes=b"\0" * 400, expected_size=1000, write_tree=False
        )
        (storage / "refs" / "main").write_text(COMMIT)

        with (
            patch("huggingface_hub.file_download.get_hf_file_metadata", return_value=_http_metadata(1000)),
            patch("huggingface_hub.file_download.http_get", side_effect=_fake_http_get) as http_get,
        ):
            path = hf_hub_download("user/repo", "model.bin", revision="main", cache_dir=tmp_path)

        http_get.assert_called_once()
        assert os.path.getsize(path) == 1000

    def test_zero_byte_file_is_valid(self, tmp_path):
        """Test 3: expected == 0 and actual == 0 => valid, no re-download, no error."""
        _seed_cache_entry(tmp_path, "empty.txt", blob_bytes=b"", expected_size=0)
        with (
            patch("huggingface_hub.file_download.http_get") as http_get,
            patch("huggingface_hub.file_download.get_hf_file_metadata", return_value=_http_metadata(0)),
        ):
            path = hf_hub_download("user/repo", "empty.txt", revision=COMMIT, cache_dir=tmp_path)
        assert os.path.getsize(path) == 0
        http_get.assert_not_called()

    def test_unknown_local_size_preserves_commit_hash_shortcut(self, tmp_path):
        """Test 4: no tree listing => size unknown locally => immutable-commit shortcut is preserved.

        The shortcut must NOT start making a HEAD call for every already-cached file, and must not
        reject the cached file. (`force_download=True` remains the escape hatch for this narrow case.)
        """
        _seed_cache_entry(tmp_path, "model.bin", blob_bytes=b"\0" * 400, expected_size=1000, write_tree=False)
        with (
            patch("huggingface_hub.file_download.http_get") as http_get,
            patch("huggingface_hub.file_download.get_hf_file_metadata") as meta,
        ):
            path = hf_hub_download("user/repo", "model.bin", revision=COMMIT, cache_dir=tmp_path)
        assert os.path.getsize(path) == 400
        http_get.assert_not_called()
        meta.assert_not_called()

    def test_interrupted_download_is_not_promoted_then_retry_succeeds(self, tmp_path):
        """Test 5: a truncated stream must not become a cache entry; a later run completes cleanly."""
        storage = _storage(tmp_path)

        with patch("huggingface_hub.file_download.get_hf_file_metadata", return_value=_http_metadata(1000)):
            # 1st attempt: server delivers only 400 bytes => http_get's own consistency check raises.
            with patch("huggingface_hub.file_download.http_get", side_effect=_fake_http_get_writing(400)):
                with pytest.raises(EnvironmentError):
                    hf_hub_download("user/repo", "model.bin", revision=COMMIT, cache_dir=tmp_path)

            # Nothing partial was promoted.
            assert list((storage / "blobs").glob("*")) == [] if (storage / "blobs").exists() else True

            # 2nd attempt: full download succeeds.
            with patch("huggingface_hub.file_download.http_get", side_effect=_fake_http_get):
                path = hf_hub_download("user/repo", "model.bin", revision=COMMIT, cache_dir=tmp_path)
        assert os.path.getsize(path) == 1000


# --------------------------------------------------------------------------------------------------
# Xet path (Case A: new truncated reconstruction ; Case B: poisoned reconstruction)
# --------------------------------------------------------------------------------------------------


@pytest.mark.xet
class TestPoisonedXetCache:
    def _xet_metadata_patch(self, size):
        return patch(
            "huggingface_hub.file_download.get_hf_file_metadata",
            return_value=HfFileMetadata(
                commit_hash=COMMIT,
                etag=ETAG,
                location=f"https://hub.example/resolve/{COMMIT}/file",
                size=size,
                xet_file_data=XetFileData(file_hash="c" * 64, refresh_route="r"),
            ),
        )

    def test_new_truncated_xet_reconstruction_raises(self, tmp_path):
        """Test 8a: reconstruction writes fewer bytes than expected, no exception => must raise."""
        storage = _storage(tmp_path)
        with (
            self._xet_metadata_patch(1000),
            patch("huggingface_hub.file_download.is_xet_available", return_value=True),
            _fake_xet_session(400),
        ):
            with pytest.raises(EnvironmentError, match="Consistency check failed"):
                hf_hub_download("user/repo", "model.bin", revision=COMMIT, cache_dir=tmp_path)
        assert (
            [p for p in (storage / "blobs").glob("*") if p.is_file()] == [] if (storage / "blobs").exists() else True
        )

    def test_correct_xet_reconstruction_succeeds(self, tmp_path):
        """Test 8b: reconstruction writes the expected size => success."""
        with (
            self._xet_metadata_patch(1000),
            patch("huggingface_hub.file_download.is_xet_available", return_value=True),
            _fake_xet_session(1000),
        ):
            path = hf_hub_download("user/repo", "model.bin", revision=COMMIT, cache_dir=tmp_path)
        assert os.path.getsize(path) == 1000

    def test_poisoned_xet_blob_is_repaired(self, tmp_path):
        """Case B for Xet: a 400-byte cached blob (tree cache present) is re-reconstructed to 1000."""
        _, blob_path, pointer = _seed_cache_entry(
            tmp_path, "model.bin", blob_bytes=b"\0" * 400, expected_size=1000, xet=True
        )
        assert os.path.getsize(pointer) == 400
        with patch("huggingface_hub.file_download.is_xet_available", return_value=True), _fake_xet_session(1000):
            path = hf_hub_download("user/repo", "model.bin", revision=COMMIT, cache_dir=tmp_path)
        assert os.path.getsize(path) == 1000
        assert os.path.getsize(blob_path) == 1000


# --------------------------------------------------------------------------------------------------
# snapshot_download level
# --------------------------------------------------------------------------------------------------


class TestSnapshotLevelCompleteness:
    def _tree(self, sizes: dict):
        return [
            RepoFile(
                path=name, size=size, oid="blob-" + name, lfs={"oid": "s-" + name, "size": size, "pointerSize": 1}
            )
            for name, size in sizes.items()
        ]

    def test_snapshot_repairs_one_poisoned_file_among_many(self, tmp_path):
        """Test 7 (recovery): file B truncated in cache => snapshot_download repairs it and succeeds."""
        sizes = {"a.bin": 10, "b.bin": 20, "c.bin": 30}
        # Let `snapshot_download` build the tree listing itself from `list_repo_tree` (write_tree=False).
        _seed_cache_entry(tmp_path, "a.bin", blob_bytes=b"\0" * 10, expected_size=10, etag="a" * 64, write_tree=False)
        _seed_cache_entry(tmp_path, "b.bin", blob_bytes=b"\0" * 4, expected_size=20, etag="b" * 64, write_tree=False)
        _seed_cache_entry(tmp_path, "c.bin", blob_bytes=b"\0" * 30, expected_size=30, etag="c" * 64, write_tree=False)

        def per_file_meta(url, **kw):
            name = str(url).rsplit("/", 1)[-1]
            return _http_metadata(sizes[name], etag=name[0] * 64)

        with (
            patch("huggingface_hub._snapshot_download.HfApi.repo_info", return_value=Mock(sha=COMMIT)),
            patch("huggingface_hub._snapshot_download.HfApi.list_repo_tree", return_value=self._tree(sizes)),
            patch("huggingface_hub.file_download.get_hf_file_metadata", side_effect=per_file_meta),
            patch("huggingface_hub.file_download.http_get", side_effect=_fake_http_get) as http_get,
        ):
            folder = snapshot_download("user/repo", cache_dir=tmp_path)

        for name, size in sizes.items():
            assert os.path.getsize(os.path.join(folder, name)) == size
        http_get.assert_called_once()  # only the poisoned file was re-downloaded

    def test_snapshot_raises_if_a_file_is_left_incomplete(self, tmp_path):
        """Test 10: the snapshot-level safety net turns a residual gap into an explicit error.

        Here `hf_hub_download` is stubbed to return a path to a short file (simulating any residual gap
        the per-file logic might miss); `snapshot_download` must not report success.
        """
        storage = _storage(tmp_path)
        snap = storage / "snapshots" / COMMIT
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "a.bin").write_bytes(b"\0" * 10)
        (snap / "b.bin").write_bytes(b"\0" * 5)  # short vs expected 20

        def fake_download(repo_id, *, filename, **kw):
            return str(snap / filename)

        with (
            patch("huggingface_hub._snapshot_download.HfApi.repo_info", return_value=Mock(sha=COMMIT)),
            patch(
                "huggingface_hub._snapshot_download.HfApi.list_repo_tree",
                return_value=self._tree({"a.bin": 10, "b.bin": 20}),
            ),
            patch("huggingface_hub._snapshot_download.hf_hub_download", side_effect=fake_download),
        ):
            with pytest.raises(IncompleteSnapshotError, match="incomplete after download"):
                snapshot_download("user/repo", cache_dir=tmp_path)

    def test_worker_exception_propagates_through_snapshot_download(self, tmp_path):
        """Test 11 / Q9: a per-file failure must not be swallowed by `hf_thread_map`."""

        def boom(repo_id, *, filename, **kw):
            if filename == "b.bin":
                raise RuntimeError("download failed for b.bin")
            return str(tmp_path / filename)

        (tmp_path / "a.bin").write_bytes(b"")
        (tmp_path / "c.bin").write_bytes(b"")
        with (
            patch("huggingface_hub._snapshot_download.HfApi.repo_info", return_value=Mock(sha=COMMIT)),
            patch(
                "huggingface_hub._snapshot_download.HfApi.list_repo_tree",
                return_value=self._tree({"a.bin": 0, "b.bin": 0, "c.bin": 0}),
            ),
            patch("huggingface_hub._snapshot_download.hf_hub_download", side_effect=boom),
        ):
            with pytest.raises(RuntimeError, match="download failed for b.bin"):
                snapshot_download("user/repo", cache_dir=tmp_path)


class TestLocalDirPoisonedCache:
    """`local_dir` downloads: the snapshot-level net always prevents false success; the per-file
    shortcut additionally self-heals when a tree listing is available, and preserves the no-network
    fast path when it is not."""

    def _seed(self, local_dir: Path, filename, *, on_disk: bytes, expected_size: int, write_tree: bool):
        local_dir.mkdir(parents=True, exist_ok=True)
        paths = get_local_download_paths(local_dir=local_dir, filename=filename)
        paths.file_path.parent.mkdir(parents=True, exist_ok=True)
        paths.file_path.write_bytes(on_disk)
        write_download_metadata(local_dir=local_dir, filename=filename, commit_hash=COMMIT, etag=ETAG)
        old = time.time() - 100  # back-date so `read_download_metadata` accepts the metadata
        os.utime(paths.file_path, (old, old))
        if write_tree:
            write_tree_cache(
                tree_cache_folder_for_local_dir(str(local_dir)),
                COMMIT,
                {filename: TreeCacheEntry(size=expected_size, blob_id="b", lfs_sha256=ETAG, lfs_size=expected_size)},
            )

    def test_single_file_repaired_when_tree_listing_available(self, tmp_path):
        local_dir = tmp_path / "ld"
        self._seed(local_dir, "model.bin", on_disk=b"\0" * 400, expected_size=1000, write_tree=True)
        with (
            patch("huggingface_hub.file_download.get_hf_file_metadata", return_value=_http_metadata(1000)),
            patch("huggingface_hub.file_download.http_get", side_effect=_fake_http_get) as http_get,
        ):
            path = hf_hub_download("user/repo", "model.bin", revision=COMMIT, local_dir=local_dir)
        assert os.path.getsize(path) == 1000
        http_get.assert_called_once()

    def test_single_file_without_tree_listing_preserves_fast_path(self, tmp_path):
        """No local size info + immutable commit + matching sidecar => no-network fast path is kept
        (documented limitation; `force_download=True` is the escape hatch)."""
        local_dir = tmp_path / "ld"
        self._seed(local_dir, "model.bin", on_disk=b"\0" * 400, expected_size=1000, write_tree=False)
        with (
            patch("huggingface_hub.file_download.http_get") as http_get,
            patch("huggingface_hub.file_download.get_hf_file_metadata") as meta,
        ):
            path = hf_hub_download("user/repo", "model.bin", revision=COMMIT, local_dir=local_dir)
        assert os.path.getsize(path) == 400
        http_get.assert_not_called()
        meta.assert_not_called()

    def test_snapshot_download_local_dir_repairs_poisoned_file(self, tmp_path):
        local_dir = tmp_path / "ld"
        self._seed(local_dir, "model.bin", on_disk=b"\0" * 400, expected_size=1000, write_tree=False)
        repo_file = RepoFile(path="model.bin", size=1000, oid="b", lfs={"oid": "s", "size": 1000, "pointerSize": 1})
        with (
            patch("huggingface_hub._snapshot_download.HfApi.repo_info", return_value=Mock(sha=COMMIT)),
            patch("huggingface_hub._snapshot_download.HfApi.list_repo_tree", return_value=[repo_file]),
            patch("huggingface_hub.file_download.get_hf_file_metadata", return_value=_http_metadata(1000)),
            patch("huggingface_hub.file_download.http_get", side_effect=_fake_http_get),
        ):
            folder = snapshot_download("user/repo", local_dir=local_dir)
        assert os.path.getsize(os.path.join(folder, "model.bin")) == 1000

    def test_snapshot_download_local_dir_raises_if_left_incomplete(self, tmp_path):
        """If per-file repair is somehow bypassed, the snapshot-level net still blocks false success."""
        local_dir = tmp_path / "ld"
        self._seed(local_dir, "model.bin", on_disk=b"\0" * 400, expected_size=1000, write_tree=False)
        repo_file = RepoFile(path="model.bin", size=1000, oid="b", lfs={"oid": "s", "size": 1000, "pointerSize": 1})
        with (
            patch("huggingface_hub._snapshot_download.HfApi.repo_info", return_value=Mock(sha=COMMIT)),
            patch("huggingface_hub._snapshot_download.HfApi.list_repo_tree", return_value=[repo_file]),
            patch(
                "huggingface_hub._snapshot_download.hf_hub_download",
                side_effect=lambda r, *, filename, **k: str(local_dir / filename),
            ),
        ):
            with pytest.raises(IncompleteSnapshotError, match="incomplete after download"):
                snapshot_download("user/repo", local_dir=local_dir)


def test_raise_if_snapshot_incomplete_accepts_zero_byte_files(tmp_path):
    """Unit check: the snapshot-level validator treats a legit 0-byte file as complete."""
    (tmp_path / "empty.txt").write_bytes(b"")
    (tmp_path / "data.bin").write_bytes(b"\0" * 5)
    # No raise: empty.txt expected 0 / actual 0, data.bin expected 5 / actual 5.
    _raise_if_snapshot_incomplete(
        base_dir=str(tmp_path),
        expected_sizes={"empty.txt": 0, "data.bin": 5},
        repo_id="user/repo",
        commit_hash=COMMIT,
    )
    # Raises once a size disagrees.
    with pytest.raises(IncompleteSnapshotError):
        _raise_if_snapshot_incomplete(
            base_dir=str(tmp_path),
            expected_sizes={"empty.txt": 0, "data.bin": 9},
            repo_id="user/repo",
            commit_hash=COMMIT,
        )
