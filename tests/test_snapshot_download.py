import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from huggingface_hub import CommitOperationAdd, HfApi, ResolvedRevision, hf_hub_download, snapshot_download
from huggingface_hub._tree_cache import read_tree_cache
from huggingface_hub.errors import (
    IncompleteSnapshotError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionResolutionError,
)
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import SoftTemporaryDirectory, _http

from .testing_constants import TOKEN
from .testing_utils import OfflineSimulationMode, offline, repo_name


COMMIT_HASH = "0123456789abcdef0123456789abcdef01234567"
MASKED_XET_HASH = "*" * 64


def test_tree_with_redacted_xet_hash_is_not_cached(tmp_path: Path):
    repo_file = RepoFile(
        path="model.safetensors",
        size=42,
        oid="blob-model",
        lfs={"oid": "sha256-model", "size": 42, "pointerSize": 128},
        xetHash=MASKED_XET_HASH,
    )

    with (
        patch("huggingface_hub._snapshot_download.HfApi.repo_info", return_value=MagicMock(sha=COMMIT_HASH)),
        patch("huggingface_hub._snapshot_download.HfApi.list_repo_tree", return_value=[repo_file]),
        patch("huggingface_hub._snapshot_download.hf_thread_map"),
    ):
        snapshot_download("user/repo", cache_dir=tmp_path)

    storage_folder = tmp_path / repo_folder_name(repo_id="user/repo", repo_type="model")
    assert read_tree_cache(str(storage_folder), COMMIT_HASH) is None


class TestSnapshotDownload:
    @pytest.fixture(scope="class", autouse=True)
    def _shared_repo(self, request, api: HfApi):
        """
        Share this valid token in all tests below.
        """
        repo_id = api.create_repo(repo_name("snapshot-download")).repo_id
        request.cls.repo_id = repo_id

        # First commit on `main`
        request.cls.first_commit_hash = api.create_commit(
            repo_id=repo_id,
            operations=[
                CommitOperationAdd(path_in_repo="dummy_file.txt", path_or_fileobj=b"v1"),
                CommitOperationAdd(path_in_repo="subpath/file.txt", path_or_fileobj=b"content in subpath"),
            ],
            commit_message="Add file to main branch",
        ).oid

        # Second commit on `main`
        request.cls.second_commit_hash = api.create_commit(
            repo_id=repo_id,
            operations=[
                CommitOperationAdd(path_in_repo="dummy_file.txt", path_or_fileobj=b"v2"),
                CommitOperationAdd(path_in_repo="file.bin", path_or_fileobj=os.urandom(1 * 1024 * 1024)),
            ],
            commit_message="Add file to main branch",
        ).oid

        # Third commit on `other`
        api.create_branch(repo_id=repo_id, branch="other")
        request.cls.third_commit_hash = api.create_commit(
            repo_id=repo_id,
            operations=[
                CommitOperationAdd(path_in_repo="dummy_file_2.txt", path_or_fileobj=b"v4"),
            ],
            commit_message="Add file to other branch",
            revision="other",
        ).oid

        yield
        api.delete_repo(repo_id=repo_id)

    def test_download_model(self):
        # Test `main` branch
        with SoftTemporaryDirectory() as tmpdir:
            storage_folder = snapshot_download(self.repo_id, revision="main", cache_dir=tmpdir)

            # folder contains the two files contributed and the .gitattributes
            folder_contents = os.listdir(storage_folder)
            assert len(folder_contents) == 4
            assert "dummy_file.txt" in folder_contents
            assert "file.bin" in folder_contents
            assert ".gitattributes" in folder_contents

            with open(os.path.join(storage_folder, "dummy_file.txt"), "r") as f:
                contents = f.read()
                assert contents == "v2"

            # folder name contains the revision's commit sha.
            assert self.second_commit_hash in storage_folder

        # Test with specific revision
        with SoftTemporaryDirectory() as tmpdir:
            storage_folder = snapshot_download(
                self.repo_id,
                revision=self.first_commit_hash,
                cache_dir=tmpdir,
            )

            # folder contains the two files contributed and the .gitattributes
            folder_contents = os.listdir(storage_folder)
            assert len(folder_contents) == 3
            assert "dummy_file.txt" in folder_contents
            assert ".gitattributes" in folder_contents

            with open(os.path.join(storage_folder, "dummy_file.txt"), "r") as f:
                contents = f.read()
                assert contents == "v1"

            # folder name contains the revision's commit sha.
            assert self.first_commit_hash in storage_folder

    @pytest.mark.xet
    def test_xet_file_skips_per_file_head_call(self):
        """A regular file is HEAD-ed during download, but a Xet file's HEAD is skipped.

        For a Xet file the metadata is rebuilt from the tree listing cached on disk by `snapshot_download`,
        so the per-file HEAD `/resolve/` call is never made (the data transfer happens inside `hf_xet`).
        """
        session = _http.get_session()
        with SoftTemporaryDirectory() as tmpdir:
            # `max_workers=1` keeps the spying deterministic (single thread).
            with patch.object(session, "request", wraps=session.request) as mock_request:
                snapshot_download(self.repo_id, revision="main", cache_dir=tmpdir, max_workers=1)

        head_urls = [
            str(call.kwargs["url"]) for call in mock_request.call_args_list if call.kwargs.get("method") == "HEAD"
        ]
        # Regular file => its `/resolve/` URL is HEAD-ed.
        assert any(url.endswith("/dummy_file.txt") for url in head_urls)
        # Xet file => no HEAD call at all.
        assert not any(url.endswith("/file.bin") for url in head_urls)

    def test_download_private_model(self, api: HfApi):
        api.update_repo_settings(repo_id=self.repo_id, private=True)

        # Test download fails without token
        with SoftTemporaryDirectory() as tmpdir:
            with pytest.raises(RepositoryNotFoundError):
                _ = snapshot_download(self.repo_id, revision="main", cache_dir=tmpdir)

        # Test we can download with token from cache
        with patch("huggingface_hub.utils._headers.get_token", return_value=TOKEN):
            with SoftTemporaryDirectory() as tmpdir:
                storage_folder = snapshot_download(self.repo_id, revision="main", cache_dir=tmpdir)
                assert self.second_commit_hash in storage_folder

        # Test we can download with explicit token
        with SoftTemporaryDirectory() as tmpdir:
            storage_folder = snapshot_download(self.repo_id, revision="main", cache_dir=tmpdir, token=TOKEN)
            assert self.second_commit_hash in storage_folder

        api.update_repo_settings(repo_id=self.repo_id, private=False)

    def test_download_model_local_only(self):
        # Test no branch specified
        with SoftTemporaryDirectory() as tmpdir:
            # first download folder to cache it
            snapshot_download(self.repo_id, cache_dir=tmpdir)
            # now load from cache
            storage_folder = snapshot_download(self.repo_id, cache_dir=tmpdir, local_files_only=True)
            assert self.second_commit_hash in storage_folder  # has expected revision

        # Test with specific revision branch
        with SoftTemporaryDirectory() as tmpdir:
            # first download folder to cache it
            snapshot_download(self.repo_id, revision="other", cache_dir=tmpdir)
            # now load from cache
            storage_folder = snapshot_download(self.repo_id, revision="other", cache_dir=tmpdir, local_files_only=True)
            assert self.third_commit_hash in storage_folder  # has expected revision

        # Test with specific revision hash
        with SoftTemporaryDirectory() as tmpdir:
            # first download folder to cache it
            snapshot_download(self.repo_id, revision=self.first_commit_hash, cache_dir=tmpdir)
            # now load from cache
            storage_folder = snapshot_download(
                self.repo_id, revision=self.first_commit_hash, cache_dir=tmpdir, local_files_only=True
            )
            assert self.first_commit_hash in storage_folder  # has expected revision

        # Test with local_dir
        with SoftTemporaryDirectory() as tmpdir:
            # first download folder to local_dir
            snapshot_download(self.repo_id, local_dir=tmpdir)
            # now load from local_dir
            storage_folder = snapshot_download(self.repo_id, local_dir=tmpdir, local_files_only=True)
            assert str(tmpdir) == storage_folder

    def test_download_model_to_local_dir_with_offline_mode(self):
        """Test that an already downloaded folder is returned when there is a connection error"""
        # first download folder to local_dir
        with SoftTemporaryDirectory() as tmpdir:
            snapshot_download(self.repo_id, local_dir=tmpdir)
            # Check that the folder is returned when there is a connection error
            for offline_mode in OfflineSimulationMode:
                with offline(mode=offline_mode):
                    storage_folder = snapshot_download(self.repo_id, local_dir=tmpdir)
                    assert str(tmpdir) == storage_folder

    def test_offline_mode_with_cache_and_empty_local_dir(self):
        """Test that when cache exists but an empty local_dir is specified in offline mode, we raise an error."""
        with SoftTemporaryDirectory() as tmpdir_cache:
            snapshot_download(self.repo_id, cache_dir=tmpdir_cache)

            for offline_mode in OfflineSimulationMode:
                with offline(mode=offline_mode):
                    with pytest.raises(LocalEntryNotFoundError):
                        with SoftTemporaryDirectory() as tmpdir:
                            snapshot_download(self.repo_id, cache_dir=tmpdir_cache, local_dir=tmpdir)

    def test_download_model_offline_mode_not_in_local_dir(self):
        """Test when connection error but local_dir is empty."""
        with SoftTemporaryDirectory() as tmpdir:
            with pytest.raises(LocalEntryNotFoundError):
                snapshot_download(self.repo_id, local_dir=tmpdir, local_files_only=True)

        for offline_mode in OfflineSimulationMode:
            with offline(mode=offline_mode):
                with SoftTemporaryDirectory() as tmpdir:
                    with pytest.raises(LocalEntryNotFoundError):
                        snapshot_download(self.repo_id, local_dir=tmpdir)

    def test_download_model_offline_mode_not_cached(self):
        """Test when connection error but cache is empty."""
        with SoftTemporaryDirectory() as tmpdir:
            with pytest.raises(LocalEntryNotFoundError):
                snapshot_download(self.repo_id, cache_dir=tmpdir, local_files_only=True)

        for offline_mode in OfflineSimulationMode:
            with offline(mode=offline_mode):
                with SoftTemporaryDirectory() as tmpdir:
                    with pytest.raises(LocalEntryNotFoundError):
                        snapshot_download(self.repo_id, cache_dir=tmpdir)

    def test_tree_cache_written_and_incomplete_detected(self):
        """The repo tree listing is cached on disk, and an incomplete snapshot is detected offline."""
        with SoftTemporaryDirectory() as tmpdir:
            snapshot_path = snapshot_download(self.repo_id, cache_dir=tmpdir)
            commit_hash = os.path.basename(snapshot_path)

            # The tree listing of the resolved commit is cached under `trees/`.
            storage_folder = os.path.join(tmpdir, repo_folder_name(repo_id=self.repo_id, repo_type="model"))
            tree_cache_file = os.path.join(storage_folder, "trees", f"{commit_hash}.json")
            assert os.path.isfile(tree_cache_file)

            # A complete cached snapshot is still returned offline.
            with offline():
                assert snapshot_download(self.repo_id, cache_dir=tmpdir) == snapshot_path

            # Remove a file from the snapshot => offline re-pull now raises instead of returning a partial folder.
            os.remove(os.path.join(snapshot_path, "dummy_file.txt"))
            for offline_mode in OfflineSimulationMode:
                with offline(mode=offline_mode):
                    with pytest.raises(IncompleteSnapshotError):
                        snapshot_download(self.repo_id, cache_dir=tmpdir)

    def test_download_model_local_only_multiple(self):
        # cache multiple commits and make sure correct commit is taken
        with SoftTemporaryDirectory() as tmpdir:
            # download folder from main and other to cache it
            snapshot_download(self.repo_id, cache_dir=tmpdir)
            snapshot_download(self.repo_id, revision="other", cache_dir=tmpdir)

            # now make sure that loading "main" branch gives correct branch
            # folder name contains the 2nd commit sha and not the 3rd
            storage_folder = snapshot_download(self.repo_id, cache_dir=tmpdir, local_files_only=True)
            assert self.second_commit_hash in storage_folder

    def check_download_model_with_pattern(self, pattern, expected, allow=True):
        # Test `main` branch
        allow_patterns = pattern if allow else None
        ignore_patterns = pattern if not allow else None

        with SoftTemporaryDirectory() as tmpdir:
            storage_folder = snapshot_download(
                self.repo_id,
                revision="main",
                cache_dir=tmpdir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            assert set(os.listdir(storage_folder)) == expected

    def test_download_model_with_allow_pattern(self):
        # `file.bin` is filtered out (not a `*.txt` file); `subpath/file.txt` keeps the `subpath` folder.
        self.check_download_model_with_pattern("*.txt", expected={"dummy_file.txt", "subpath"})

    def test_download_model_with_allow_pattern_list(self):
        self.check_download_model_with_pattern(
            ["dummy_file.txt", "file.bin", "subpath/*"], expected={"dummy_file.txt", "file.bin", "subpath"}
        )

    def test_download_model_with_ignore_pattern(self):
        self.check_download_model_with_pattern(
            ".gitattributes", expected={"dummy_file.txt", "file.bin", "subpath"}, allow=False
        )

    def test_download_model_with_ignore_pattern_list(self):
        self.check_download_model_with_pattern(
            ["*.git*", "*.pt"], expected={"dummy_file.txt", "file.bin", "subpath"}, allow=False
        )

    def test_download_to_local_dir(self) -> None:
        """Download a repository to local dir.

        Cache dir is not used.
        Symlinks are not used.

        This test is here to check once the normal behavior with snapshot_download.
        More individual tests exists in `test_file_download.py`.
        """
        with SoftTemporaryDirectory() as cache_dir:
            with SoftTemporaryDirectory() as local_dir:
                returned_path = snapshot_download(self.repo_id, cache_dir=cache_dir, local_dir=local_dir)

                # Files have been downloaded in correct structure
                assert (Path(local_dir) / "dummy_file.txt").is_file()
                assert (Path(local_dir) / "file.bin").is_file()
                assert (Path(local_dir) / "subpath" / "file.txt").is_file()

                # Symlinks are not used anymore
                assert not (Path(local_dir) / "dummy_file.txt").is_symlink()
                assert not (Path(local_dir) / "file.bin").is_symlink()
                assert not (Path(local_dir) / "subpath" / "file.txt").is_symlink()

                # Check returns local dir and not cache dir
                assert Path(returned_path).resolve() == Path(local_dir).resolve()

                # Nothing has been added to cache dir (except some subfolders created)
                for path in cache_dir.glob("*"):
                    assert path.is_dir()


def test_revision_str():
    revision = ResolvedRevision(resolved=COMMIT_HASH)
    assert revision == "main"  # defaults to `main` when no revision was requested
    assert revision.initial is None
    assert revision.resolved == COMMIT_HASH
    assert repr(revision) == f"ResolvedRevision(initial=None, resolved='{COMMIT_HASH}')"

    revision = ResolvedRevision(resolved=COMMIT_HASH, initial="refs/pr/4")
    assert revision == "refs/pr/4"
    assert revision.resolved == COMMIT_HASH


class TestResolveRevision:
    @pytest.fixture(scope="class", autouse=True)
    def _shared_repo(self, request, api: HfApi):
        repo_id = api.create_repo(repo_name("resolve-revision")).repo_id
        request.cls.repo_id = repo_id
        request.cls.commit_hash = api.create_commit(
            repo_id=repo_id,
            operations=[CommitOperationAdd(path_in_repo="dummy_file.txt", path_or_fileobj=b"v1")],
            commit_message="Add file to main branch",
        ).oid
        yield
        api.delete_repo(repo_id=repo_id)

    def test_resolve_revision(self, api: HfApi, tmp_path: Path):
        revision = api.resolve_revision(self.repo_id, cache_dir=tmp_path)
        assert revision == "main"
        assert revision.resolved == self.commit_hash

        # The mapping is cached on disk => can be resolved again without network
        ref_path = tmp_path / repo_folder_name(repo_id=self.repo_id, repo_type="model") / "refs" / "main"
        assert ref_path.read_text() == self.commit_hash
        with offline():
            assert api.resolve_revision(self.repo_id, cache_dir=tmp_path).resolved == self.commit_hash

        # Already resolved => returned as is (no network call)
        with offline():
            assert api.resolve_revision(self.repo_id, revision=revision, cache_dir=tmp_path) is revision

    def test_resolve_revision_not_cached(self, api: HfApi, tmp_path: Path):
        with offline():
            with pytest.raises(RevisionResolutionError):
                api.resolve_revision(self.repo_id, cache_dir=tmp_path)

    def test_download_with_resolved_revision(self, api: HfApi, tmp_path: Path):
        """A resolved revision is used directly: no extra call to resolve it again."""
        revision = api.resolve_revision(self.repo_id, cache_dir=tmp_path)

        with patch("huggingface_hub._snapshot_download.HfApi.repo_info", side_effect=AssertionError) as mock:
            snapshot_path = snapshot_download(self.repo_id, revision=revision, cache_dir=tmp_path)
        mock.assert_not_called()
        assert snapshot_path.endswith(self.commit_hash)

        # The `refs/` entry is not touched: it has been written by `resolve_revision` and might be more recent
        # than the pinned commit hash.
        ref_path = tmp_path / repo_folder_name(repo_id=self.repo_id, repo_type="model") / "refs" / "main"
        ref_path.write_text(COMMIT_HASH)
        snapshot_download(self.repo_id, revision=revision, cache_dir=tmp_path)
        assert ref_path.read_text() == COMMIT_HASH

        # Everything is cached at this point => works offline as well
        with offline():
            assert hf_hub_download(self.repo_id, "dummy_file.txt", revision=revision, cache_dir=tmp_path).endswith(
                os.path.join(self.commit_hash, "dummy_file.txt")
            )
