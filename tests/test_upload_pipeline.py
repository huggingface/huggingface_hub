"""Unit tests for the streamed multi-commit upload pipeline (`_upload_pipeline.py`).

Everything is mocked: no network, no `hf_xet` runtime. The fake Xet session mimics the
`XetSession.new_upload_commit` contract (background uploads, sha256 computation, finalize).
"""

import hashlib
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import huggingface_hub._commit_api as commit_api
import huggingface_hub._upload_pipeline as upload_pipeline
from huggingface_hub._commit_api import CommitOperationAdd, CommitOperationDelete, _compute_missing_sha256s
from huggingface_hub._upload_pipeline import _CommitPacer, _LiveDisplay, _UploadPipeline


pytestmark = pytest.mark.xet

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeHandle:
    def __init__(self, sha256_hex: str):
        self._sha256 = sha256_hex

    def result(self):
        return SimpleNamespace(xet_info=SimpleNamespace(sha256=self._sha256))


class FakeXetCommit:
    """Mimics XetUploadCommit: computes sha256 'in the background' like COMPUTE_SHA256 does."""

    def __init__(self):
        self.files: list[str] = []
        self.finished = False
        self.aborted = False

    def start_upload_file(self, path, sha256=None):
        self.files.append(path)
        with open(path, "rb") as f:
            return FakeHandle(hashlib.sha256(f.read()).hexdigest())

    def start_upload_bytes(self, data, sha256=None, name=None):
        self.files.append(name)
        return FakeHandle(hashlib.sha256(data).hexdigest())

    def wait_to_finish(self):
        self.finished = True

    def abort(self):
        self.aborted = True


class FakeXetSession:
    def __init__(self):
        self.commits: list[FakeXetCommit] = []

    def new_upload_commit(self, **kwargs):
        commit = FakeXetCommit()
        self.commits.append(commit)
        return commit


class FakeCommitEndpoint:
    """Records commit POSTs and returns fake commit responses. Can fail on demand."""

    def __init__(self, fail_on_nth_call: set | None = None):
        self.calls: list[dict] = []  # {url, params, operations}
        self.fail_on = fail_on_nth_call or set()
        self.lock = threading.Lock()

    def __call__(self, method, url, headers=None, content=None, params=None):
        with self.lock:
            n = len(self.calls)
            payload = [json.loads(line) for line in content.decode().strip().split("\n")]
            self.calls.append({"url": url, "params": params, "payload": payload})
            if n in self.fail_on:
                raise RuntimeError(f"injected commit failure (call {n})")
            return SimpleNamespace(
                json=lambda: {
                    "commitUrl": f"https://hub.fake/user/repo/commit/{n}",
                    "commitOid": f"oid{n}",
                }
            )

    def committed_paths(self, call_idx):
        return [item["value"]["path"] for item in self.calls[call_idx]["payload"] if item["key"] != "header"]


def fake_fetch_upload_modes(upload_modes_config=None):
    """Returns a `_fetch_upload_modes` replacement. `.bin` files -> xet ("lfs" mode), others -> regular."""

    def _fake(additions, **kwargs):
        _fake.calls.append(kwargs)
        for op in additions:
            config = (upload_modes_config or {}).get(op.path_in_repo, {})
            op._upload_mode = config.get("mode", "lfs" if op.path_in_repo.endswith(".bin") else "regular")
            op._should_ignore = config.get("ignore", False)
            op._remote_oid = config.get("remote_oid")

    _fake.calls = []
    return _fake


@pytest.fixture
def fake_api(tmp_path):
    api = SimpleNamespace(
        endpoint="https://hub.fake",
        _build_hf_headers=lambda token=None: {"user-agent": "test"},
        repo_info=lambda **kwargs: SimpleNamespace(sha="latestsha"),
        pr_calls=[],
    )

    def create_pull_request(**kwargs):
        api.pr_calls.append(kwargs)
        return SimpleNamespace(url="https://huggingface.co/fake/repo/discussions/7", git_reference="refs/pr/7")

    api.create_pull_request = create_pull_request
    return api


def make_ops(tmp_path, names_and_content):
    ops = []
    for name, content in names_and_content:
        path = tmp_path / name.replace("/", "_")
        path.write_bytes(content)
        ops.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)))
    return ops


def run_pipeline(api, add_operations, commit_endpoint=None, modes=None, **kwargs):
    commit_endpoint = commit_endpoint or FakeCommitEndpoint()
    session = FakeXetSession()
    fetcher = fake_fetch_upload_modes(modes)
    commit_endpoint.preupload_calls = fetcher.calls  # exposed for assertions
    with (
        patch.object(upload_pipeline, "_fetch_upload_modes", fetcher),
        patch.object(upload_pipeline, "get_xet_session", lambda: session),
        patch.object(commit_api, "http_backoff", commit_endpoint),
        patch.object(commit_api, "hf_raise_for_status", lambda *a, **k: None),
        patch.object(upload_pipeline, "are_progress_bars_disabled", lambda: True),
    ):
        pipeline = _UploadPipeline(
            api,
            repo_id="user/repo",
            repo_type="model",
            add_operations=add_operations,
            delete_operations=kwargs.pop("delete_operations", []),
            commit_message=kwargs.pop("commit_message", "msg"),
            commit_description=None,
            token=None,
            revision=kwargs.pop("revision", None),
            create_pr=kwargs.pop("create_pr", False),
            parent_commit=kwargs.pop("parent_commit", None),
        )
        info = pipeline.run()
    return info, commit_endpoint, session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCommitPacer:
    def test_scale_up_on_fast_full_commit(self):
        pacer = _CommitPacer()
        initial = pacer.target
        pacer.record_success(duration=2.0, nb_files=initial)
        assert pacer.target > initial

    def test_no_scale_up_on_partial_commit(self):
        pacer = _CommitPacer()
        initial = pacer.target
        pacer.record_success(duration=2.0, nb_files=initial - 1)
        assert pacer.target == initial

    def test_scale_down_on_slow_commit(self):
        pacer = _CommitPacer()
        initial = pacer.target
        pacer.record_success(duration=100.0, nb_files=initial)
        assert pacer.target < initial

    def test_scale_down_on_failure_and_bounds(self):
        pacer = _CommitPacer()
        for _ in range(20):
            pacer.record_failure()
        assert pacer.target == upload_pipeline.COMMIT_SIZE_SCALE[0]
        for _ in range(20):
            pacer.record_success(duration=1.0, nb_files=pacer.target)
        assert pacer.target == upload_pipeline.COMMIT_SIZE_SCALE[-1]


class TestLiveDisplayCounters:
    def test_xet_callback_sums_increments_across_concurrent_commits(self):
        with patch.object(upload_pipeline.logger, "isEnabledFor", return_value=True):
            display = _LiveDisplay(total_files=10, enabled=True)
        cb1, cb2 = display.new_xet_callback(), display.new_xet_callback()

        def report(n):
            return SimpleNamespace(total_transfer_bytes_completed=n)

        cb1(report(100), {})
        cb2(report(50), {})  # concurrent commit with its own cumulative counter
        cb1(report(300), {})
        assert display._xet_bytes == 350

    def test_disabled_display_returns_no_callback_when_logging_off(self):
        with patch.object(upload_pipeline.logger, "isEnabledFor", return_value=False):
            display = _LiveDisplay(total_files=10, enabled=False)
        assert display.new_xet_callback() is None

    def test_disabled_display_keeps_callback_for_log_summaries(self):
        # Disabling progress bars (e.g. agent mode) must not kill the periodic log summaries.
        with patch.object(upload_pipeline.logger, "isEnabledFor", return_value=True):
            display = _LiveDisplay(total_files=10, enabled=False)
        assert display.new_xet_callback() is not None


class TestUploadPipeline:
    def test_single_commit_mixed_files(self, fake_api, tmp_path):
        ops = make_ops(tmp_path, [("a.txt", b"regular"), ("b.bin", b"x" * 1000), ("c.bin", b"y" * 1000)])
        info, endpoint, session = run_pipeline(fake_api, ops)

        assert len(endpoint.calls) == 1
        assert sorted(endpoint.committed_paths(0)) == ["a.txt", "b.bin", "c.bin"]
        # xet files uploaded through one upload-commit, finalized before the git commit
        assert len(session.commits) == 1
        assert session.commits[0].finished
        assert len(session.commits[0].files) == 2
        # sha256 backfilled from the xet result for unhashed operations
        assert ops[1].upload_info.sha256 == hashlib.sha256(b"x" * 1000).digest()
        assert all(op._is_committed for op in ops)
        assert info.oid == "oid0"

    def test_multi_commit_with_suffix_and_first_commit_extras(self, fake_api, tmp_path):
        ops = make_ops(tmp_path, [(f"f{i}.bin", f"{i}".encode() * 100) for i in range(5)])
        deletes = [CommitOperationDelete(path_in_repo="old.bin")]
        with (
            patch.object(upload_pipeline, "COMMIT_SIZE_SCALE", [2, 2, 2]),
            patch.object(upload_pipeline, "INITIAL_COMMIT_SIZE_INDEX", 0),
        ):
            info, endpoint, _ = run_pipeline(
                fake_api, ops, delete_operations=deletes, parent_commit="a" * 40, commit_message="msg"
            )

        assert len(endpoint.calls) == 3  # 2 + 2 + 1 files
        headers = [call["payload"][0]["value"] for call in endpoint.calls]
        assert headers[0]["summary"] == "msg"
        assert headers[1]["summary"] == "msg (part 2)"
        assert headers[2]["summary"] == "msg (part 3)"
        # deletions and parent_commit ride the first commit only
        assert headers[0].get("parentCommit") == "a" * 40
        assert "parentCommit" not in headers[1]
        assert "old.bin" in endpoint.committed_paths(0)
        assert info.oid == "oid2"  # last commit

    def test_unchanged_files_are_skipped(self, fake_api, tmp_path):
        content = b"z" * 1000
        sha = hashlib.sha256(content).hexdigest()
        ops = make_ops(tmp_path, [("same.bin", content), ("new.bin", b"w" * 1000)])
        modes = {"same.bin": {"remote_oid": sha}}
        info, endpoint, _ = run_pipeline(fake_api, ops, modes=modes)

        assert len(endpoint.calls) == 1
        assert endpoint.committed_paths(0) == ["new.bin"]

    def test_everything_unchanged_creates_no_commit(self, fake_api, tmp_path):
        content = b"z" * 1000
        ops = make_ops(tmp_path, [("same.bin", content)])
        modes = {"same.bin": {"remote_oid": hashlib.sha256(content).hexdigest()}}
        info, endpoint, _ = run_pipeline(fake_api, ops, modes=modes)

        assert len(endpoint.calls) == 0
        assert info.oid == "latestsha"  # from repo_info

    def test_gitignored_files_are_not_committed(self, fake_api, tmp_path):
        ops = make_ops(tmp_path, [("kept.bin", b"a" * 100), ("ignored.bin", b"b" * 100)])
        modes = {"ignored.bin": {"ignore": True}}
        _, endpoint, session = run_pipeline(fake_api, ops, modes=modes)

        assert endpoint.committed_paths(0) == ["kept.bin"]
        assert len(session.commits[0].files) == 1  # ignored file never registered with xet

    def test_create_pr_single_pr_for_multiple_commits(self, fake_api, tmp_path):
        ops = make_ops(tmp_path, [(f"f{i}.bin", f"{i}".encode() * 100) for i in range(4)])
        with (
            patch.object(upload_pipeline, "COMMIT_SIZE_SCALE", [2, 2]),
            patch.object(upload_pipeline, "INITIAL_COMMIT_SIZE_INDEX", 0),
            patch.object(upload_pipeline, "PREUPLOAD_BATCH_SIZE", 2),  # several preupload calls
        ):
            info, endpoint, _ = run_pipeline(fake_api, ops, create_pr=True)

        assert len(endpoint.calls) == 2
        # a single PR is created upfront (via the discussions API, not `?create_pr=1`)...
        assert len(fake_api.pr_calls) == 1
        assert fake_api.pr_calls[0]["title"] == "msg"
        # ...and ALL commits push to the (url-quoted) PR ref
        assert all(call["url"].endswith("/commit/refs%2Fpr%2F7") for call in endpoint.calls)
        assert all(not call["params"] for call in endpoint.calls)
        assert info.pr_url == "https://huggingface.co/fake/repo/discussions/7"
        assert info.pr_revision == "refs/pr/7"
        # preupload always targets the base revision with create_pr=True, even after the PR exists
        # (mirrors `create_commit` semantics; the PR ref is only used for the commit calls)
        assert all(call["revision"] == "main" and call["create_pr"] is True for call in endpoint.preupload_calls)

    def test_commit_failure_splits_batch(self, fake_api, tmp_path):
        ops = make_ops(tmp_path, [(f"f{i}.bin", f"{i}".encode() * 100) for i in range(4)])
        endpoint = FakeCommitEndpoint(fail_on_nth_call={0})  # first attempt (4 files) fails
        with (
            patch.object(upload_pipeline, "COMMIT_SIZE_SCALE", [2, 4]),
            patch.object(upload_pipeline, "INITIAL_COMMIT_SIZE_INDEX", 1),
        ):
            info, endpoint, _ = run_pipeline(fake_api, ops, commit_endpoint=endpoint)

        # 1 failed attempt + 2 successful halves
        assert len(endpoint.calls) == 3
        assert len(endpoint.committed_paths(1)) == 2
        assert len(endpoint.committed_paths(2)) == 2
        committed = endpoint.committed_paths(1) + endpoint.committed_paths(2)
        assert sorted(committed) == [f"f{i}.bin" for i in range(4)]

    def test_create_pr_interrupted_warns_with_resume_instructions(self, fake_api, tmp_path, caplog):
        """If the upload fails after the PR was created, tell the user how to resume into the SAME PR
        (re-running with create_pr=True would open a second one)."""
        ops = make_ops(tmp_path, [(f"f{i}.bin", f"{i}".encode() * 100) for i in range(2)])
        endpoint = FakeCommitEndpoint(fail_on_nth_call=set(range(100)))  # PR created, then every commit fails
        with (
            patch.object(upload_pipeline, "COMMIT_SIZE_SCALE", [1, 1]),
            patch.object(upload_pipeline, "INITIAL_COMMIT_SIZE_INDEX", 0),
        ):
            with pytest.raises(RuntimeError, match="injected commit failure"):
                run_pipeline(fake_api, ops, commit_endpoint=endpoint, create_pr=True)
        assert 'revision="refs/pr/7"' in caplog.text

    def test_create_pr_with_revision_raises(self, tmp_path):
        """PRs are always opened against the default branch (the discussions API has no base-revision
        support), so `create_pr=True` + `revision` is rejected upfront."""
        from huggingface_hub import HfApi

        with pytest.raises(ValueError, match="create_pr"):
            HfApi().upload_folder(repo_id="user/repo", folder_path=tmp_path, create_pr=True, revision="my-branch")

    def test_persistent_commit_failure_raises(self, fake_api, tmp_path):
        ops = make_ops(tmp_path, [("f.bin", b"x" * 100)])
        endpoint = FakeCommitEndpoint(fail_on_nth_call=set(range(100)))
        with pytest.raises(RuntimeError, match="injected commit failure"):
            run_pipeline(fake_api, ops, commit_endpoint=endpoint)

    def test_coordinator_error_propagates_without_hanging(self, fake_api, tmp_path):
        ops = make_ops(tmp_path, [("f.bin", b"x" * 100)])

        def failing_fetch(additions, **kwargs):
            raise ValueError("preupload exploded")

        session = FakeXetSession()
        with (
            patch.object(upload_pipeline, "_fetch_upload_modes", failing_fetch),
            patch.object(upload_pipeline, "get_xet_session", lambda: session),
            patch.object(upload_pipeline, "abort_xet_session", lambda: None),
            patch.object(upload_pipeline, "are_progress_bars_disabled", lambda: True),
        ):
            pipeline = _UploadPipeline(
                fake_api,
                repo_id="user/repo",
                repo_type="model",
                add_operations=ops,
                delete_operations=[],
                commit_message="msg",
                commit_description=None,
                token=None,
                revision=None,
                create_pr=False,
                parent_commit=None,
            )
            with pytest.raises(ValueError, match="preupload exploded"):
                pipeline.run()


class TestComputeMissingSha256s:
    def test_hashes_only_missing(self, tmp_path):
        path = tmp_path / "f.bin"
        path.write_bytes(b"content")
        op_lazy = CommitOperationAdd(path_in_repo="f.bin", path_or_fileobj=str(path))
        op_eager = CommitOperationAdd(path_in_repo="b.bin", path_or_fileobj=b"other")
        assert not op_lazy.upload_info.is_hashed
        assert op_eager.upload_info.is_hashed

        _compute_missing_sha256s([op_lazy, op_eager], num_threads=2)
        assert op_lazy.upload_info.is_hashed
        assert op_lazy.upload_info.sha256 == hashlib.sha256(b"content").digest()
