import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import pytest

import huggingface_hub._sandbox as sandbox_mod
import huggingface_hub._sandbox_cache as cache_mod
from huggingface_hub._sandbox import (
    HOST_LABEL,
    POOL_LABEL,
    SANDBOX_LABEL,
    Sandbox,
    SandboxCommandResult,
    SandboxPool,
    _derive_sandbox_token,
    _duration_to_secs,
    _SandboxServer,
)
from huggingface_hub._sandbox_cache import CachedHost, read_pool_cache, save_pool_cache
from huggingface_hub.errors import SandboxCommandError, SandboxError


@pytest.fixture(autouse=True)
def _isolate_pool_cache(tmp_path, monkeypatch):
    """Point the best-effort pool cache at a throwaway dir so tests never touch ~/.cache."""
    monkeypatch.setattr(cache_mod.constants, "HF_HOME", str(tmp_path))


class TestHelpers:
    def test_derive_sandbox_token_deterministic(self) -> None:
        token = _derive_sandbox_token("hf_xxx", "abcd1234")
        assert token == _derive_sandbox_token("hf_xxx", "abcd1234")
        assert len(token) == 64  # hex sha256

    def test_derive_sandbox_token_unique_per_nonce_and_token(self) -> None:
        base = _derive_sandbox_token("hf_xxx", "nonce1")
        assert base != _derive_sandbox_token("hf_xxx", "nonce2")
        assert base != _derive_sandbox_token("hf_yyy", "nonce1")

    @pytest.mark.parametrize(
        "value, expected",
        [(300, 300), (1.5, 1), ("300", 300), ("300s", 300), ("10m", 600), ("2h", 7200), ("1d", 86400)],
    )
    def test_duration_to_secs(self, value, expected) -> None:
        assert _duration_to_secs(value) == expected

    def test_duration_to_secs_invalid(self) -> None:
        with pytest.raises(ValueError):
            _duration_to_secs("oops")

    def test_command_result_ok(self) -> None:
        assert SandboxCommandResult(exit_code=0, stdout="", stderr="").ok
        assert not SandboxCommandResult(exit_code=1, stdout="", stderr="").ok

    def test_command_error_message(self) -> None:
        result = SandboxCommandResult(exit_code=2, stdout="", stderr="boom")
        error = SandboxCommandError(cmd="make", result=result)
        assert "exited with code 2" in str(error)
        assert "boom" in str(error)
        assert error.result is result


def _make_server(base_url: str, job_id: str = "job123", capacity: int = 0) -> _SandboxServer:
    """Build a _SandboxServer wired to a local fake server, bypassing job creation."""
    api = MagicMock()
    api.token = "hf_test"
    return _SandboxServer(
        job_id=job_id,
        owner="user",
        image="python:3.12",
        base_url=base_url,
        nonce="nonce",
        sandbox_token="secret",
        api=api,
        capacity=capacity,
    )


def _make_sandbox(base_url: str) -> Sandbox:
    """A dedicated sandbox (one job) wired to a local server."""
    server = _make_server(base_url)
    return Sandbox(id="job123", server=server, local_id=None, owns_sandbox=True, owns_server=True)


class _FakeServer(BaseHTTPRequestHandler):
    """Minimal stand-in for sbx-server speaking the same protocol (both modes)."""

    sandboxes: set = set()
    capacity = None  # None == unlimited; set per-subclass to test the full handshake
    seq = 0  # monotonic id source (survives deletes)
    last_exec: dict | None = None  # body of the most recent /exec call (for assertions)

    def log_message(self, *args) -> None:
        pass

    def _ndjson(self, events) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for event in events:
            self.wfile.write((json.dumps(event) + "\n").encode())
            self.wfile.flush()

    def _json(self, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _exec(self, body) -> None:
        type(self).last_exec = body
        self._ndjson(
            [
                {"event": "start", "pid": 42},
                {"event": "stdout", "data": "out1"},
                {"event": "ping"},
                {"event": "stderr", "data": "err1"},
                {"event": "exit", "exit_code": 0 if body["cmd"] != "fail" else 3, "duration_ms": 5},
            ]
        )

    def do_POST(self) -> None:
        assert self.headers["X-Sandbox-Token"] == "secret"
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        # exec (dedicated /v1/exec or shared /v1/sandboxes/<id>/exec)
        if self.path == "/v1/exec" or (self.path.startswith("/v1/sandboxes/") and self.path.endswith("/exec")):
            self._exec(body)
        elif self.path == "/v1/sandboxes":  # batch-create sandboxes (server-authoritative capacity)
            cls = type(self)
            count = int(body.get("count", 1))
            created = []
            rejected = 0
            for i in range(count):
                if cls.capacity is not None and len(cls.sandboxes) >= cls.capacity:
                    rejected = count - i
                    break
                sid = f"sbx{cls.seq}"
                cls.seq += 1
                cls.sandboxes.add(sid)
                created.append({"id": sid})
            self._json({"sandboxes": created, "rejected": rejected})

    def do_DELETE(self) -> None:
        assert self.headers["X-Sandbox-Token"] == "secret"
        sid = self.path.rsplit("/", 1)[-1]
        type(self).sandboxes.discard(sid)
        self._json({"id": sid, "deleted": True})

    def do_GET(self) -> None:
        cls = type(self)
        if self.path.startswith("/v1/files/stat") or "/files/stat" in self.path:
            self._json({"name": "x", "path": "/x", "type": "file", "size": 5})
        elif self.path.startswith("/v1/files/read") or "/files/read" in self.path:
            self.send_response(200)
            self.send_header("Content-Length", "5")
            self.end_headers()
            self.wfile.write(b"hello")
        elif self.path == "/v1/sandboxes":
            self._json([{"id": sid} for sid in sorted(cls.sandboxes)])


@pytest.fixture()
def fake_server():
    _FakeServer.sandboxes = set()
    _FakeServer.seq = 0
    _FakeServer.capacity = None
    _FakeServer.last_exec = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _spawn_fake(capacity=None):
    """Start an independent fake server (its own sandbox set/capacity). Returns (url, cls)."""

    class _Fake(_FakeServer):
        sandboxes: set = set()
        seq = 0

    _Fake.capacity = capacity
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}", _Fake


class TestSandboxClient:
    def test_run_collects_streams_and_skips_pings(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        result = sandbox.run("echo")
        assert result.exit_code == 0
        assert result.stdout == "out1"
        assert result.stderr == "err1"

    def test_run_raises_on_nonzero_exit(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        with pytest.raises(SandboxCommandError) as exc_info:
            sandbox.run("fail")
        assert exc_info.value.result.exit_code == 3
        assert sandbox.run("fail", check=False).exit_code == 3

    def test_run_callbacks(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        stdout_chunks: list = []
        sandbox.run("echo", on_stdout=stdout_chunks.append)
        assert stdout_chunks == ["out1"]

    def test_run_infers_shell_from_type(self, fake_server: str) -> None:
        # Without an explicit `shell`, the mode is inferred from the type and no `shell`
        # field is sent on the wire (the server keeps inferring from the type of `cmd`).
        sandbox = _make_sandbox(fake_server)
        sandbox.run("echo hi")
        assert _FakeServer.last_exec == {"cmd": "echo hi"}
        sandbox.run(["echo", "hi"])
        assert _FakeServer.last_exec == {"cmd": ["echo", "hi"]}

    def test_run_explicit_shell_is_sent(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        sandbox.run("echo hi", shell=True)
        assert _FakeServer.last_exec == {"cmd": "echo hi", "shell": True}
        sandbox.run(["echo", "hi"], shell=False)
        assert _FakeServer.last_exec == {"cmd": ["echo", "hi"], "shell": False}

    def test_run_rejects_mismatched_shell_and_cmd(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        with pytest.raises(ValueError, match="shell=True requires"):
            sandbox.run(["echo", "hi"], shell=True)
        with pytest.raises(ValueError, match="shell=False requires"):
            sandbox.run("echo hi", shell=False)

    def test_files_read(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        assert sandbox.files.read_text("/x") == "hello"

    def test_kill_is_idempotent(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        sandbox.kill()
        sandbox.kill()
        sandbox._server._api.cancel_job.assert_called_once_with(job_id="job123", namespace="user")

    def test_kill_classmethod_connects_and_kills(self, monkeypatch) -> None:
        # `Sandbox.kill(id)` is sugar for `connect(id).kill()` (mirrors `hf sandbox kill <id>`),
        # while `sbx.kill()` still works on a live handle — both via the _KillMethod descriptor.
        connected = MagicMock()
        monkeypatch.setattr(
            sandbox_mod.Sandbox,
            "connect",
            classmethod(lambda cls, sid, namespace=None, token=None: connected),
        )
        Sandbox.kill("job-xyz", namespace="org")
        connected.kill.assert_called_once_with()

    def test_context_manager_kills(self, fake_server: str) -> None:
        with _make_sandbox(fake_server) as sandbox:
            pass
        sandbox._server._api.cancel_job.assert_called_once()

    def test_context_manager_closes_when_reattached(self, fake_server: str) -> None:
        # A sandbox reattached via `connect` (owns_sandbox=False) keeps running on exit: the
        # local HTTP client is released but the job is not cancelled.
        sandbox = _make_sandbox(fake_server)
        sandbox._owns_sandbox = False
        with sandbox:
            pass
        sandbox._server._api.cancel_job.assert_not_called()
        assert sandbox._server._client.is_closed


class TestSharedSandbox:
    """A shared sandbox routes operations under /v1/sandboxes/<local_id>/ and is
    terminated with a DELETE on the host (the host job keeps running)."""

    def _make_shared(self, base_url: str) -> Sandbox:
        server = _make_server(base_url, capacity=10)
        _FakeServer.sandboxes.add("local1")
        return Sandbox(id="job123.local1", server=server, local_id="local1", owns_sandbox=True, owns_server=False)

    def test_base_path_is_scoped(self, fake_server: str) -> None:
        sandbox = self._make_shared(fake_server)
        assert sandbox._base_path == "/v1/sandboxes/local1"
        assert sandbox.host_id == "job123"
        # exec is routed under the per-sandbox prefix and still parsed correctly.
        assert sandbox.run("echo").stdout == "out1"

    def test_kill_deletes_sandbox_not_job(self, fake_server: str) -> None:
        sandbox = self._make_shared(fake_server)
        sandbox.kill()
        sandbox._server._api.cancel_job.assert_not_called()  # host keeps running
        assert "local1" not in _FakeServer.sandboxes


class TestSandboxPool:
    def _pool(self, fake_server: str, monkeypatch, per_host: int = 4) -> SandboxPool:
        # Patch before construction: the constructor warms up `warm_up` (=1) host(s), so the
        # fake boot + empty discovery must already be in place. No warm hosts to discover here;
        # discovery is covered separately. Every host boot returns a server at the fake server.
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", lambda self, **kw: [])
        monkeypatch.setattr(SandboxPool, "_boot_host", lambda self: _make_server(fake_server, capacity=per_host))
        return SandboxPool(image="python:3.12", sandboxes_per_host=per_host, token="hf_test")

    def test_packs_into_hosts_and_tracks_slots(self, fake_server: str, monkeypatch) -> None:
        pool = self._pool(fake_server, monkeypatch, per_host=4)
        boxes = [pool.create() for _ in range(6)]  # 6 sandboxes, 4 per host -> 2 hosts
        assert len(boxes) == 6
        assert pool.num_hosts == 2
        assert pool.num_sandboxes == 6
        # Each sandbox id is "<host_job_id>.<local_id>".
        assert all("." in b.id for b in boxes)

    def test_create_returns_one_sandbox(self, fake_server: str, monkeypatch) -> None:
        pool = self._pool(fake_server, monkeypatch)
        box = pool.create()
        assert isinstance(box, Sandbox)
        assert pool.num_sandboxes == 1

    def test_kill_frees_slot(self, fake_server: str, monkeypatch) -> None:
        pool = self._pool(fake_server, monkeypatch, per_host=4)
        boxes = [pool.create() for _ in range(2)]
        assert pool.num_sandboxes == 2
        boxes[0].kill()
        assert pool.num_sandboxes == 1  # slot reclaimed via the pool callback

    def test_reuses_free_slots_before_new_host(self, fake_server: str, monkeypatch) -> None:
        pool = self._pool(fake_server, monkeypatch, per_host=4)
        for _ in range(4):
            pool.create()  # all fit on the same host (4 slots)
        assert pool.num_hosts == 1
        assert pool.num_sandboxes == 4

    def test_warm_up_preprovisions_hosts(self, fake_server: str, monkeypatch) -> None:
        # warm_up=3 boots 3 hosts in the constructor; the rest stay warm for later.
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", lambda self, **kw: [])
        monkeypatch.setattr(SandboxPool, "_boot_host", lambda self: _make_server(fake_server, capacity=4))
        pool = SandboxPool(image="python:3.12", sandboxes_per_host=4, warm_up=3, token="hf_test")
        assert pool.num_hosts == 3  # pre-provisioned by the constructor, before any create()
        pool.create()
        assert pool.num_hosts == 3  # still 3 despite only one sandbox created
        assert pool.num_sandboxes == 1

    def test_max_hosts_enforced(self, fake_server: str, monkeypatch) -> None:
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", lambda self, **kw: [])
        monkeypatch.setattr(SandboxPool, "_boot_host", lambda self: _make_server(fake_server, capacity=2))
        pool = SandboxPool(sandboxes_per_host=2, max_hosts=1, token="hf_test")
        pool.create()
        pool.create()  # fills the single allowed host (capacity 2)
        with pytest.raises(SandboxError, match="max_hosts"):
            pool.create()  # would need a 2nd host, only 1 allowed

    def test_close_cancels_hosts(self, fake_server: str, monkeypatch) -> None:
        pool = self._pool(fake_server, monkeypatch)
        pool.create()
        host = pool._hosts[0]
        pool.close()
        host._api.cancel_job.assert_called_once()
        assert pool.num_hosts == 0
        with pytest.raises(SandboxError, match="closed"):
            pool.create()

    def test_full_host_triggers_duplicate(self, monkeypatch) -> None:
        # First host fills at 1 sandbox (server-authoritative); the duplicate has room.
        # The 2nd create() must boot a second host when the first reports full.
        url1, _ = _spawn_fake(capacity=1)
        url2, _ = _spawn_fake(capacity=10)
        servers = iter([_make_server(url1, job_id="h0", capacity=2), _make_server(url2, job_id="h1", capacity=2)])
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", lambda self, **kw: [])
        monkeypatch.setattr(SandboxPool, "_boot_host", lambda self: next(servers))
        pool = SandboxPool(sandboxes_per_host=2, token="hf_test")  # constructor warms host h0
        boxes = [pool.create(), pool.create()]
        assert pool.num_hosts == 2  # had to boot a duplicate
        assert {b.id.split(".")[0] for b in boxes} == {"h0", "h1"}


class TestHostDiscovery:
    """`create()` should attach to a warm host found via job labels (e.g. left by
    another process) before booting a new one."""

    def _host_job(self, job_id: str = "host9", capacity: int = 4, pool_name: str = "p1") -> MagicMock:
        job = MagicMock()
        job.id = job_id
        job.owner.name = "user"
        job.docker_image = "python:3.12"
        job.space_id = None
        job.flavor = "cpu-basic"
        job.status.stage = "RUNNING"
        job.labels = {SANDBOX_LABEL: "nonce", HOST_LABEL: "1", POOL_LABEL: pool_name}
        job.environment = {"SBX_CAPACITY": str(capacity)}  # config lives in env vars, not labels
        return job

    def _pool(self, fake_server, monkeypatch, jobs, name: str = "p1", **kwargs) -> SandboxPool:
        # Patch discovery + boot before construction: the constructor warms up, adopting any
        # matching running host found via labels (a freshly booted host is only the fallback).
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", lambda self, **kw: jobs)
        # A new host boot would fail (no real Jobs); discovery must avoid it here.
        monkeypatch.setattr(SandboxPool, "_boot_host", lambda self: _make_server(fake_server, capacity=4))
        # `_connect_host` (module-level) returns a server wired to the fake server.
        monkeypatch.setattr(
            sandbox_mod, "_connect_host", lambda api, jid, namespace=None: _make_server(fake_server, job_id=jid)
        )
        return SandboxPool(image="python:3.12", flavor="cpu-basic", name=name, token="hf_test", **kwargs)

    def test_discovers_and_reuses_existing_host(self, fake_server: str, monkeypatch) -> None:
        pool = self._pool(fake_server, monkeypatch, jobs=[self._host_job("host9", capacity=4)])
        box = pool.create()
        assert isinstance(box, Sandbox)
        assert pool.num_hosts == 1
        assert pool._hosts[0].job_id == "host9"  # adopted by the constructor, not freshly booted
        assert pool._hosts[0].capacity == 4  # read from the host's env var
        assert box.host_id == "host9"

    def test_discovery_respects_pool_name(self, fake_server: str, monkeypatch) -> None:
        # A host from a different pool must not be adopted.
        pool = self._pool(fake_server, monkeypatch, jobs=[self._host_job("host9", pool_name="other")], name="mine")
        pool.create()
        assert pool._hosts[0].job_id != "host9"  # booted its own host (name mismatch)

    def test_discovery_falls_back_to_inspect_for_capacity(self, fake_server: str, monkeypatch) -> None:
        # When list_jobs omits the host env, capacity is fetched via inspect_job (like connect),
        # not silently defaulted to the pool's per-host setting.
        listed = self._host_job("host9", capacity=4)
        listed.environment = {}  # list_jobs omitted the env
        inspect = MagicMock(return_value=self._host_job("host9", capacity=4))
        monkeypatch.setattr(sandbox_mod.HfApi, "inspect_job", inspect)
        pool = self._pool(fake_server, monkeypatch, jobs=[listed])  # pool default per-host is 50
        assert pool._hosts[0].capacity == 4  # read from inspect_job's env, not the pool default
        inspect.assert_called_once()

    def test_adopts_scheduling_host_instead_of_booting(self, fake_server: str, monkeypatch) -> None:
        # A host already SCHEDULING for this pool (e.g. booted by another process) is waited
        # for and adopted, rather than piling on a duplicate.
        job = self._host_job("host-sched", capacity=4)
        job.status.stage = "SCHEDULING"

        def fake_inspect(self, **kwargs) -> MagicMock:
            job.status.stage = "RUNNING"  # the scheduling host has come up
            return job

        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", lambda self, **kw: [job])
        monkeypatch.setattr(sandbox_mod.HfApi, "inspect_job", fake_inspect)
        monkeypatch.setattr(
            sandbox_mod, "_connect_host", lambda api, jid, namespace=None: _make_server(fake_server, job_id=jid)
        )
        # `_connect_mode` skips the constructor warm-up so we exercise adoption in isolation.
        pool = SandboxPool(name="p1", token="hf_test", _connect_mode=True)
        assert pool._adopt_pending_host() is True
        assert [host.job_id for host in pool._hosts] == ["host-sched"]  # adopted, not booted


class TestPoolConnect:
    """`SandboxPool.connect(pool_id)` rebuilds a pool from a running host's job spec +
    env vars — no local state, no config endpoint — then packs onto that host."""

    def test_connect_reads_config_from_host_env(self, monkeypatch) -> None:
        url, _ = _spawn_fake(capacity=7)
        job = MagicMock()
        job.id = "hostA"
        job.owner.name = "user"
        job.status.stage = "RUNNING"
        job.docker_image = "alpine:3.20"
        job.space_id = None
        job.flavor = "cpu-basic"
        job.labels = {SANDBOX_LABEL: "n", HOST_LABEL: "1", POOL_LABEL: "pool-x"}
        job.environment = {"SBX_CAPACITY": "7", "SBX_IDLE_TIMEOUT": "600", "SBX_MAX_HOSTS": "3"}
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", lambda self, namespace=None: [job])
        monkeypatch.setattr(
            sandbox_mod, "_connect_host", lambda api, jid, namespace=None: _make_server(url, job_id=jid, capacity=7)
        )

        pool = SandboxPool.connect("pool-x", token="hf_test")
        assert pool.image == "alpine:3.20"
        assert pool.flavor == "cpu-basic"
        assert pool.sandboxes_per_host == 7
        assert pool.max_hosts == 3  # cost ceiling restored from the host env, not lost
        assert pool.name == "pool-x"

        assert pool._owns_hosts is False  # attached to shared hosts; close() must not kill them

        box = pool.create()  # packs onto the discovered host, no boot
        assert isinstance(box, Sandbox)
        assert box.host_id == "hostA"

    def test_connected_pool_close_leaves_hosts_running(self, fake_server: str, monkeypatch) -> None:
        # A connect()'d handle doesn't own the shared hosts: close()/`with` releases the local
        # HTTP client but must not cancel the host job (other clients may be using it).
        # `_connect_mode=True` mirrors connect(): no warm-up boot, hosts not owned.
        pool = SandboxPool(name="pool-x", token="hf_test", _connect_mode=True)
        assert pool._owns_hosts is False
        host = _make_server(fake_server, job_id="hostA")
        pool._hosts.append(host)
        pool.close()
        host._api.cancel_job.assert_not_called()  # host left running
        assert host._client.is_closed  # but the local client is released

    def test_connect_raises_when_pool_gone(self, monkeypatch) -> None:
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", lambda self, namespace=None: [])
        with pytest.raises(SandboxError, match="No running host found for pool"):
            SandboxPool.connect("pool-dead", token="hf_test")


def _save_cache(pool_id: str, hosts, **overrides) -> None:
    """Write a pool cache with sane defaults, overriding config fields as needed."""
    config = {
        "image": "python:3.12",
        "flavor": "cpu-basic",
        "sandboxes_per_host": 4,
        "max_hosts": None,
        "idle_timeout": 600,
        "namespace": None,
        **overrides,
    }
    save_pool_cache(pool_id, hosts=hosts, **config)


class TestPoolCacheFile:
    """Unit tests for the on-disk cache layout (`$HF_HOME/sandbox/pools/<id>.json`)."""

    def test_round_trip(self) -> None:
        _save_cache("p", [CachedHost("h1", "user", "http://h1", "n1", 4, 1)], namespace="ns")
        cache = read_pool_cache("p")
        assert cache is not None
        assert (cache.image, cache.sandboxes_per_host, cache.namespace) == ("python:3.12", 4, "ns")
        assert cache.hosts[0].job_id == "h1" and cache.hosts[0].live == 1

    def test_missing_returns_none(self) -> None:
        assert read_pool_cache("does-not-exist") is None

    def test_corrupt_returns_none(self) -> None:
        path = cache_mod.pool_cache_path("bad")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not valid json")
        assert read_pool_cache("bad") is None  # tolerated as a cache miss

    def test_version_mismatch_returns_none(self, monkeypatch) -> None:
        _save_cache("p", [CachedHost("h1", "user", "http://h1", "n1", 4, 0)])
        monkeypatch.setattr(cache_mod, "_CACHE_VERSION", 999)
        assert read_pool_cache("p") is None

    def test_merge_upserts_and_prunes(self) -> None:
        _save_cache(
            "p", [CachedHost("h1", "user", "http://h1", "n1", 4, 0), CachedHost("h2", "user", "u2", "n2", 4, 0)]
        )
        # A second writer updates h2's live count and reports h1 as dead.
        _save_cache("p", [CachedHost("h2", "user", "u2", "n2", 4, 3)], dead_host_ids={"h1"})
        cache = read_pool_cache("p")
        assert cache is not None
        assert {h.job_id for h in cache.hosts} == {"h2"}  # h1 pruned, h2 kept
        assert cache.hosts[0].live == 3  # updated value won

    def test_delete(self) -> None:
        _save_cache("p", [CachedHost("h1", "user", "http://h1", "n1", 4, 0)])
        cache_mod.delete_pool_cache("p")
        assert read_pool_cache("p") is None


class TestPoolCacheIntegration:
    """`SandboxPool.connect` + `create` use the cache to skip list_jobs/inspect_job when warm."""

    def test_warm_cache_packs_without_listing_jobs(self, fake_server: str, monkeypatch) -> None:
        # An earlier run left a warm host in the cache; a fresh pool must reach it with no HTTP
        # other than the create POST itself.
        _save_cache("pool-cached", [CachedHost("host-cached", "user", fake_server, "nonce", 4, 0)])
        monkeypatch.setattr(sandbox_mod, "_derive_sandbox_token", lambda *a: "secret")

        pool = SandboxPool.connect("pool-cached", token="hf_test")
        pool._api.list_jobs = MagicMock(side_effect=AssertionError("must not list jobs"))
        monkeypatch.setattr(pool, "_boot_host", lambda: pytest.fail("must not boot a host"))

        box = pool.create()
        assert isinstance(box, Sandbox) and box.host_id == "host-cached"
        pool._api.list_jobs.assert_not_called()
        cache = read_pool_cache("pool-cached")
        assert cache is not None and cache.hosts[0].live == 1  # live count persisted back

    def test_stale_host_falls_back_to_discovery_and_prunes(self, fake_server: str, monkeypatch) -> None:
        # The cached host is dead (refused connection); discovery finds a live one via labels.
        _save_cache("pool-stale", [CachedHost("dead", "user", "http://127.0.0.1:1", "n", 4, 0)])
        monkeypatch.setattr(sandbox_mod, "_derive_sandbox_token", lambda *a: "secret")
        pool = SandboxPool.connect("pool-stale", token="hf_test")

        live_job = MagicMock()
        live_job.id, live_job.flavor = "live", "cpu-basic"
        live_job.owner.name, live_job.docker_image, live_job.space_id = "user", "python:3.12", None
        live_job.status.stage = "RUNNING"
        live_job.labels = {SANDBOX_LABEL: "n", HOST_LABEL: "1", POOL_LABEL: "pool-stale"}
        live_job.environment = {"SBX_CAPACITY": "4"}
        pool._api.list_jobs = MagicMock(return_value=[live_job])
        monkeypatch.setattr(
            sandbox_mod, "_connect_host", lambda api, jid, namespace=None: _make_server(fake_server, job_id=jid)
        )

        box = pool.create()
        assert box.host_id == "live"
        cache = read_pool_cache("pool-stale")
        assert cache is not None and [h.job_id for h in cache.hosts] == ["live"]  # dead host pruned

    def test_stale_cache_does_not_resurrect_dead_pool(self, monkeypatch) -> None:
        # connect() trusted a stale cache, but every host is gone and labels find nothing:
        # create() must refuse to boot a fresh host under the same id, and clear the cache.
        _save_cache("pool-ghost", [CachedHost("dead", "user", "http://127.0.0.1:1", "n", 4, 0)])
        monkeypatch.setattr(sandbox_mod, "_derive_sandbox_token", lambda *a: "secret")
        pool = SandboxPool.connect("pool-ghost", token="hf_test")
        pool._api.list_jobs = MagicMock(return_value=[])
        monkeypatch.setattr(pool, "_boot_host", lambda: pytest.fail("must not resurrect the pool"))

        with pytest.raises(SandboxError, match="No running host found"):
            pool.create()
        assert read_pool_cache("pool-ghost") is None  # cleared
