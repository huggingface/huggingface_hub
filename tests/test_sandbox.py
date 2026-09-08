import json
import os
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import pytest

import huggingface_hub._sandbox as sandbox_mod
import huggingface_hub._sandbox_cache as cache_mod
from huggingface_hub._sandbox import (
    MODE_LABEL,
    MODE_POOL,
    NONCE_LABEL,
    POOL_LABEL,
    SANDBOX_LABEL,
    SANDBOX_SERVER_PORT,
    Sandbox,
    SandboxPool,
    _SandboxServer,
)
from huggingface_hub._sandbox_cache import (
    HOST_TRUST_TTL,
    CacheContext,
    CachedHost,
    read_pool_cache,
    save_pool_cache,
)
from huggingface_hub.errors import SandboxCommandError, SandboxError


# A job nonce shaped like the real thing (minted by `token_hex(16)`): the public label the
# sandbox token derives from, and a value the pool cache checks the shape of on read.
NONCE = "1e" * 16


def _fake_list_jobs(jobs):
    """Stand-in for `HfApi.list_jobs` that mimics the server-side `status`/`labels` filtering.

    The real endpoint filters by stage and by AND-matched `key=value` labels, so the pool's
    discovery code now passes those down instead of filtering client-side. The fake applies the
    same filtering so tests exercise the params we send (e.g. a host from another pool is excluded
    by the server, not by the client). Accepts an optional leading `self` so it works both as a
    class attribute (`monkeypatch.setattr(HfApi, "list_jobs", ...)`) and as a bound `MagicMock`.
    """

    def _list(self=None, *, status=None, labels=None, **kwargs):
        result = jobs
        if status is not None:
            wanted = {status} if isinstance(status, str) else set(status)
            wanted = {s.upper() for s in wanted}
            result = [job for job in result if job.status.stage in wanted]
        if labels:
            result = [job for job in result if all((job.labels or {}).get(k) == v for k, v in labels.items())]
        return result

    return _list


@pytest.fixture(autouse=True)
def _isolate_pool_cache(tmp_path, monkeypatch):
    """Point the best-effort pool cache at a throwaway dir so tests never touch ~/.cache."""
    monkeypatch.setattr(cache_mod.constants, "HF_HOME", str(tmp_path))


def _make_server(base_url: str, job_id: str = "job123", capacity: int = 0) -> _SandboxServer:
    """Build a _SandboxServer wired to a local fake server, bypassing job creation."""
    api = MagicMock()
    api.token = "hf_test"
    return _SandboxServer(
        job_id=job_id,
        owner="user",
        image="python:3.12",
        base_url=base_url,
        nonce=NONCE,
        sandbox_token="secret",
        api=api,
        capacity=capacity,
    )


def _booted_server(base_url: str, job_id: str = "job123", capacity: int = 0) -> _SandboxServer:
    """A server as `_boot_host` would return one: this process started the job.

    Stubs for `_boot_host` must set this, or `close()` treats the host as one
    someone else started and correctly refuses to cancel it.
    """
    server = _make_server(base_url, job_id=job_id, capacity=capacity)
    server.owned = True
    return server


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
    processes: list = []  # background processes started via /processes
    proc_seq = 0  # monotonic process id source
    # Every request that reached this server, as (method, path). A server that must never be
    # contacted at all (a URL only a cache file named, a redirect target) is asserted on this.
    requests: list = []
    writes: list = []  # (path, body) of every /files/write received
    redirect_to = ""  # where the /v1/redirect route points (set per-subclass)

    def log_message(self, *args) -> None:
        pass

    def _record(self) -> None:
        type(self).requests.append((self.command, self.path))

    def _ndjson(self, events) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for event in events:
            self.wfile.write((json.dumps(event) + "\n").encode())
            self.wfile.flush()

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _expect_token(self) -> None:
        """Mirror the real server's credential scoping.

        Management routes (`/v1/sandboxes`, token recovery) take the host token;
        every per-sandbox route takes that sandbox's own capability token. Keeping
        the fake as strict as the server is the point: a lax fake is how a client
        bug like addressing a process by pid survives a green test suite.
        """
        provided = self.headers["X-Sandbox-Token"]
        parts = self.path.split("?")[0].strip("/").split("/")
        scoped = len(parts) >= 3 and parts[:2] == ["v1", "sandboxes"] and parts[3:4] != ["token"]
        expected = f"tok-{parts[2]}" if scoped else "secret"
        assert provided == expected, f"{self.path}: expected {expected!r}, got {provided!r}"

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
        self._record()
        self._expect_token()
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        cls = type(self)
        # exec (dedicated /v1/exec or shared /v1/sandboxes/<id>/exec)
        if self.path == "/v1/exec" or (self.path.startswith("/v1/sandboxes/") and self.path.endswith("/exec")):
            self._exec(body)
        elif self.path.endswith("/processes"):  # spawn a background process
            type(self).last_exec = body
            proc = {
                "id": f"p-{cls.proc_seq}",
                "pid": 9000 + cls.proc_seq,
                "tag": body.get("tag"),
                "cmd": body["cmd"],
                "started_at_ms": 1_700_000_000_000 + cls.proc_seq,
                "running": True,
                "exit_code": None,
            }
            cls.proc_seq += 1
            cls.processes.append(proc)
            self._json({"id": proc["id"], "pid": proc["pid"], "tag": proc["tag"]})
        elif self.path == "/v1/sandboxes":  # batch-create sandboxes (server-authoritative capacity)
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
                created.append({"id": sid, "token": f"tok-{sid}"})
            self._json({"sandboxes": created, "rejected": rejected})

    def do_PUT(self) -> None:
        self._record()
        self._expect_token()
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).writes.append((self.path, body))
        self._json({"written": len(body)})

    def do_DELETE(self) -> None:
        self._record()
        self._expect_token()
        last = self.path.rsplit("/", 1)[-1]
        if "/processes/" in self.path:  # kill a background process
            # Deletion is by opaque id only, and a pid is a 400 -- exactly like the
            # real server. The fake used to delete by pid, which is what let the
            # client's `kill()` send a pid and still pass every test.
            if not (last.startswith("p-") and last[2:].isdigit()):
                self._json({"error": f"{last!r} is not a process id"}, status=400)
                return
            before = len(type(self).processes)
            type(self).processes = [p for p in type(self).processes if p["id"] != last]
            self._json({"id": last, "killed": len(type(self).processes) != before})
            return
        type(self).sandboxes.discard(last)
        self._json({"id": last, "deleted": True})

    def do_GET(self) -> None:
        self._record()
        self._expect_token()
        cls = type(self)
        if self.path == "/v1/redirect":  # a redirect the client must not follow
            self.send_response(302)
            self.send_header("Location", cls.redirect_to)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path.startswith("/v1/files/stat") or "/files/stat" in self.path:
            self._json({"name": "x", "path": "/x", "type": "file", "size": 5})
        elif self.path.startswith("/v1/files/read") or "/files/read" in self.path:
            self.send_response(200)
            self.send_header("Content-Length", "5")
            self.end_headers()
            self.wfile.write(b"hello")
        elif "/files/list" in self.path:
            # Paginated, like the real server: `list_pages` lets a test hand out a
            # cursor and check the client follows it.
            pages = getattr(cls, "list_pages", None)
            if pages:
                after = ""
                if "after=" in self.path:
                    after = self.path.split("after=")[1].split("&")[0]
                page = pages[0] if not after else next((p for p in pages if p.get("_after") == after), pages[-1])
                self._json({"entries": page["entries"], "next": page.get("next")})
            else:
                self._json({"entries": [], "next": None})
        elif self.path.endswith("/processes"):  # list background processes
            self._json(cls.processes)
        elif self.path == "/v1/sandboxes":
            self._json([{"id": sid} for sid in sorted(cls.sandboxes)])
        elif self.path.startswith("/v1/sandboxes/") and self.path.endswith("/token"):
            sid = self.path.split("/")[3]
            self._json({"id": sid, "token": f"tok-{sid}"})


@pytest.fixture()
def fake_server():
    _FakeServer.sandboxes = set()
    _FakeServer.seq = 0
    _FakeServer.capacity = None
    _FakeServer.last_exec = None
    _FakeServer.processes = []
    _FakeServer.proc_seq = 0
    _FakeServer.requests = []
    _FakeServer.writes = []
    _FakeServer.redirect_to = ""
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
        processes: list = []
        proc_seq = 0
        requests: list = []
        writes: list = []

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

    def test_run_background_returns_process(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        process = sandbox.run("python -m http.server 8000", background=True)
        assert isinstance(process, sandbox_mod.SandboxProcess)
        assert process.pid == 9000
        assert process.cmd == "python -m http.server 8000"
        assert process.running is True
        # background spawn doesn't stream/wait: the cmd/shell payload is POSTed as-is.
        assert _FakeServer.last_exec == {"cmd": "python -m http.server 8000"}

    def test_processes_list_and_kill(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        sandbox.run(["sleep", "100"], background=True)
        sandbox.run("sleep 200", background=True)
        processes = sandbox.processes()
        assert [p.pid for p in processes] == [9000, 9001]
        assert processes[0].cmd == ["sleep", "100"]
        # status fields from the listing are carried through.
        assert processes[0].running is True
        assert processes[0].exit_code is None
        assert processes[0].started_at_ms == 1_700_000_000_000
        processes[0].kill()
        assert [p.pid for p in sandbox.processes()] == [9001]

    def test_proxy_url_for(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        host = fake_server.split("://", 1)[-1]
        # default scheme is https; the in-sandbox port and path are appended under /v1/proxy.
        assert sandbox.proxy_url_for(8000, "/hello") == f"https://{host}/v1/proxy/8000/hello"
        # a path without a leading slash is normalized.
        assert sandbox.proxy_url_for(8000, "ws").endswith("/v1/proxy/8000/ws")
        # the scheme is swapped in for WebSocket clients (host/path unchanged).
        assert sandbox.proxy_url_for(8000, "/ws", scheme="wss://") == f"wss://{host}/v1/proxy/8000/ws"
        assert sandbox.proxy_headers["X-Sandbox-Token"] == "secret"

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
            sandbox_mod.Sandbox, "connect", classmethod(lambda cls, sid, namespace=None, token=None: connected)
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


class TestResourceBounds:
    """Client-side memory bounds. Each of these was unbounded: the caller's
    process grew with whatever the sandbox produced, whether or not it was
    already consuming it."""

    def test_output_is_not_accumulated_when_the_caller_opts_out(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        chunks: list = []
        result = sandbox.run("echo", on_stdout=chunks.append, capture_output=False)

        # The callback still sees everything; the result deliberately holds nothing.
        assert chunks == ["out1"]
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exit_code == 0

    def test_output_is_accumulated_by_default(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        assert sandbox.run("echo").stdout == "out1"

    def test_runaway_output_raises_instead_of_growing_without_bound(self, fake_server: str, monkeypatch) -> None:
        # Lowered so the test need not actually produce 64 MB.
        monkeypatch.setattr(sandbox_mod, "MAX_CAPTURED_OUTPUT_CHARS", 2)
        sandbox = _make_sandbox(fake_server)
        with pytest.raises(SandboxError, match="capture_output=False"):
            sandbox.run("echo")
        # ...and the escape hatch the error names actually works.
        assert sandbox.run("echo", capture_output=False).stdout == ""

    def test_read_refuses_a_file_too_large_to_hold_in_memory(self, fake_server: str, monkeypatch) -> None:
        monkeypatch.setattr(sandbox_mod.SandboxFiles, "MAX_READ_BYTES", 1)
        sandbox = _make_sandbox(fake_server)
        with pytest.raises(SandboxError, match="files.download"):
            sandbox.files.read("big.bin")

    def test_list_follows_the_servers_pagination(self, fake_server: str) -> None:
        # The server pages; a caller should still see one list.
        sandbox = _make_sandbox(fake_server)
        _FakeServer.list_pages = [
            {"entries": [{"name": "a", "path": "/a", "type": "file", "size": 1}], "next": "a"},
            {"_after": "a", "entries": [{"name": "b", "path": "/b", "type": "file", "size": 1}]},
        ]
        try:
            assert [entry.name for entry in sandbox.files.list("/dir")] == ["a", "b"]
        finally:
            _FakeServer.list_pages = None


class TestBackgroundProcesses:
    """`kill()` has to address a process the way the server identifies it. It used
    to send the OS pid, which matches nothing server-side, and the server answered
    200 anyway -- so a process the user asked to stop kept running, kept the
    sandbox non-idle, and kept the job billing."""

    def test_kill_uses_the_server_assigned_id(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        process = sandbox.run("sleep 60", background=True)

        assert process.id == "p-0"
        assert process.pid == 9000  # still exposed, for correlating with `ps`
        # The fake rejects a pid with a 400, exactly like the real server, so this
        # passing is evidence the opaque id was sent.
        assert process.kill() is True
        assert sandbox.processes() == []

    def test_kill_is_idempotent_and_says_whether_it_did_anything(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        process = sandbox.run("sleep 60", background=True)

        assert process.kill() is True
        assert process.kill() is False  # already gone: not an error

    def test_listed_processes_carry_their_id(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        sandbox.run("sleep 60", background=True)
        listed = sandbox.processes()

        assert [p.id for p in listed] == ["p-0"]
        assert listed[0].kill() is True

    def test_sending_a_pid_is_refused_by_the_server(self, fake_server: str) -> None:
        # Guards the fake as much as the client: if this ever passes, the fake has
        # gone lax again and the original bug could return unnoticed.
        sandbox = _make_sandbox(fake_server)
        process = sandbox.run("sleep 60", background=True)
        with pytest.raises(SandboxError):
            sandbox._request("DELETE", f"/processes/{process.pid}")

    def test_a_process_without_an_id_refuses_to_be_killed(self, fake_server: str) -> None:
        # A host running a server that predates opaque ids issues no id. Better a
        # clear error than a pid the server will reject.
        sandbox = _make_sandbox(fake_server)
        process = sandbox_mod.SandboxProcess(id=None, pid=9001, cmd="sleep 60", _sandbox=sandbox)
        with pytest.raises(SandboxError, match="predates opaque process ids"):
            process.kill()


class TestSharedSandbox:
    """A shared sandbox routes operations under /v1/sandboxes/<local_id>/ and is
    terminated with a DELETE on the host (the host job keeps running)."""

    def _make_shared(self, base_url: str) -> Sandbox:
        server = _make_server(base_url, capacity=10)
        _FakeServer.sandboxes.add("local1")
        return Sandbox(
            id="job123.local1",
            server=server,
            local_id="local1",
            owns_sandbox=True,
            owns_server=False,
            # What `SandboxPool.create()` would have received in the create response.
            sandbox_token="tok-local1",
        )

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

    def test_per_sandbox_operations_present_the_sandbox_token(self, fake_server: str) -> None:
        # The fake asserts the scoping itself (host token on management routes, the
        # sandbox's own token on scoped ones), so any operation reaching the server
        # is evidence the narrow credential was sent.
        sandbox = self._make_shared(fake_server)
        assert sandbox.run("echo").stdout == "out1"
        assert sandbox.files.read_text("f") == "hello"
        assert sandbox.processes() == []

    def test_proxy_headers_carry_the_sandbox_token_not_the_host_one(self, fake_server: str) -> None:
        # These headers are handed to browsers and WebSocket clients, so they must
        # not confer authority over the pool or over sibling sandboxes.
        sandbox = self._make_shared(fake_server)
        assert sandbox.proxy_headers["X-Sandbox-Token"] == "tok-local1"
        assert sandbox.proxy_headers["X-Sandbox-Token"] != sandbox._server._sandbox_token

    def test_falls_back_to_the_host_token_on_an_older_server(self, fake_server: str) -> None:
        # A host running a server that predates per-sandbox tokens returns no token
        # in the create response; those sandboxes keep working with the host one.
        server = _make_server(fake_server, capacity=10)
        _FakeServer.sandboxes.add("local1")
        sandbox = Sandbox(id="job123.local1", server=server, local_id="local1", owns_sandbox=True, owns_server=False)
        assert sandbox._sandbox_token is None
        assert sandbox.proxy_headers["X-Sandbox-Token"] == "secret"


class TestSandboxPool:
    def _pool(self, fake_server: str, monkeypatch, per_host: int = 4) -> SandboxPool:
        # Patch before construction: the constructor warms up `warm_up` (=1) host(s), so the
        # fake boot + empty discovery must already be in place. No warm hosts to discover here;
        # discovery is covered separately. Every host boot returns a server at the fake server.
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", lambda self, **kw: [])
        monkeypatch.setattr(SandboxPool, "_boot_host", lambda self: _booted_server(fake_server, capacity=per_host))
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
        monkeypatch.setattr(SandboxPool, "_boot_host", lambda self: _booted_server(fake_server, capacity=4))
        pool = SandboxPool(image="python:3.12", sandboxes_per_host=4, warm_up=3, token="hf_test")
        assert pool.num_hosts == 3  # pre-provisioned by the constructor, before any create()
        pool.create()
        assert pool.num_hosts == 3  # still 3 despite only one sandbox created
        assert pool.num_sandboxes == 1

    def test_max_hosts_enforced(self, fake_server: str, monkeypatch) -> None:
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", lambda self, **kw: [])
        monkeypatch.setattr(SandboxPool, "_boot_host", lambda self: _booted_server(fake_server, capacity=2))
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


# Id `whoami()` returns in these tests; a host must be started by this principal
# to be adopted under the default policy.
PRINCIPAL_ID = "principal-self"


@pytest.fixture(autouse=True)
def _principal(monkeypatch):
    """Resolve the calling principal without a network call.

    Host adoption compares a Job's backend-asserted initiator against this, so
    every test that discovers a host needs it.
    """
    monkeypatch.setattr(sandbox_mod.HfApi, "whoami", lambda self, **kwargs: {"id": PRINCIPAL_ID})


def _pool_host_job(
    job_id: str = "host9",
    *,
    capacity: int = 4,
    pool_name: str = "p1",
    image: str = "python:3.12",
    env: dict | None = None,
) -> MagicMock:
    """A Job as the Jobs API would really describe one of our pool hosts.

    Everything beyond the labels is asserted by the backend and checked by the
    client before a credential is sent, so a fixture that set only labels would
    let an adoption bug through unnoticed.
    """
    job = MagicMock()
    job.id = job_id
    job.owner.name = "user"
    job.initiator.id = PRINCIPAL_ID
    job.initiator.name = "user"
    job.docker_image = image
    job.space_id = None
    job.flavor = "cpu-basic"
    job.command = ["/bin/sh", "-c", sandbox_mod._BOOTSTRAP_DOWNLOAD]
    job.status.stage = "RUNNING"
    job.status.expose_urls = [f"https://{job_id}--{sandbox_mod.SANDBOX_SERVER_PORT}.hf.jobs"]
    job.labels = {SANDBOX_LABEL: "1", MODE_LABEL: MODE_POOL, POOL_LABEL: pool_name, NONCE_LABEL: NONCE}
    job.environment = {"SBX_CAPACITY": str(capacity)} if env is None else env
    return job


class TestPoolLifecycle:
    """Who owns a host, and whether teardown tells the truth. Both used to be
    decided per pool *handle*, which meant a `with` block could cancel a host it
    had merely discovered -- killing another process' (or another person's)
    sandboxes -- and a failed cancellation was logged and then reported as
    success while the job kept billing."""

    def _pool(self, fake_server: str, monkeypatch, jobs=(), **kwargs) -> SandboxPool:
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", _fake_list_jobs(list(jobs)))
        monkeypatch.setattr(
            sandbox_mod.SandboxPool, "_boot_host", lambda self: _booted_server(fake_server, capacity=4)
        )
        monkeypatch.setattr(
            sandbox_mod, "_connect_host", lambda api, jid, namespace=None: _make_server(fake_server, job_id=jid)
        )
        return SandboxPool(image="python:3.12", flavor="cpu-basic", name="p1", token="hf_test", **kwargs)

    def test_a_host_we_booted_is_cancelled(self, fake_server: str, monkeypatch) -> None:
        pool = self._pool(fake_server, monkeypatch)
        host = pool._hosts[0]
        pool.close()
        host._api.cancel_job.assert_called_once()

    def test_a_discovered_host_is_released_not_cancelled(self, fake_server: str, monkeypatch) -> None:
        # It may be serving another process' sandboxes. Releasing the local HTTP
        # client is this handle's business; terminating the job is not.
        discovered = _pool_host_job("theirs")
        pool = self._pool(fake_server, monkeypatch, jobs=[discovered])
        adopted = next(host for host in pool._hosts if host.job_id == "theirs")

        pool.close()
        adopted._api.cancel_job.assert_not_called()
        assert adopted._client.is_closed

    def test_a_failed_cancellation_raises_and_keeps_the_host_discoverable(self, fake_server: str, monkeypatch) -> None:
        pool = self._pool(fake_server, monkeypatch)
        host = pool._hosts[0]
        host._api.cancel_job.side_effect = RuntimeError("backend said no")

        with pytest.raises(SandboxError, match="still running and billing"):
            pool.close()
        # The cache entry is the caller's only handle on a job that is still
        # billing, so it must survive -- deleting it was the old behaviour.
        cache = read_pool_cache("p1", pool._cache_context)
        assert cache is not None and [entry.job_id for entry in cache.hosts] == [host.job_id]

    def test_a_teardown_failure_does_not_mask_the_original_exception(self, fake_server: str, monkeypatch) -> None:
        pool = self._pool(fake_server, monkeypatch)
        pool._hosts[0]._api.cancel_job.side_effect = RuntimeError("backend said no")

        with pytest.raises(ValueError, match="the real problem"):
            with pool:
                raise ValueError("the real problem")

    def test_close_waits_for_an_in_flight_create(self, fake_server: str, monkeypatch) -> None:
        # A create() already past its closed check must not leave a host behind:
        # close() waits for it, so whatever it booted is torn down.
        pool = self._pool(fake_server, monkeypatch)
        released = threading.Event()
        booted: list = []

        def slow_boot(self) -> _SandboxServer:
            released.wait(5)
            server = _booted_server(fake_server, job_id="slow", capacity=4)
            booted.append(server)
            return server

        monkeypatch.setattr(sandbox_mod.SandboxPool, "_boot_host", slow_boot)
        # Fill the warm host so the next create() has to boot.
        for host in pool._hosts:
            host.live = host.capacity

        creating = threading.Thread(target=lambda: pool.create())
        creating.start()
        time.sleep(0.2)  # let create() get past the closed check and into the boot
        closing = threading.Thread(target=lambda: pool.close())
        closing.start()
        time.sleep(0.2)
        released.set()
        creating.join(10)
        closing.join(10)

        assert booted, "the slow boot never happened, so this proves nothing"
        booted[0]._api.cancel_job.assert_called_once()

    def test_max_hosts_counts_every_host_running_for_the_pool(self, fake_server: str, monkeypatch) -> None:
        # `max_hosts` is a cost ceiling, and it was compared against an
        # in-process count -- so another process' hosts did not count towards it.
        theirs = _pool_host_job("theirs")
        pool = self._pool(fake_server, monkeypatch, jobs=[theirs], max_hosts=1)
        # The discovered host already fills the cap, so booting another must refuse.
        for host in pool._hosts:
            host.live = host.capacity
        with pytest.raises(SandboxError, match="max_hosts"):
            pool.create()


class TestHostDiscovery:
    """`create()` should attach to a warm host found via job labels (e.g. left by
    another process) before booting a new one."""

    def _host_job(self, job_id: str = "host9", capacity: int = 4, pool_name: str = "p1") -> MagicMock:
        return _pool_host_job(job_id, capacity=capacity, pool_name=pool_name)

    def _pool(self, fake_server, monkeypatch, jobs, name: str = "p1", **kwargs) -> SandboxPool:
        # Patch discovery + boot before construction: the constructor warms up, adopting any
        # matching running host found via labels (a freshly booted host is only the fallback).
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", _fake_list_jobs(jobs))
        # A new host boot would fail (no real Jobs); discovery must avoid it here.
        monkeypatch.setattr(SandboxPool, "_boot_host", lambda self: _booted_server(fake_server, capacity=4))
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

        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", _fake_list_jobs([job]))
        monkeypatch.setattr(sandbox_mod.HfApi, "inspect_job", fake_inspect)
        monkeypatch.setattr(
            sandbox_mod, "_connect_host", lambda api, jid, namespace=None: _make_server(fake_server, job_id=jid)
        )
        # `_connect_mode` skips the constructor warm-up so we exercise adoption in isolation.
        pool = SandboxPool(name="p1", token="hf_test", _connect_mode=True)
        assert pool._adopt_pending_host() is True
        assert [host.job_id for host in pool._hosts] == ["host-sched"]  # adopted, not booted


class TestHostAdmission:
    """A Job carrying our pool's labels is a *claim* to be one of our hosts, not
    proof: labels are set by whoever creates the Job, and the nonce that derives
    the host token is a public label. These assert that everything else about the
    Job -- all of it backend-asserted -- has to line up before a credential is
    sent to it."""

    def _reject(self, job, **kwargs) -> str | None:
        defaults = dict(
            policy=sandbox_mod.ADOPT_OWN,
            principal_id=PRINCIPAL_ID,
            namespace=None,
            image="python:3.12",
            flavor="cpu-basic",
        )
        defaults.update(kwargs)
        return sandbox_mod._host_rejection(job, **defaults)

    def test_a_genuine_host_is_accepted(self) -> None:
        assert self._reject(_pool_host_job()) is None

    def test_a_job_started_by_someone_else_is_refused(self) -> None:
        # The attack: a namespace member creates a Job with our pool's labels and
        # the real host's nonce, and our client sends it the real host's token.
        impostor = _pool_host_job("impostor")
        impostor.initiator.id = "principal-attacker"
        impostor.initiator.name = "attacker"
        assert "different principal" in (self._reject(impostor) or "")
        # `initiator` is the only field here the Jobs API does not let a client
        # set, which is what makes this check worth anything.

    def test_an_unknown_initiator_is_refused_rather_than_assumed(self) -> None:
        job = _pool_host_job()
        job.initiator = None
        assert self._reject(job) is not None
        assert self._reject(_pool_host_job(), principal_id=None) is not None

    def test_namespace_policy_accepts_a_sibling_member_but_still_checks_the_spec(self) -> None:
        other = _pool_host_job("other")
        other.initiator.id = "principal-colleague"
        assert self._reject(other, policy=sandbox_mod.ADOPT_NAMESPACE) is None
        # Opting into sharing does not opt out of the rest.
        other.docker_image = "evil:latest"
        assert self._reject(other, policy=sandbox_mod.ADOPT_NAMESPACE) is not None

    def test_never_policy_refuses_everything(self) -> None:
        assert self._reject(_pool_host_job(), policy=sandbox_mod.ADOPT_NEVER) is not None

    def test_a_mismatched_spec_is_refused(self) -> None:
        wrong_image = _pool_host_job()
        wrong_image.docker_image = "attacker/image:latest"
        assert "image" in (self._reject(wrong_image) or "")

        wrong_flavor = _pool_host_job()
        wrong_flavor.flavor = "a10g-large"
        assert "flavor" in (self._reject(wrong_flavor) or "")

        # A Job labelled as a host but running something else is not a host.
        wrong_command = _pool_host_job()
        wrong_command.command = ["/bin/sh", "-c", "nc -l -p 49983"]
        assert "bootstrap" in (self._reject(wrong_command) or "")

        wrong_namespace = _pool_host_job()
        wrong_namespace.owner.name = "someone-else"
        assert "namespace" in (self._reject(wrong_namespace, namespace="mine") or "")

    def test_registry_prefixes_do_not_cause_a_spurious_mismatch(self) -> None:
        # Image names get normalized server-side; a mis-parse here must not stop a
        # legitimate pool from working.
        job = _pool_host_job()
        job.docker_image = "docker.io/library/Python:3.12"
        assert self._reject(job) is None

    def test_the_exposed_url_must_belong_to_this_job(self) -> None:
        # Otherwise a Job could simply name where the credentials should go.
        elsewhere = _pool_host_job("victim")
        elsewhere.status.expose_urls = ["https://attacker--49983.hf.jobs"]
        assert self._reject(elsewhere) is not None

        insecure = _pool_host_job("plain")
        insecure.status.expose_urls = ["http://plain--49983.hf.jobs"]
        assert "https" in (self._reject(insecure) or "")

        extra = _pool_host_job("extra")
        extra.status.expose_urls = [
            "https://extra--49983.hf.jobs",
            "https://extra--8080.hf.jobs",
        ]
        assert self._reject(extra) is not None

        none_exposed = _pool_host_job("bare")
        none_exposed.status.expose_urls = None
        assert self._reject(none_exposed) is not None

    def test_discovery_does_not_adopt_an_impostor(self, fake_server: str, monkeypatch) -> None:
        # End to end through `create()`: the impostor is listed and matches on
        # labels, and must not be adopted -- the pool boots its own host instead.
        impostor = _pool_host_job("impostor")
        impostor.initiator.id = "principal-attacker"
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", _fake_list_jobs([impostor]))
        monkeypatch.setattr(
            sandbox_mod.SandboxPool, "_boot_host", lambda self: _booted_server(fake_server, job_id="mine", capacity=4)
        )
        monkeypatch.setattr(
            sandbox_mod, "_connect_host", lambda api, jid, namespace=None: _make_server(fake_server, job_id=jid)
        )
        pool = SandboxPool(image="python:3.12", flavor="cpu-basic", name="p1", token="hf_test")

        assert [host.job_id for host in pool._hosts] == ["mine"]
        assert pool.create().host_id == "mine"

    def test_connect_refuses_a_pool_whose_only_host_is_an_impostor(self, monkeypatch) -> None:
        impostor = _pool_host_job("impostor", pool_name="pool-x")
        impostor.initiator.id = "principal-attacker"
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", _fake_list_jobs([impostor]))
        with pytest.raises(SandboxError, match="none usable"):
            SandboxPool.connect("pool-x", token="hf_test")
        # The error names the opt-in, so a legitimate shared-host user is not stuck.
        with pytest.raises(SandboxError, match="none usable") as exc_info:
            SandboxPool.connect("pool-x", token="hf_test")
        assert "adopt_hosts='namespace'" in str(exc_info.value)

    def test_an_invalid_policy_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="adopt_hosts"):
            SandboxPool(name="p", adopt_hosts="anything", token="hf_test")


class TestPoolConnect:
    """`SandboxPool.connect(pool_id)` rebuilds a pool from a running host's job spec +
    env vars — no local state, no config endpoint — then packs onto that host."""

    def test_connect_reads_config_from_host_env(self, monkeypatch) -> None:
        url, _ = _spawn_fake(capacity=7)
        job = _pool_host_job(
            "hostA",
            pool_name="pool-x",
            image="alpine:3.20",
            env={"SBX_CAPACITY": "7", "SBX_IDLE_TIMEOUT": "600", "SBX_MAX_HOSTS": "3"},
        )
        job.labels = {SANDBOX_LABEL: "1", MODE_LABEL: MODE_POOL, POOL_LABEL: "pool-x", NONCE_LABEL: "n"}
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", _fake_list_jobs([job]))
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
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", _fake_list_jobs([]))
        with pytest.raises(SandboxError, match="No running host found for pool"):
            SandboxPool.connect("pool-dead", token="hf_test")


def _ctx(namespace: str | None = None, *, token: str = "hf_test", endpoint: str | None = None) -> CacheContext:
    """The cache context a pool with this endpoint / credential / namespace reads and writes under."""
    return sandbox_mod._cache_context(sandbox_mod.HfApi(token=token, endpoint=endpoint), namespace)


def _cached_host(
    job_id: str = "h1",
    *,
    capacity: int = 4,
    live: int = 0,
    url: str | None = None,
    nonce: str = NONCE,
    owner: str = "user",
    age: float = 0.0,
) -> CachedHost:
    """A cache entry as a real run would have written it, `age` seconds ago.

    The default URL is the one the entry's own job would expose, because that is the only
    kind the client will rebuild a transport from -- a fixture naming anything else would be
    testing a code path that no longer exists.
    """
    return CachedHost(
        job_id=job_id,
        owner=owner,
        base_url=url if url is not None else f"https://{job_id}--{SANDBOX_SERVER_PORT}.hf.jobs",
        nonce=nonce,
        capacity=capacity,
        live=live,
        updated_at=time.time() - age,
    )


def _save_cache(pool_id: str, hosts, *, context: CacheContext | None = None, **overrides) -> None:
    """Write a pool cache with sane defaults, overriding config fields as needed."""
    config = {
        "image": "python:3.12",
        "flavor": "cpu-basic",
        "sandboxes_per_host": 4,
        "max_hosts": None,
        "idle_timeout": 600,
        **overrides,
    }
    save_pool_cache(pool_id, context=context if context is not None else _ctx(), hosts=hosts, **config)


@pytest.fixture()
def loopback_cache_hosts(monkeypatch):
    """Let a cached host point at a local fake server.

    A cached URL is only admitted if it is the HTTPS jobs-proxy URL of the entry's own job,
    which a loopback test server can never be (that check is exercised directly in
    `TestCachedHostAdmission`). The tests using this fixture are about what the cache does
    once a host *is* seeded, so they opt out of that one check and nothing else.
    """
    monkeypatch.setattr(sandbox_mod, "_cached_host_rejection", lambda host: None)


class TestPoolCacheFile:
    """Unit tests for the on-disk cache layout (`$HF_HOME/sandbox/pools/<context>/<id>.json`)."""

    def test_round_trip(self) -> None:
        _save_cache("p", [_cached_host("h1", live=1)], context=_ctx("ns"))
        cache = read_pool_cache("p", _ctx("ns"))
        assert cache is not None
        assert (cache.image, cache.sandboxes_per_host, cache.namespace) == ("python:3.12", 4, "ns")
        assert cache.hosts[0].job_id == "h1" and cache.hosts[0].live == 1

    def test_missing_returns_none(self) -> None:
        assert read_pool_cache("does-not-exist", _ctx()) is None

    def test_path_rejects_traversal(self) -> None:
        for bad in ("../evil", "a/b", "..", "x\x00y"):
            with pytest.raises(ValueError):
                cache_mod.pool_cache_path(bad, _ctx())
        # read/save/delete stay best-effort (no raise) even for an invalid id.
        assert read_pool_cache("../evil", _ctx()) is None
        _save_cache("../evil", [_cached_host()])  # no raise
        cache_mod.delete_pool_cache("../evil")  # no raise

    def test_corrupt_returns_none(self) -> None:
        path = cache_mod.pool_cache_path("bad", _ctx())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not valid json")
        assert read_pool_cache("bad", _ctx()) is None  # tolerated as a cache miss

    def test_version_mismatch_returns_none(self, monkeypatch) -> None:
        _save_cache("p", [_cached_host()])
        monkeypatch.setattr(cache_mod, "_CACHE_VERSION", 999)
        assert read_pool_cache("p", _ctx()) is None

    def test_merge_upserts_and_prunes(self) -> None:
        _save_cache("p", [_cached_host("h1"), _cached_host("h2")])
        # A second writer updates h2's live count and reports h1 as dead.
        _save_cache("p", [_cached_host("h2", live=3)], dead_host_ids={"h1"})
        cache = read_pool_cache("p", _ctx())
        assert cache is not None
        assert {h.job_id for h in cache.hosts} == {"h2"}  # h1 pruned, h2 kept
        assert cache.hosts[0].live == 3  # updated value won

    def test_delete(self) -> None:
        _save_cache("p", [_cached_host()])
        cache_mod.delete_pool_cache("p", _ctx())
        assert read_pool_cache("p", _ctx()) is None

    def test_delete_without_a_context_clears_every_context(self) -> None:
        # `hf sandbox pool delete` knows the pool id, not which credential cached it. Over-
        # deleting a disposable cache is the safe direction, under-deleting leaves a stale entry.
        _save_cache("p", [_cached_host()], context=_ctx("org-a"))
        _save_cache("p", [_cached_host()], context=_ctx("org-b"))
        cache_mod.delete_pool_cache("p")
        assert read_pool_cache("p", _ctx("org-a")) is None
        assert read_pool_cache("p", _ctx("org-b")) is None

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
    def test_the_cache_is_private_to_this_user(self) -> None:
        # It holds reusable host URLs and the public nonces the host tokens derive from.
        _save_cache("p", [_cached_host()])
        path = cache_mod.pool_cache_path("p", _ctx())
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700  # the context dir
        assert stat.S_IMODE(path.parent.parent.stat().st_mode) == 0o700  # sandbox/pools
        # No temp file left behind (and none with a name another process could have guessed).
        assert sorted(p.name for p in path.parent.iterdir()) == ["p.json", "p.json.lock"]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
    def test_a_lax_pre_existing_directory_is_tightened(self) -> None:
        directory = cache_mod._pools_dir()
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o777)  # e.g. written by an older version, under a lax umask
        _save_cache("p", [_cached_host()])
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700


class TestPoolCacheIntegration:
    """`SandboxPool.connect` + `create` use the cache to skip list_jobs/inspect_job when warm."""

    def test_warm_cache_packs_without_listing_jobs(self, fake_server: str, monkeypatch, loopback_cache_hosts) -> None:
        # An earlier run left a warm host in the cache; a fresh pool must reach it with no HTTP
        # other than the create POST itself.
        _save_cache("pool-cached", [_cached_host("host-cached", url=fake_server)])
        monkeypatch.setattr(sandbox_mod, "_derive_sandbox_token", lambda *a: "secret")

        pool = SandboxPool.connect("pool-cached", token="hf_test")
        pool._api.list_jobs = MagicMock(side_effect=AssertionError("must not list jobs"))
        pool._api.inspect_job = MagicMock(side_effect=AssertionError("must not inspect the host job"))
        monkeypatch.setattr(pool, "_boot_host", lambda: pytest.fail("must not boot a host"))

        box = pool.create()
        assert isinstance(box, Sandbox) and box.host_id == "host-cached"
        pool._api.list_jobs.assert_not_called()
        pool._api.inspect_job.assert_not_called()  # a fresh entry costs no extra round-trip
        assert _FakeServer.requests == [("POST", "/v1/sandboxes")]  # ...and exactly one request
        cache = read_pool_cache("pool-cached", _ctx())
        assert cache is not None and cache.hosts[0].live == 1  # live count persisted back

    def test_stale_host_falls_back_to_discovery_and_prunes(
        self, fake_server: str, monkeypatch, loopback_cache_hosts
    ) -> None:
        # The cached host is dead (refused connection); discovery finds a live one via labels.
        _save_cache("pool-stale", [_cached_host("dead", url="http://127.0.0.1:1")])
        monkeypatch.setattr(sandbox_mod, "_derive_sandbox_token", lambda *a: "secret")
        pool = SandboxPool.connect("pool-stale", token="hf_test")

        live_job = _pool_host_job("live", pool_name="pool-stale")
        pool._api.list_jobs = MagicMock(return_value=[live_job])
        monkeypatch.setattr(
            sandbox_mod, "_connect_host", lambda api, jid, namespace=None: _make_server(fake_server, job_id=jid)
        )

        box = pool.create()
        assert box.host_id == "live"
        cache = read_pool_cache("pool-stale", _ctx())
        assert cache is not None and [h.job_id for h in cache.hosts] == ["live"]  # dead host pruned

    def test_stale_cache_does_not_resurrect_dead_pool(self, monkeypatch, loopback_cache_hosts) -> None:
        # connect() trusted a stale cache, but every host is gone and labels find nothing:
        # create() must refuse to boot a fresh host under the same id, and clear the cache.
        _save_cache("pool-ghost", [_cached_host("dead", url="http://127.0.0.1:1")])
        monkeypatch.setattr(sandbox_mod, "_derive_sandbox_token", lambda *a: "secret")
        pool = SandboxPool.connect("pool-ghost", token="hf_test")
        pool._api.list_jobs = MagicMock(return_value=[])
        monkeypatch.setattr(pool, "_boot_host", lambda: pytest.fail("must not resurrect the pool"))

        with pytest.raises(SandboxError, match="No running host found"):
            pool.create()
        assert read_pool_cache("pool-ghost", _ctx()) is None  # cleared


class TestPoolCacheBinding:
    """The cache is a credential delivery instruction, not just data: the fast path rebuilds a
    host transport from a file and sends it the HF bearer plus the derived host token. So an
    entry is only ever readable by the endpoint, credential and namespace that wrote it, and
    only if it still says what it said when it was written."""

    def test_another_credential_gets_a_miss(self) -> None:
        _save_cache("p", [_cached_host()], context=_ctx(token="hf_someone_else"))
        assert read_pool_cache("p", _ctx()) is None

    def test_another_namespace_gets_a_miss(self) -> None:
        _save_cache("p", [_cached_host()], context=_ctx("org-a"))
        assert read_pool_cache("p", _ctx("org-b")) is None
        assert read_pool_cache("p", _ctx()) is None  # nor does "no namespace" match one
        assert read_pool_cache("p", _ctx("org-a")) is not None

    def test_another_endpoint_gets_a_miss(self) -> None:
        _save_cache("p", [_cached_host()], context=_ctx(endpoint="https://hub.staging.example"))
        assert read_pool_cache("p", _ctx()) is None

    def test_an_entry_moved_into_our_directory_is_still_a_miss(self) -> None:
        # The context is in the payload as well as the path, so copying a file (from a
        # colleague's $HF_HOME, say) into the right-looking directory does not make it ours.
        _save_cache("p", [_cached_host()], context=_ctx("org-a"))
        source = cache_mod.pool_cache_path("p", _ctx("org-a"))
        target = cache_mod.pool_cache_path("p", _ctx())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text())
        assert read_pool_cache("p", _ctx()) is None

    def test_a_renamed_entry_is_a_miss(self) -> None:
        _save_cache("mine", [_cached_host()])
        source = cache_mod.pool_cache_path("mine", _ctx())
        source.rename(source.parent / "other.json")
        assert read_pool_cache("other", _ctx()) is None

    def test_a_mistyped_entry_is_a_miss_not_a_type_error(self) -> None:
        # `dataclass` does not enforce types, so a well-shaped file with a stringly-typed
        # capacity used to parse fine and only fail later, on `capacity - live`, as an
        # uncaught TypeError in the middle of create().
        _save_cache("p", [_cached_host()])
        path = cache_mod.pool_cache_path("p", _ctx())
        data = json.loads(path.read_text())
        data["hosts"][0]["capacity"] = "50"
        path.write_text(json.dumps(data))
        assert read_pool_cache("p", _ctx()) is None

    def test_implausible_values_are_a_miss(self) -> None:
        for field, value in (
            ("capacity", True),  # `bool` is an `int`, but not a count
            ("live", -1),
            ("nonce", "not-hex"),
            ("nonce", NONCE[:8]),
            ("job_id", "../../evil"),
            ("owner", "user/../.."),
            ("base_url", 1234),
            ("updated_at", "yesterday"),
        ):
            _save_cache("p", [_cached_host()])
            path = cache_mod.pool_cache_path("p", _ctx())
            data = json.loads(path.read_text())
            data["hosts"][0][field] = value
            path.write_text(json.dumps(data))
            assert read_pool_cache("p", _ctx()) is None, f"{field}={value!r} should be a miss"

        for field, value in (("sandboxes_per_host", 0), ("sandboxes_per_host", "4"), ("max_hosts", -2), ("image", 7)):
            _save_cache("p", [_cached_host()])
            path = cache_mod.pool_cache_path("p", _ctx())
            data = json.loads(path.read_text())
            data[field] = value
            path.write_text(json.dumps(data))
            assert read_pool_cache("p", _ctx()) is None, f"{field}={value!r} should be a miss"

    def test_a_mistyped_entry_does_not_break_a_connect(self, monkeypatch) -> None:
        # The whole point: the failure mode is the documented cache miss (here, the cold path
        # finding no host) rather than a TypeError surfacing from inside create().
        _save_cache("p", [_cached_host()])
        path = cache_mod.pool_cache_path("p", _ctx())
        data = json.loads(path.read_text())
        data["hosts"][0]["capacity"] = "50"
        path.write_text(json.dumps(data))
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", _fake_list_jobs([]))
        with pytest.raises(SandboxError, match="No running host found"):
            SandboxPool.connect("p", token="hf_test")


class TestCachedHostAdmission:
    """Where a cached host says it lives decides where the HF bearer and the host token go,
    before any API call could contradict it. So the URL has to be one that only the job the
    entry names could have: HTTPS, on that job's own proxy hostname."""

    def test_a_loopback_url_is_refused(self) -> None:
        assert sandbox_mod._cached_host_rejection(_cached_host("h1", url="http://127.0.0.1:1234")) is not None

    def test_plain_http_is_refused(self) -> None:
        reason = sandbox_mod._cached_host_rejection(
            _cached_host("h1", url=f"http://h1--{SANDBOX_SERVER_PORT}.hf.jobs")
        )
        assert "https" in (reason or "")

    def test_another_jobs_url_is_refused(self) -> None:
        entry = _cached_host("h1", url=f"https://h2--{SANDBOX_SERVER_PORT}.hf.jobs")
        assert sandbox_mod._cached_host_rejection(entry) is not None

    def test_another_port_is_refused(self) -> None:
        assert sandbox_mod._cached_host_rejection(_cached_host("h1", url="https://h1--8080.hf.jobs")) is not None

    def test_the_jobs_url_of_the_entrys_own_job_is_accepted(self) -> None:
        assert sandbox_mod._cached_host_rejection(_cached_host("h1")) is None
        # The domain is not hard-coded, so staging endpoints keep working.
        entry = _cached_host("h1", url=f"https://h1--{SANDBOX_SERVER_PORT}.staging.hf.jobs/")
        assert sandbox_mod._cached_host_rejection(entry) is None

    def test_a_cache_naming_a_local_server_makes_no_request_to_it(self, monkeypatch) -> None:
        # The reproducer: a cache entry pointing at 127.0.0.1 used to receive the full bearer
        # and sandbox token on the first create. The fake records every request it gets.
        url, listener = _spawn_fake()
        _save_cache("pool-local", [_cached_host("hostA", url=url)])
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", _fake_list_jobs([]))

        pool = SandboxPool.connect("pool-local", token="hf_test")  # config from the cache, no HTTP
        with pytest.raises(SandboxError, match="No running host found"):
            pool.create()  # the host is never seeded, so discovery finds nothing
        assert listener.requests == []

    def test_a_cross_namespace_connect_does_not_reach_the_other_namespaces_hosts(self, monkeypatch) -> None:
        # `connect(pool, namespace="org-b")` used to be served org-a's cached hosts, because
        # the cache was keyed by pool id alone and its namespace was read as configuration.
        url, listener = _spawn_fake()
        _save_cache("pool-x", [_cached_host("hostA", url=url)], context=_ctx("org-a"))
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", _fake_list_jobs([]))

        with pytest.raises(SandboxError, match="No running host found"):
            SandboxPool.connect("pool-x", namespace="org-b", token="hf_test")
        assert listener.requests == []

    def test_a_connect_without_a_namespace_does_not_inherit_the_cached_one(self, monkeypatch) -> None:
        # It used to: `namespace=cache.namespace if namespace is None else namespace`.
        url, listener = _spawn_fake()
        _save_cache("pool-x", [_cached_host("hostA", url=url)], context=_ctx("org-a"))
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", _fake_list_jobs([]))

        with pytest.raises(SandboxError, match="No running host found"):
            SandboxPool.connect("pool-x", token="hf_test")
        assert listener.requests == []

    def test_an_inadmissible_url_is_pruned_from_the_cache(self, monkeypatch) -> None:
        _save_cache("pool-local", [_cached_host("hostA", url="http://127.0.0.1:1234")])
        monkeypatch.setattr(sandbox_mod.HfApi, "list_jobs", _fake_list_jobs([]))
        pool = SandboxPool.connect("pool-local", token="hf_test")
        pool._seed_hosts_from_cache()
        assert pool._hosts == []
        pool._save_cache()
        cache = read_pool_cache("pool-local", _ctx())
        assert cache is not None and cache.hosts == []


class TestCachedHostFreshness:
    """A cached host is credited without a round-trip only while the entry is fresh -- written
    by this same principal minutes ago, which is the `pool create` -> `create --pool` case the
    cache exists for. Past that, it is a hint to be checked against the Jobs API."""

    def _pool(self, name: str = "p1") -> SandboxPool:
        # `_connect_mode` skips the constructor warm-up, so seeding runs in isolation.
        return SandboxPool(image="python:3.12", flavor="cpu-basic", name=name, token="hf_test", _connect_mode=True)

    def test_a_fresh_entry_is_seeded_with_no_round_trip(self, monkeypatch) -> None:
        _save_cache("p1", [_cached_host("hostA")])
        monkeypatch.setattr(
            sandbox_mod.HfApi, "inspect_job", MagicMock(side_effect=AssertionError("must not inspect"))
        )
        pool = self._pool()
        pool._seed_hosts_from_cache()
        assert [host.job_id for host in pool._hosts] == ["hostA"]

    def test_an_aged_entry_is_confirmed_against_the_jobs_api_first(self, monkeypatch) -> None:
        _save_cache("p1", [_cached_host("hostA", age=HOST_TRUST_TTL + 60)])
        inspect = MagicMock(return_value=_pool_host_job("hostA", pool_name="p1"))
        monkeypatch.setattr(sandbox_mod.HfApi, "inspect_job", inspect)
        pool = self._pool()
        pool._seed_hosts_from_cache()
        assert [host.job_id for host in pool._hosts] == ["hostA"]
        inspect.assert_called_once()

    def test_an_aged_entry_the_job_no_longer_backs_is_dropped(self, monkeypatch) -> None:
        cases = {
            "not running": lambda job: setattr(job.status, "stage", "CANCELED"),
            "another pool": lambda job: job.labels.update({POOL_LABEL: "someone-elses-pool"}),
            "another nonce": lambda job: job.labels.update({NONCE_LABEL: "ab" * 16}),
            "another principal": lambda job: setattr(job.initiator, "id", "principal-attacker"),
            "another url": lambda job: setattr(job.status, "expose_urls", ["https://elsewhere--49983.hf.jobs"]),
        }
        for label, mutate in cases.items():
            job = _pool_host_job("hostA", pool_name="p1")
            mutate(job)
            monkeypatch.setattr(sandbox_mod.HfApi, "inspect_job", MagicMock(return_value=job))
            _save_cache("p1", [_cached_host("hostA", age=HOST_TRUST_TTL + 60)])
            pool = self._pool()
            pool._seed_hosts_from_cache()
            assert pool._hosts == [], label
            assert pool._dead_host_ids == {"hostA"}, label  # and pruned on the next save

    def test_an_unreachable_jobs_api_only_skips_the_entry(self, monkeypatch) -> None:
        # A transient failure must not prune the entry: the host may well be fine.
        _save_cache("p1", [_cached_host("hostA", age=HOST_TRUST_TTL + 60)])
        monkeypatch.setattr(sandbox_mod.HfApi, "inspect_job", MagicMock(side_effect=OSError("network is down")))
        pool = self._pool()
        pool._seed_hosts_from_cache()
        assert pool._hosts == []
        assert pool._dead_host_ids == set()


class TestTransportHardening:
    """The client's own end of the protocol: no redirect-following on a credentialed client,
    and no local file left mangled by a transfer."""

    def test_a_redirect_is_not_followed(self, fake_server: str) -> None:
        # Both credentials ride on every request, and httpx re-sends `Authorization` on a
        # same-scheme redirect, so a 302 must be surfaced rather than chased.
        elsewhere, listener = _spawn_fake()
        _FakeServer.redirect_to = elsewhere + "/v1/sandboxes"
        server = _make_server(fake_server)
        response = server.request("GET", "/v1/redirect")
        assert response.status_code == 302
        assert listener.requests == []

    def test_download_replaces_a_symlink_instead_of_writing_through_it(self, fake_server: str, tmp_path) -> None:
        sentinel = tmp_path / "precious"
        sentinel.write_text("do not touch")
        destination = tmp_path / "download.txt"
        destination.symlink_to(sentinel)

        _make_sandbox(fake_server).files.download("/x", destination)
        assert sentinel.read_text() == "do not touch"
        assert not destination.is_symlink()
        assert destination.read_bytes() == b"hello"

    def test_a_failed_download_leaves_nothing_behind(self, fake_server: str, tmp_path, monkeypatch) -> None:
        sandbox = _make_sandbox(fake_server)
        monkeypatch.setattr(sandbox, "_stream", MagicMock(side_effect=SandboxError("connection lost")))
        destination = tmp_path / "downloads" / "download.txt"
        destination.parent.mkdir()
        with pytest.raises(SandboxError):
            sandbox.files.download("/x", destination)
        assert list(destination.parent.iterdir()) == []  # no truncated file, no leftover temp file

    def test_upload_reads_the_file_it_measured(self, fake_server: str, tmp_path) -> None:
        # One descriptor is opened, `fstat`ed and streamed, instead of stat-then-reopen. A
        # symlinked source still works: `hf_hub_download` returns one.
        source = tmp_path / "data.bin"
        source.write_bytes(b"x" * 1024)
        link = tmp_path / "link.bin"
        link.symlink_to(source)
        _make_sandbox(fake_server).files.upload(link, "/dest")
        assert sum(len(body) for _, body in _FakeServer.writes) == 1024
