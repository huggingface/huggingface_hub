import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import pytest

from huggingface_hub._sandbox import (
    CommandResult,
    Sandbox,
    _derive_sandbox_token,
    _duration_to_secs,
)
from huggingface_hub.errors import SandboxCommandError, SandboxError


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
        [(300, 300), (1.5, 1), ("300", 300), ("300s", 300), ("10m", 600), ("2h", 7200), ("1d", 86400), ("1.5h", 5400)],
    )
    def test_duration_to_secs(self, value, expected) -> None:
        assert _duration_to_secs(value) == expected

    def test_duration_to_secs_invalid(self) -> None:
        with pytest.raises(ValueError):
            _duration_to_secs("oops")

    def test_command_result_ok(self) -> None:
        assert CommandResult(exit_code=0, stdout="", stderr="").ok
        assert not CommandResult(exit_code=1, stdout="", stderr="").ok

    def test_command_error_message(self) -> None:
        result = CommandResult(exit_code=2, stdout="", stderr="boom")
        error = SandboxCommandError(cmd="make", result=result)
        assert "exited with code 2" in str(error)
        assert "boom" in str(error)
        assert error.result is result


def _make_sandbox(base_url: str) -> Sandbox:
    """Build a Sandbox wired to a local server, bypassing job creation."""
    job = MagicMock()
    job.id = "job123"
    job.owner.name = "user"
    job.status.expose_urls = [base_url]
    api = MagicMock()
    api.token = "hf_test"
    return Sandbox(job_id="job123", base_url=base_url, sandbox_token="secret", job=job, api=api)


class _FakeServer(BaseHTTPRequestHandler):
    """Minimal stand-in for sbx-server speaking the same protocol."""

    files: dict = {}

    def log_message(self, *args) -> None:
        pass

    def _ndjson(self, events) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for event in events:
            self.wfile.write((json.dumps(event) + "\n").encode())
            self.wfile.flush()

    def do_POST(self) -> None:
        assert self.headers["X-Sandbox-Token"] == "secret"
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.path == "/v1/exec":
            self._ndjson(
                [
                    {"event": "start", "pid": 42},
                    {"event": "stdout", "data": "out1"},
                    {"event": "ping"},
                    {"event": "stderr", "data": "err1"},
                    {"event": "exit", "exit_code": 0 if body["cmd"] != "fail" else 3, "duration_ms": 5},
                ]
            )

    def do_GET(self) -> None:
        if self.path.startswith("/v1/files/stat"):
            body = json.dumps({"name": "x", "path": "/x", "type": "file", "size": 5}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/v1/files/read"):
            self.send_response(200)
            self.send_header("Content-Length", "5")
            self.end_headers()
            self.wfile.write(b"hello")
        elif self.path == "/v1/procs":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            body = json.dumps([{"pid": 42, "cmd": "x", "running": True, "exit_code": None}]).encode()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


@pytest.fixture()
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


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

    def test_files_read(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        assert sandbox.files.read_text("/x") == "hello"

    def test_processes(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        procs = sandbox.processes()
        assert procs[0].pid == 42 and procs[0].running

    def test_url_requires_exposed_port(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        with pytest.raises(SandboxError):
            sandbox.url(8080)

    def test_kill_is_idempotent(self, fake_server: str) -> None:
        sandbox = _make_sandbox(fake_server)
        sandbox.kill()
        sandbox.kill()
        sandbox._api.cancel_job.assert_called_once_with(job_id="job123", namespace="user")

    def test_context_manager_kills(self, fake_server: str) -> None:
        with _make_sandbox(fake_server) as sandbox:
            pass
        sandbox._api.cancel_job.assert_called_once()
