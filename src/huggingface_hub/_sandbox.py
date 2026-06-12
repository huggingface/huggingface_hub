# coding=utf-8
# Copyright 2026-present, the HuggingFace Inc. team.
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
"""Sandboxes on Hugging Face Jobs: isolated cloud machines with command execution,
file transfer and port forwarding.

```python
>>> from huggingface_hub import Sandbox
>>> with Sandbox.create() as sbx:
...     sbx.files.write("/app/hello.py", "print('hello from the sandbox')")
...     result = sbx.run("python /app/hello.py")
...     print(result.stdout)
hello from the sandbox
```
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from secrets import token_hex
from typing import Any, BinaryIO, Callable, Iterator, List

import httpx

from . import constants
from ._space_api import Volume
from .errors import SandboxCommandError, SandboxError
from .hf_api import HfApi, JobInfo
from .utils import get_token, logging


logger = logging.get_logger(__name__)

# Port the sandbox server listens on inside the job. Deliberately uncommon so
# that typical user ports (3000, 8000, 8080, ...) stay free.
SANDBOX_SERVER_PORT = 49983

# Label attached to sandbox jobs; its value is the public nonce used to derive
# the sandbox token (see _derive_sandbox_token).
SANDBOX_LABEL = "hf-sandbox"

DEFAULT_IMAGE = "python:3.12"
DEFAULT_IDLE_TIMEOUT = 10 * 60  # 10 minutes

# Tries wget (alpine/busybox/debian), curl, then python3 (python:*-slim) to fetch
# the static server binary, then replaces itself with it. Requires only /bin/sh.
_BOOTSTRAP_DOWNLOAD = """\
set -e
d=/tmp/.sbx-server
if command -v wget >/dev/null 2>&1; then wget -q --header "Authorization: Bearer $SBX_DL_TOKEN" -O "$d" "$SBX_SERVER_URL"
elif command -v curl >/dev/null 2>&1; then curl -fsSL -H "Authorization: Bearer $SBX_DL_TOKEN" -o "$d" "$SBX_SERVER_URL"
elif command -v python3 >/dev/null 2>&1; then python3 -c 'import os,urllib.request as u; r=u.Request(os.environ["SBX_SERVER_URL"],headers={"Authorization":"Bearer "+os.environ["SBX_DL_TOKEN"]}); open("/tmp/.sbx-server","wb").write(u.urlopen(r).read())'
else echo "hf-sandbox: image has none of wget/curl/python3. Create the sandbox with server_source='mount' instead." >&2; exit 96; fi
chmod +x "$d"
unset SBX_DL_TOKEN SBX_SERVER_URL
exec "$d"
"""

# Fallback for images with /bin/sh but no downloader: the binary repo is mounted
# as a volume (the mount does not preserve the executable bit, hence cp + chmod).
_BOOTSTRAP_MOUNT = """\
set -e
cp /.hf-sandbox/sbx-server /tmp/.sbx-server
chmod +x /tmp/.sbx-server
exec /tmp/.sbx-server
"""


def _derive_sandbox_token(hf_token: str, nonce: str) -> str:
    """Derive the per-sandbox auth token from the user's HF token and the sandbox nonce.

    Stateless: any machine holding the same HF token can recompute it from the
    nonce stored in the job's labels, so `Sandbox.connect(job_id)` needs no local state.
    The HF token itself is never sent to the sandbox.
    """
    return hmac.new(hf_token.encode(), f"hf-sandbox:{nonce}".encode(), hashlib.sha256).hexdigest()


def _duration_to_secs(duration: int | float | str) -> int:
    """Parse a duration like 300, "300s", "10m", "2h", "1d" into seconds."""
    if isinstance(duration, (int, float)):
        return int(duration)
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    duration = duration.strip().lower()
    unit = 1
    if duration and duration[-1] in units:
        unit = units[duration[-1]]
        duration = duration[:-1]
    try:
        return int(float(duration) * unit)
    except ValueError:
        raise ValueError(f"Invalid duration: {duration!r}. Use e.g. 300, '300s', '10m', '2h'.") from None


@dataclass
class CommandResult:
    """Result of a command executed in a sandbox with [`Sandbox.run`]."""

    exit_code: int | None
    stdout: str
    stderr: str
    signal: int | None = None
    timed_out: bool = False
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def __repr__(self) -> str:
        out = self.stdout if len(self.stdout) <= 80 else self.stdout[:77] + "..."
        return f"CommandResult(exit_code={self.exit_code}, stdout={out!r}, duration_ms={self.duration_ms})"


@dataclass
class SandboxProcessInfo:
    """A process running (or finished) inside a sandbox."""

    pid: int
    cmd: str
    running: bool
    exit_code: int | None
    tag: str | None = None
    started_at_ms: int = 0


@dataclass
class FileEntry:
    """A file or directory inside a sandbox."""

    name: str
    path: str
    type: str  # "file" | "dir" | "symlink"
    size: int
    mtime_ms: int | None = None
    mode: str = ""


class SandboxProcess:
    """Handle on a background process started with [`Sandbox.spawn`]."""

    def __init__(self, sandbox: "Sandbox", pid: int, tag: str | None = None) -> None:
        self._sandbox = sandbox
        self.pid = pid
        self.tag = tag

    def wait(self) -> int | None:
        """Block until the process exits and return its exit code (None if killed by signal)."""
        with self._sandbox._stream("GET", f"/v1/procs/{self.pid}/wait") as response:
            for event in _iter_events(response):
                if event["event"] == "exit":
                    return event["exit_code"]
        raise SandboxError(f"connection lost while waiting for process {self.pid}")

    def kill(self, signal: str | int = "KILL") -> None:
        """Send a signal (default SIGKILL) to the process group."""
        self._sandbox._request("POST", f"/v1/procs/{self.pid}/kill", json={"signal": signal})

    def logs(self, follow: bool = False) -> Iterator[tuple[str, str]]:
        """Yield `(stream, data)` tuples ("stdout"/"stderr") from the process output.

        With `follow=True`, keeps streaming live output until the process exits.
        """
        params = {"follow": "1"} if follow else {}
        with self._sandbox._stream("GET", f"/v1/procs/{self.pid}/logs", params=params) as response:
            for event in _iter_events(response):
                if event["event"] in ("stdout", "stderr"):
                    yield event["event"], event["data"]

    def send_stdin(self, data: str | bytes, eof: bool = False) -> None:
        """Write data to the process stdin. Set `eof=True` to close the pipe afterwards."""
        payload = data.encode() if isinstance(data, str) else data
        self._sandbox._request(
            "POST", f"/v1/procs/{self.pid}/stdin", params={"eof": "1"} if eof else {}, content=payload
        )

    @property
    def running(self) -> bool:
        for proc in self._sandbox.processes():
            if proc.pid == self.pid:
                return proc.running
        return False

    def __repr__(self) -> str:
        return f"SandboxProcess(pid={self.pid}, tag={self.tag!r})"


class SandboxFiles:
    """Filesystem operations inside a sandbox, available as [`Sandbox.files`]."""

    # Above this size, transfers are split into ranged requests over parallel
    # connections: a single TCP stream through the jobs proxy is limited by the
    # bandwidth-delay product (~2 MiB/s at ~100ms RTT); parallel streams scale it.
    PARALLEL_THRESHOLD = 8 * 1024 * 1024
    PARALLEL_CHUNK_SIZE = 4 * 1024 * 1024
    PARALLEL_MAX_WORKERS = 8

    def __init__(self, sandbox: "Sandbox") -> None:
        self._sandbox = sandbox

    def read(self, path: str) -> bytes:
        """Read a file from the sandbox and return its content as bytes."""
        size = self.stat(path).size
        if size > self.PARALLEL_THRESHOLD:
            return b"".join(self._read_ranges(path, size))
        response = self._sandbox._request("GET", "/v1/files/read", params={"path": path})
        return response.content

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read a file from the sandbox and return its content as a string."""
        return self.read(path).decode(encoding)

    def write(self, path: str, data: str | bytes | BinaryIO, mode: str | None = None) -> None:
        """Write content to a file in the sandbox (parent directories are created).

        Args:
            path: Destination path in the sandbox.
            data: Content as `str`, `bytes`, or a binary file object.
            mode: Optional octal permission string, e.g. `"755"`.
        """
        if isinstance(data, str):
            data = data.encode()
        elif not isinstance(data, bytes):
            data = data.read()  # binary file object
        if len(data) > self.PARALLEL_THRESHOLD:
            self._write_ranges(path, data, mode)
            return
        params = {"path": path}
        if mode is not None:
            params["mode"] = mode
        self._sandbox._request("PUT", "/v1/files/write", params=params, content=data)

    def upload(self, local_path: str | Path, path: str, mode: str | None = None) -> None:
        """Upload a local file to the sandbox (parallel ranged transfer for large files)."""
        size = Path(local_path).stat().st_size
        if size > self.PARALLEL_THRESHOLD:
            self._write_ranges(path, Path(local_path).read_bytes(), mode)
            return
        with open(local_path, "rb") as f:
            self.write(path, f, mode=mode)

    def download(self, path: str, local_path: str | Path) -> None:
        """Download a file from the sandbox (parallel ranged transfer for large files)."""
        size = self.stat(path).size
        if size > self.PARALLEL_THRESHOLD:
            with open(local_path, "wb") as f:
                for part in self._read_ranges(path, size):
                    f.write(part)
            return
        with self._sandbox._stream("GET", "/v1/files/read", params={"path": path}) as response:
            with open(local_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)

    def _ranges(self, size: int) -> List[tuple[int, int]]:
        chunk = self.PARALLEL_CHUNK_SIZE
        return [(offset, min(chunk, size - offset)) for offset in range(0, size, chunk)]

    def _parallel(self, items: List[Any], fn: Callable[[Any], Any]) -> List[Any]:
        """Run `fn(item)` over items concurrently.

        All workers share the sandbox's `httpx.Client`, which is thread-safe and pools
        connections, so parallel transfers fan out over several streams at once.
        """
        from concurrent.futures import ThreadPoolExecutor

        workers = min(self.PARALLEL_MAX_WORKERS, len(items))
        with ThreadPoolExecutor(workers) as executor:
            return list(executor.map(fn, items))

    def _read_ranges(self, path: str, size: int) -> List[bytes]:
        def fetch(rng: tuple[int, int]) -> bytes:
            offset, length = rng
            response = self._sandbox._request(
                "GET", "/v1/files/read", params={"path": path, "offset": offset, "length": length}
            )
            return response.content

        return self._parallel(self._ranges(size), fetch)

    def _write_ranges(self, path: str, data: bytes, mode: str | None) -> None:
        def push(rng: tuple[int, int]) -> None:
            offset, length = rng
            params: dict[str, Any] = {"path": path, "offset": offset}
            if mode is not None:
                params["mode"] = mode
            self._sandbox._request("PUT", "/v1/files/write", params=params, content=data[offset : offset + length])

        self._parallel(self._ranges(len(data)), push)

    def list(self, path: str) -> List[FileEntry]:
        """List a directory in the sandbox."""
        response = self._sandbox._request("GET", "/v1/files/list", params={"path": path})
        return [FileEntry(**entry) for entry in response.json()["entries"]]

    def stat(self, path: str) -> FileEntry:
        """Get metadata of a file or directory in the sandbox."""
        response = self._sandbox._request("GET", "/v1/files/stat", params={"path": path})
        return FileEntry(**response.json())

    def exists(self, path: str) -> bool:
        """Check whether a path exists in the sandbox."""
        try:
            self.stat(path)
            return True
        except SandboxError:
            return False

    def delete(self, path: str, recursive: bool = False) -> None:
        """Delete a file or directory in the sandbox."""
        params = {"path": path}
        if recursive:
            params["recursive"] = "1"
        self._sandbox._request("DELETE", "/v1/files/delete", params=params)

    def mkdir(self, path: str) -> None:
        """Create a directory (and parents) in the sandbox."""
        self._sandbox._request("POST", "/v1/files/mkdir", params={"path": path})


def _iter_events(response: httpx.Response) -> Iterator[dict]:
    """Iterate NDJSON events from a streaming response, skipping keepalive pings."""
    for line in response.iter_lines():
        if not line:
            continue
        event = json.loads(line)
        if event.get("event") != "ping":
            yield event


def _raise_for_status(response: httpx.Response) -> None:
    """Read the error body and raise a SandboxError (works for streaming responses too)."""
    response.read()  # no-op for buffered responses, reads the body for streaming ones
    try:
        message = response.json()["error"]
    except Exception:
        message = response.text[:500]
    raise SandboxError(f"Sandbox API error ({response.status_code}): {message}")


class Sandbox:
    """An isolated cloud machine running on Hugging Face Jobs.

    Create one with [`Sandbox.create`], reattach to a running one with
    [`Sandbox.connect`]. Use as a context manager to kill it on exit:

    ```python
    >>> from huggingface_hub import Sandbox
    >>> with Sandbox.create(image="python:3.12") as sbx:
    ...     print(sbx.run("python --version").stdout)
    ```
    """

    def __init__(self, *, job_id: str, base_url: str, sandbox_token: str, job: JobInfo, api: HfApi) -> None:
        """Use [`Sandbox.create`] or [`Sandbox.connect`] instead of calling this directly."""
        self.id = job_id
        self._base_url = base_url
        self._job = job
        self._api = api
        self._killed = False
        self.files = SandboxFiles(self)

        self._sandbox_token = sandbox_token
        self._auth_token = _effective_token(api)
        # httpx.Client is thread-safe, so a single client serves both sequential requests
        # and the concurrent workers used for parallel file transfers.
        self._client = self._new_client()

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    def create(
        cls,
        image: str = DEFAULT_IMAGE,
        *,
        flavor: str = "cpu-basic",
        timeout: int | float | str | None = None,
        idle_timeout: int | float | str | None = DEFAULT_IDLE_TIMEOUT,
        env: dict[str, Any] | None = None,
        secrets: dict[str, Any] | None = None,
        volumes: List[Volume] | None = None,
        expose: List[int] | None = None,
        namespace: str | None = None,
        forward_hf_token: bool = False,
        server_source: str = "download",
        start_timeout: float = 120.0,
        token: str | None = None,
    ) -> "Sandbox":
        """Create a sandbox and block until it is ready (~7s on cpu-basic).

        Args:
            image: Any Docker image with `/bin/sh` (Docker Hub or `hf.co/spaces/...`).
            flavor: Hardware flavor, e.g. `"cpu-basic"`, `"a10g-small"`. See `hf jobs hardware`.
            timeout: Max sandbox lifetime (job timeout), e.g. `"1h"`. Jobs default: 30 min.
            idle_timeout: Auto-shutdown after this much inactivity (no API calls, no running
                processes). Defaults to 10 minutes; pass `None` to disable.
            env: Environment variables available in the sandbox.
            secrets: Secret environment variables (encrypted server-side).
            volumes: HF repos/buckets to mount, see [`Volume`].
            expose: Extra container ports to expose publicly; URLs via [`Sandbox.url`].
            namespace: User or org namespace to run under (defaults to current user).
            forward_hf_token: If True, your HF token is injected as `HF_TOKEN` (opt-in).
            server_source: `"download"` (default, needs wget/curl/python3 in image) or
                `"mount"` (needs only `/bin/sh`; adds ~3s to cold start).
            start_timeout: Max seconds to wait for the sandbox to become ready.
            token: HF token override.
        """
        api = HfApi(token=token)
        hf_token = _effective_token(api)
        nonce = token_hex(16)
        sandbox_token = _derive_sandbox_token(hf_token, nonce)

        job_env: dict[str, Any] = {"SBX_PORT": str(SANDBOX_SERVER_PORT), **(env or {})}
        job_secrets: dict[str, Any] = {"SBX_TOKEN": sandbox_token, **(secrets or {})}
        job_volumes = list(volumes or [])
        if idle_timeout is not None:
            job_env["SBX_IDLE_TIMEOUT"] = str(_duration_to_secs(idle_timeout))
        if forward_hf_token:
            job_secrets["HF_TOKEN"] = hf_token

        if server_source == "download":
            job_env["SBX_SERVER_URL"] = f"{api.endpoint}/{constants.SANDBOX_SERVER_REPO}/resolve/main/sbx-server"
            job_secrets["SBX_DL_TOKEN"] = hf_token
            command = ["/bin/sh", "-c", _BOOTSTRAP_DOWNLOAD]
        elif server_source == "mount":
            job_volumes.append(
                Volume(
                    type="model",
                    source=constants.SANDBOX_SERVER_REPO,
                    mount_path="/.hf-sandbox",
                    read_only=True,
                )
            )
            command = ["/bin/sh", "-c", _BOOTSTRAP_MOUNT]
        else:
            raise ValueError(f"server_source must be 'download' or 'mount', not {server_source!r}")

        job = api.run_job(
            image=image,
            command=command,
            env=job_env,
            secrets=job_secrets,
            flavor=flavor,
            timeout=timeout,
            labels={SANDBOX_LABEL: nonce},
            volumes=job_volumes or None,
            expose=[SANDBOX_SERVER_PORT, *(expose or [])],
            namespace=namespace,
        )
        try:
            base_url = _find_server_url(job)
            sandbox = cls(job_id=job.id, base_url=base_url, sandbox_token=sandbox_token, job=job, api=api)
            sandbox._wait_ready(start_timeout)
        except Exception:
            # run_job already started a billable job; cancel it before re-raising so it
            # doesn't linger as an orphan (e.g. if the server port isn't exposed or startup fails).
            try:
                api.cancel_job(job_id=job.id, namespace=job.owner.name)
            except Exception as e:
                logger.warning(f"Failed to cancel sandbox job {job.id} after startup failure: {e}")
            raise
        return sandbox

    @classmethod
    def connect(cls, sandbox_id: str, *, namespace: str | None = None, token: str | None = None) -> "Sandbox":
        """Reattach to a running sandbox from anywhere, using only its id.

        Works from any machine holding the same HF token that created the sandbox
        (the sandbox auth token is derived, not stored).
        """
        api = HfApi(token=token)
        sandbox_id, namespace = _split_sandbox_id(sandbox_id, namespace)
        job = api.inspect_job(job_id=sandbox_id, namespace=namespace)
        nonce = (job.labels or {}).get(SANDBOX_LABEL)
        if nonce is None:
            raise SandboxError(f"Job {sandbox_id} is not a sandbox (missing '{SANDBOX_LABEL}' label).")
        if job.status.stage != "RUNNING":
            raise SandboxError(f"Sandbox {sandbox_id} is not running (status: {job.status.stage}).")
        sandbox_token = _derive_sandbox_token(_effective_token(api), nonce)
        return cls(
            job_id=job.id,
            base_url=_find_server_url(job),
            sandbox_token=sandbox_token,
            job=job,
            api=api,
        )

    @classmethod
    def list(cls, *, namespace: str | None = None, token: str | None = None) -> List[JobInfo]:
        """List running sandboxes (jobs created by `Sandbox.create`)."""
        api = HfApi(token=token)
        return [
            job
            for job in api.list_jobs(namespace=namespace)
            if (job.labels or {}).get(SANDBOX_LABEL) and job.status.stage == "RUNNING"
        ]

    def kill(self) -> None:
        """Terminate the sandbox (cancels the underlying job). Idempotent."""
        if self._killed:
            return
        try:
            self._api.cancel_job(job_id=self.id, namespace=self._job.owner.name)
        except Exception as e:
            # Don't mark as killed: a later kill() call should retry so the job isn't left running.
            logger.warning(f"Failed to cancel sandbox job {self.id}: {e}")
            return
        self._killed = True
        self._client.close()

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc_info) -> None:
        self.kill()

    # ------------------------------------------------------------------ exec

    def run(
        self,
        cmd: str | List[str],
        *,
        env: dict[str, Any] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        stdin: str | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
        check: bool = True,
    ) -> CommandResult:
        """Run a command in the sandbox and wait for it, streaming output live.

        Args:
            cmd: A shell command string (run with `/bin/sh -c`) or an argv list.
            env: Extra environment variables for this command.
            cwd: Working directory.
            timeout: Kill the command (whole process group) after this many seconds.
            stdin: Data to write to the command's stdin.
            on_stdout / on_stderr: Callbacks invoked with output chunks as they arrive.
            check: If True (default), raise [`SandboxCommandError`] on non-zero exit.

        Returns: [`CommandResult`] with `exit_code`, `stdout`, `stderr`, `duration_ms`.
        """
        payload: dict[str, Any] = {"cmd": cmd}
        if env:
            payload["env"] = env
        if cwd:
            payload["cwd"] = cwd
        if timeout is not None:
            payload["timeout"] = timeout
        if stdin is not None:
            payload["stdin"] = stdin

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        result: CommandResult | None = None
        with self._stream("POST", "/v1/exec", json=payload) as response:
            for event in _iter_events(response):
                if event["event"] == "stdout":
                    stdout_parts.append(event["data"])
                    if on_stdout is not None:
                        on_stdout(event["data"])
                elif event["event"] == "stderr":
                    stderr_parts.append(event["data"])
                    if on_stderr is not None:
                        on_stderr(event["data"])
                elif event["event"] == "exit":
                    result = CommandResult(
                        exit_code=event["exit_code"],
                        stdout="".join(stdout_parts),
                        stderr="".join(stderr_parts),
                        signal=event.get("signal"),
                        timed_out=event.get("timed_out", False),
                        duration_ms=event.get("duration_ms", 0),
                    )
        if result is None:
            raise SandboxError("connection lost while running command")
        if check and (result.exit_code != 0 or result.timed_out):
            raise SandboxCommandError(cmd=cmd, result=result)
        return result

    def spawn(
        self,
        cmd: str | List[str],
        *,
        env: dict[str, Any] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        tag: str | None = None,
    ) -> SandboxProcess:
        """Start a background process and return immediately with a [`SandboxProcess`] handle."""
        payload: dict[str, Any] = {"cmd": cmd, "background": True}
        if env:
            payload["env"] = env
        if cwd:
            payload["cwd"] = cwd
        if timeout is not None:
            payload["timeout"] = timeout
        if tag is not None:
            payload["tag"] = tag
        response = self._request("POST", "/v1/exec", json=payload)
        return SandboxProcess(self, pid=response.json()["pid"], tag=tag)

    def processes(self) -> List[SandboxProcessInfo]:
        """List background processes started in this sandbox."""
        response = self._request("GET", "/v1/procs")
        return [SandboxProcessInfo(**proc) for proc in response.json()]

    # ------------------------------------------------------------------ misc

    def url(self, port: int) -> str:
        """Public URL of an exposed port (requires `expose=[port]` at creation).

        Requests to it must carry an HF token with read access to the sandbox's
        namespace: `Authorization: Bearer <token>`.
        """
        for url in self._job.status.expose_urls or []:
            if f"--{port}." in url:
                return url
        raise SandboxError(f"Port {port} is not exposed. Pass expose=[{port}] when creating the sandbox.")

    @property
    def image(self) -> str | None:
        return self._job.docker_image or self._job.space_id

    def __repr__(self) -> str:
        return f"Sandbox(id={self.id!r}, image={self.image!r})"

    # ------------------------------------------------------------------ internals

    def _new_client(self) -> httpx.Client:
        """Build an httpx.Client configured with the sandbox auth headers and connection pool."""
        max_connections = SandboxFiles.PARALLEL_MAX_WORKERS + 2
        return httpx.Client(
            headers={
                "Authorization": f"Bearer {self._auth_token}",
                "X-Sandbox-Token": self._sandbox_token,
            },
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
            follow_redirects=True,
        )

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Request to the in-sandbox server. Raises SandboxError on API errors."""
        timeout = httpx.Timeout(60.0, connect=10.0)
        response = self._client.request(method, self._base_url + path, timeout=timeout, **kwargs)
        if response.status_code >= 400:
            _raise_for_status(response)
        return response

    @contextmanager
    def _stream(self, method: str, path: str, **kwargs) -> Iterator[httpx.Response]:
        """Streaming request to the in-sandbox server. Raises SandboxError on API errors."""
        timeout = httpx.Timeout(70.0, connect=10.0)  # server pings every 15s on streams
        with self._client.stream(method, self._base_url + path, timeout=timeout, **kwargs) as response:
            if response.status_code >= 400:
                _raise_for_status(response)
            yield response

    def _wait_ready(self, start_timeout: float) -> None:
        """Poll /health until the server answers; fail fast (with logs) if the job dies."""
        deadline = time.time() + start_timeout
        last_job_check = 0.0
        while time.time() < deadline:
            try:
                response = self._client.get(self._base_url + "/health", timeout=httpx.Timeout(5.0))
                if response.status_code == 200:
                    return
            except httpx.RequestError:
                pass
            if time.time() - last_job_check > 2.0:
                last_job_check = time.time()
                namespace = self._job.owner.name
                job = self._api.inspect_job(job_id=self.id, namespace=namespace)
                if job.status.stage in ("COMPLETED", "ERROR", "DELETED"):
                    logs = _tail_job_logs(self._api, self.id, namespace=namespace)
                    raise SandboxError(
                        f"Sandbox job {self.id} terminated during startup "
                        f"(status: {job.status.stage}, message: {job.status.message}).{logs}"
                    )
            time.sleep(0.15)
        raise SandboxError(f"Sandbox {self.id} did not become ready within {start_timeout:.0f}s.")


def _effective_token(api: HfApi) -> str:
    token = api.token if isinstance(api.token, str) else get_token()
    if not token:
        raise SandboxError("A Hugging Face token is required to use sandboxes. Run `hf auth login` first.")
    return token


def _split_sandbox_id(sandbox_id: str, namespace: str | None) -> tuple[str, str | None]:
    """Accept `namespace/sandbox_id` ids (as shown in the Hub UI), like `hf jobs` does."""
    if "/" not in sandbox_id:
        return sandbox_id, namespace
    extracted_namespace, parsed_id = sandbox_id.split("/", 1)
    if not extracted_namespace or not parsed_id or "/" in parsed_id:
        raise SandboxError(f"Sandbox id must be 'sandbox_id' or 'namespace/sandbox_id', got {sandbox_id!r}.")
    if namespace is not None and namespace != extracted_namespace:
        raise SandboxError(
            f"Conflicting namespace: got namespace={namespace!r} but sandbox id implies namespace={extracted_namespace!r}."
        )
    return parsed_id, extracted_namespace


def _find_server_url(job: JobInfo) -> str:
    for url in job.status.expose_urls or []:
        if f"--{SANDBOX_SERVER_PORT}." in url:
            return url
    raise SandboxError(f"Job {job.id} does not expose the sandbox server port {SANDBOX_SERVER_PORT}.")


def _tail_job_logs(api: HfApi, job_id: str, *, namespace: str | None = None, limit: int = 20) -> str:
    try:
        lines = list(api.fetch_job_logs(job_id=job_id, namespace=namespace))[-limit:]
    except Exception:
        return ""
    return " Last logs:\n" + "\n".join(f"  {line}" for line in lines) if lines else ""
