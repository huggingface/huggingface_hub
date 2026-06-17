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

Two ways to get a sandbox, sharing the same `Sandbox` surface (`run`, `spawn`,
`files`, ...):

- [`Sandbox.create`] — one dedicated HF Job per sandbox (full VM isolation, GPU
  support, public port forwarding). Best for a single sandbox, untrusted code or
  GPU workloads. ~6s cold start.
- [`SandboxPool`] — many lightweight sandboxes packed inside shared "host" jobs,
  isolated from each other by uid + the Landlock LSM. Best for fanning out many
  cheap CPU sandboxes (e.g. RL rollouts): the cost of one VM is amortized across
  dozens of sandboxes and per-sandbox cold start is ~one proxy round-trip.

```python
>>> from huggingface_hub import Sandbox
>>> with Sandbox.create() as sbx:
...     sbx.files.write("/app/hello.py", "print('hello from the sandbox')")
...     result = sbx.run("python /app/hello.py")
...     print(result.stdout)
hello from the sandbox

>>> from huggingface_hub import SandboxPool
>>> with SandboxPool(image="python:3.12") as pool:
...     boxes = pool.create(count=100)          # 100 sandboxes across shared hosts
...     print(boxes[0].run("echo hi").stdout)
hi
```
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
# the sandbox token (see _derive_sandbox_token). Present on both dedicated
# sandbox jobs and shared host jobs.
SANDBOX_LABEL = "hf-sandbox"
# Marks a job as a *host* (shared mode: one job, many landlock sandboxes).
HOST_LABEL = "hf-sandbox-host"
# Host packing capacity, so a pool discovering an existing host (possibly created
# by another process) knows how many sandboxes it was sized for.
HOST_CAPACITY_LABEL = "hf-sandbox-capacity"
# Optional pool name, to scope host reuse to a named group (see SandboxPool(name=...)).
POOL_LABEL = "hf-sandbox-pool"

DEFAULT_IMAGE = "python:3.12"
DEFAULT_IDLE_TIMEOUT = 10 * 60  # 10 minutes
# How many sandboxes a single host job packs by default (shared mode). One host
# is one billed VM, so this is the per-VM density: higher = cheaper per sandbox,
# bounded by the host's CPU/RAM. 50 is a safe default for light CPU workloads on
# cpu-basic; tune via SandboxPool(sandboxes_per_host=...).
DEFAULT_SANDBOXES_PER_HOST = 50

# Separator between a host job id and a host-local sandbox id in the public id of
# a shared sandbox: "<host_job_id>.<local_id>". Job ids are hex (no dots), so this
# is unambiguous and lets Sandbox.connect() reattach to a shared sandbox with no
# local state.
SHARED_ID_SEP = "."

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
        with self._sandbox._stream("GET", f"/procs/{self.pid}/wait") as response:
            for event in _iter_events(response):
                if event["event"] == "exit":
                    return event["exit_code"]
        raise SandboxError(f"connection lost while waiting for process {self.pid}")

    def kill(self, signal: str | int = "KILL") -> None:
        """Send a signal (default SIGKILL) to the process group."""
        self._sandbox._request("POST", f"/procs/{self.pid}/kill", json={"signal": signal})

    def logs(self, follow: bool = False) -> Iterator[tuple[str, str]]:
        """Yield `(stream, data)` tuples ("stdout"/"stderr") from the process output.

        With `follow=True`, keeps streaming live output until the process exits.
        """
        params = {"follow": "1"} if follow else {}
        with self._sandbox._stream("GET", f"/procs/{self.pid}/logs", params=params) as response:
            for event in _iter_events(response):
                if event["event"] in ("stdout", "stderr"):
                    yield event["event"], event["data"]

    def send_stdin(self, data: str | bytes, eof: bool = False) -> None:
        """Write data to the process stdin. Set `eof=True` to close the pipe afterwards."""
        payload = data.encode() if isinstance(data, str) else data
        self._sandbox._request("POST", f"/procs/{self.pid}/stdin", params={"eof": "1"} if eof else {}, content=payload)

    @property
    def running(self) -> bool:
        for proc in self._sandbox.processes():
            if proc.pid == self.pid:
                return proc.running
        return False

    def __repr__(self) -> str:
        return f"SandboxProcess(pid={self.pid}, tag={self.tag!r})"


class SandboxFiles:
    """Filesystem operations inside a sandbox, available as [`Sandbox.files`].

    In shared (pool) mode, paths are rooted at the sandbox's private home — the
    only place its code can write — so a leading `/` is taken relative to that
    home. In dedicated mode, paths are absolute on the container filesystem.
    """

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
        response = self._sandbox._request("GET", "/files/read", params={"path": path})
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
        self._sandbox._request("PUT", "/files/write", params=params, content=data)

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
        with self._sandbox._stream("GET", "/files/read", params={"path": path}) as response:
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
        workers = min(self.PARALLEL_MAX_WORKERS, len(items))
        with ThreadPoolExecutor(workers) as executor:
            return list(executor.map(fn, items))

    def _read_ranges(self, path: str, size: int) -> List[bytes]:
        def fetch(rng: tuple[int, int]) -> bytes:
            offset, length = rng
            response = self._sandbox._request(
                "GET", "/files/read", params={"path": path, "offset": offset, "length": length}
            )
            return response.content

        return self._parallel(self._ranges(size), fetch)

    def _write_ranges(self, path: str, data: bytes, mode: str | None) -> None:
        def push(rng: tuple[int, int]) -> None:
            offset, length = rng
            params: dict[str, Any] = {"path": path, "offset": offset}
            if mode is not None:
                params["mode"] = mode
            self._sandbox._request("PUT", "/files/write", params=params, content=data[offset : offset + length])

        self._parallel(self._ranges(len(data)), push)

    def list(self, path: str) -> List[FileEntry]:
        """List a directory in the sandbox."""
        response = self._sandbox._request("GET", "/files/list", params={"path": path})
        return [FileEntry(**entry) for entry in response.json()["entries"]]

    def stat(self, path: str) -> FileEntry:
        """Get metadata of a file or directory in the sandbox."""
        response = self._sandbox._request("GET", "/files/stat", params={"path": path})
        return FileEntry(**response.json())

    def exists(self, path: str) -> bool:
        """Check whether a path exists in the sandbox."""
        try:
            self.stat(path)
            return True
        except SandboxError as e:
            if e.status_code == 404:
                return False
            raise  # network/auth/server errors must not be silently reported as "missing"

    def delete(self, path: str, recursive: bool = False) -> None:
        """Delete a file or directory in the sandbox."""
        params = {"path": path}
        if recursive:
            params["recursive"] = "1"
        self._sandbox._request("DELETE", "/files/delete", params=params)

    def mkdir(self, path: str) -> None:
        """Create a directory (and parents) in the sandbox."""
        self._sandbox._request("POST", "/files/mkdir", params={"path": path})


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
    raise SandboxError(f"Sandbox API error ({response.status_code}): {message}", status_code=response.status_code)


class _SandboxServer:
    """HTTP transport to one `sbx-server` instance — a dedicated job or a shared host.

    Owns the `httpx.Client`, the base URL and the auth headers. In dedicated mode
    a server is paired 1:1 with its [`Sandbox`]; in shared mode one server (one
    host job) is shared by many sandboxes, and `live`/`capacity` track packing.
    """

    def __init__(
        self,
        *,
        job: JobInfo,
        base_url: str,
        sandbox_token: str,
        api: HfApi,
        max_connections: int = 10,
        capacity: int = 0,
    ) -> None:
        self.job = job
        self.job_id = job.id
        self.owner = job.owner.name
        self.base_url = base_url
        self._api = api
        self._auth_token = _effective_token(api)
        self._sandbox_token = sandbox_token
        # Packing bookkeeping (shared mode only).
        self.capacity = capacity
        self.live = 0
        # httpx.Client is thread-safe, so a single client serves both sequential requests
        # and the concurrent workers used for parallel file transfers / many sandboxes.
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self._auth_token}",
                "X-Sandbox-Token": sandbox_token,
            },
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
            follow_redirects=True,
        )

    @property
    def image(self) -> str | None:
        return self.job.docker_image or self.job.space_id

    def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Request to the in-job server. Raises SandboxError on API errors."""
        timeout = kwargs.pop("timeout", httpx.Timeout(60.0, connect=10.0))
        response = self._client.request(method, self.base_url + path, timeout=timeout, **kwargs)
        if response.status_code >= 400:
            _raise_for_status(response)
        return response

    @contextmanager
    def stream(self, method: str, path: str, **kwargs) -> Iterator[httpx.Response]:
        """Streaming request to the in-job server. Raises SandboxError on API errors."""
        timeout = kwargs.pop("timeout", httpx.Timeout(70.0, connect=10.0))  # server pings every 15s
        with self._client.stream(method, self.base_url + path, timeout=timeout, **kwargs) as response:
            if response.status_code >= 400:
                _raise_for_status(response)
            yield response

    def close(self) -> None:
        self._client.close()

    def cancel_job(self) -> None:
        self._api.cancel_job(job_id=self.job_id, namespace=self.owner)

    def wait_ready(self, start_timeout: float) -> None:
        """Poll /health until the server answers; fail fast (with logs) if the job dies."""
        deadline = time.time() + start_timeout
        last_job_check = 0.0
        while time.time() < deadline:
            try:
                response = self._client.get(self.base_url + "/health", timeout=httpx.Timeout(5.0))
                if response.status_code == 200:
                    return
            except httpx.RequestError:
                pass
            if time.time() - last_job_check > 2.0:
                last_job_check = time.time()
                job = self._api.inspect_job(job_id=self.job_id, namespace=self.owner)
                if job.status.stage in ("COMPLETED", "ERROR", "DELETED", "CANCELED"):
                    logs = _tail_job_logs(self._api, self.job_id, namespace=self.owner)
                    raise SandboxError(
                        f"Sandbox job {self.job_id} terminated during startup "
                        f"(status: {job.status.stage}, message: {job.status.message}).{logs}"
                    )
            time.sleep(0.15)
        raise SandboxError(f"Sandbox job {self.job_id} did not become ready within {start_timeout:.0f}s.")


class Sandbox:
    """An isolated cloud machine running on Hugging Face Jobs.

    Create a dedicated one with [`Sandbox.create`] (one job per sandbox), or get
    many cheap shared ones from a [`SandboxPool`]. Reattach to a running sandbox
    from anywhere with [`Sandbox.connect`]. Use as a context manager to terminate
    it on exit:

    ```python
    >>> from huggingface_hub import Sandbox
    >>> with Sandbox.create(image="python:3.12") as sbx:
    ...     print(sbx.run("python --version").stdout)
    ```
    """

    def __init__(
        self,
        *,
        id: str,
        server: _SandboxServer,
        local_id: str | None,
        owns_sandbox: bool,
        owns_server: bool,
    ) -> None:
        """Use [`Sandbox.create`], [`SandboxPool.create`] or [`Sandbox.connect`] instead."""
        self.id = id
        self._server = server
        # None in dedicated mode; the host-local sandbox id in shared mode.
        self._local_id = local_id
        # Path prefix for all in-server operations: dedicated routes live under
        # /v1/*, shared ones under /v1/sandboxes/<local_id>/*.
        self._base_path = "/v1" if local_id is None else f"/v1/sandboxes/{local_id}"
        # Whether exiting a `with` block terminates the sandbox (True for sandboxes
        # we created, False for ones reattached via connect()).
        self._owns_sandbox = owns_sandbox
        # Whether closing/killing this sandbox also closes the HTTP client. False
        # for pool sandboxes, whose client (the host) is owned by the pool.
        self._owns_server = owns_server
        # Set by SandboxPool to free a packing slot when a shared sandbox is killed.
        self._on_kill: Callable[["Sandbox"], None] | None = None
        self._killed = False
        self.files = SandboxFiles(self)

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
        """Create a dedicated sandbox (one HF Job) and block until it is ready (~7s on cpu-basic).

        Each sandbox is a full isolated VM, so this is the right choice for GPU
        workloads, untrusted code, or public port forwarding. To fan out many cheap
        CPU sandboxes instead, use [`SandboxPool`].

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

        command, job_env, job_secrets, job_volumes = _bootstrap_job_spec(
            api,
            hf_token,
            env=env,
            secrets=secrets,
            volumes=volumes,
            idle_timeout=idle_timeout,
            forward_hf_token=forward_hf_token,
            server_source=server_source,
            sandbox_token=sandbox_token,
        )

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
        server: "_SandboxServer | None" = None
        try:
            server = _SandboxServer(
                job=job,
                base_url=_find_server_url(job),
                sandbox_token=sandbox_token,
                api=api,
                max_connections=SandboxFiles.PARALLEL_MAX_WORKERS + 2,
            )
            server.wait_ready(start_timeout)
        except Exception:
            # run_job already started a billable job; cancel it before re-raising so it
            # doesn't linger as an orphan (e.g. if the server port isn't exposed or startup fails).
            try:
                api.cancel_job(job_id=job.id, namespace=job.owner.name)
            except Exception as e:
                logger.warning(f"Failed to cancel sandbox job {job.id} after startup failure: {e}")
            if server is not None:
                server.close()
            raise
        return cls(id=job.id, server=server, local_id=None, owns_sandbox=True, owns_server=True)

    @classmethod
    def connect(cls, sandbox_id: str, *, namespace: str | None = None, token: str | None = None) -> "Sandbox":
        """Reattach to a running sandbox from anywhere, using only its id.

        Works for both dedicated sandboxes (id is the job id) and shared/pool
        sandboxes (id is `<host_job_id>.<local_id>`), from any machine holding the
        same HF token that created it (the sandbox auth token is derived, not stored).
        """
        api = HfApi(token=token)
        sandbox_id, namespace = _split_sandbox_id(sandbox_id, namespace)
        if SHARED_ID_SEP in sandbox_id:
            host_job_id, local_id = sandbox_id.split(SHARED_ID_SEP, 1)
            server = _connect_host(api, host_job_id, namespace=namespace)
            existing = {item["id"] for item in server.request("GET", "/v1/sandboxes").json()}
            if local_id not in existing:
                server.close()
                raise SandboxError(f"Sandbox {sandbox_id} no longer exists on host {host_job_id}.")
            return cls(id=sandbox_id, server=server, local_id=local_id, owns_sandbox=False, owns_server=True)

        job = api.inspect_job(job_id=sandbox_id, namespace=namespace)
        nonce = (job.labels or {}).get(SANDBOX_LABEL)
        if nonce is None:
            raise SandboxError(f"Job {sandbox_id} is not a sandbox (missing '{SANDBOX_LABEL}' label).")
        if (job.labels or {}).get(HOST_LABEL):
            raise SandboxError(
                f"Job {sandbox_id} is a sandbox host, not a single sandbox. Connect to one of its "
                f"sandboxes with id '<host_job_id>{SHARED_ID_SEP}<local_id>' (see `hf sandbox ls`)."
            )
        if job.status.stage != "RUNNING":
            raise SandboxError(f"Sandbox {sandbox_id} is not running (status: {job.status.stage}).")
        sandbox_token = _derive_sandbox_token(_effective_token(api), nonce)
        server = _SandboxServer(
            job=job,
            base_url=_find_server_url(job),
            sandbox_token=sandbox_token,
            api=api,
            max_connections=SandboxFiles.PARALLEL_MAX_WORKERS + 2,
        )
        return cls(id=job.id, server=server, local_id=None, owns_sandbox=False, owns_server=True)

    @classmethod
    def list(cls, *, namespace: str | None = None, token: str | None = None) -> List[JobInfo]:
        """List running dedicated sandboxes (jobs created by `Sandbox.create`).

        Shared/pool sandboxes are not jobs; list those via [`SandboxPool`] or the
        `hf sandbox ls` CLI.
        """
        api = HfApi(token=token)
        return [
            job
            for job in api.list_jobs(namespace=namespace)
            if (job.labels or {}).get(SANDBOX_LABEL)
            and not (job.labels or {}).get(HOST_LABEL)
            and job.status.stage == "RUNNING"
        ]

    def kill(self) -> None:
        """Terminate the sandbox. Idempotent.

        Dedicated sandboxes cancel their underlying job; shared sandboxes are
        removed from their host (freeing a slot) while the host keeps running.
        """
        if self._killed:
            return
        try:
            if self._local_id is None:
                self._server.cancel_job()
            else:
                self._server.request("DELETE", f"/v1/sandboxes/{self._local_id}")
        except Exception as e:
            # Don't mark as killed: a later kill() call should retry so nothing leaks.
            logger.warning(f"Failed to kill sandbox {self.id}: {e}")
            return
        self._killed = True
        if self._on_kill is not None:
            self._on_kill(self)
        if self._owns_server:
            self._server.close()

    def close(self) -> None:
        """Release the local HTTP client without terminating the sandbox. Idempotent.

        No-op for pool sandboxes (the client belongs to the pool's host).
        """
        if self._owns_server:
            self._server.close()

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc_info) -> None:
        if self._owns_sandbox:
            self.kill()
        else:
            self.close()

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
        with self._stream("POST", "/exec", json=payload) as response:
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
        response = self._request("POST", "/exec", json=payload)
        return SandboxProcess(self, pid=response.json()["pid"], tag=tag)

    def processes(self) -> List[SandboxProcessInfo]:
        """List background processes started in this sandbox."""
        response = self._request("GET", "/procs")
        return [SandboxProcessInfo(**proc) for proc in response.json()]

    # ------------------------------------------------------------------ misc

    def url(self, port: int) -> str:
        """Public URL of an exposed port (requires `expose=[port]` at creation).

        Only available for dedicated sandboxes ([`Sandbox.create`]); shared/pool
        sandboxes cannot bind ports (Landlock blocks TCP bind, and ports would be
        shared across the host anyway).

        Requests to it must carry an HF token with read access to the sandbox's
        namespace: `Authorization: Bearer <token>`.
        """
        if self._local_id is not None:
            raise SandboxError(
                "Port forwarding is not available for shared/pool sandboxes. "
                "Use Sandbox.create() for per-sandbox exposed ports."
            )
        for url in self._server.job.status.expose_urls or []:
            if f"--{port}." in url:
                return url
        raise SandboxError(f"Port {port} is not exposed. Pass expose=[{port}] when creating the sandbox.")

    @property
    def image(self) -> str | None:
        return self._server.image

    @property
    def host_id(self) -> str | None:
        """For a shared/pool sandbox, the job id of the host running it (else None)."""
        return self._server.job_id if self._local_id is not None else None

    def __repr__(self) -> str:
        return f"Sandbox(id={self.id!r}, image={self.image!r})"

    # ------------------------------------------------------------------ internals

    def _request(self, method: str, resource: str, **kwargs) -> httpx.Response:
        return self._server.request(method, self._base_path + resource, **kwargs)

    @contextmanager
    def _stream(self, method: str, resource: str, **kwargs) -> Iterator[httpx.Response]:
        with self._server.stream(method, self._base_path + resource, **kwargs) as response:
            yield response


class SandboxPool:
    """A fleet of shared "host" jobs, each packing many landlock-isolated sandboxes.

    One host is one billed HF Job (a VM); it runs the sandbox server and multiplexes
    up to `sandboxes_per_host` lightweight sandboxes, isolated from each other by
    uid + the Landlock LSM. This makes large fan-outs cheap (the VM cost is shared
    across all its sandboxes) and fast (creating a sandbox is ~one proxy round-trip
    once a host is warm). Best for many parallel CPU sandboxes such as RL rollouts;
    for GPU or strong VM-level isolation between mutually-distrusting workloads, use
    [`Sandbox.create`] instead.

    Hosts are provisioned lazily as sandboxes are requested and torn down on
    `close()` (or when idle, via `idle_timeout`). The user never manages jobs:

    ```python
    >>> from huggingface_hub import SandboxPool
    >>> with SandboxPool(image="python:3.12", flavor="cpu-basic") as pool:
    ...     boxes = pool.create(count=100)   # ~ceil(100/50)=2 hosts spun up in parallel
    ...     print(boxes[0].run("echo hi").stdout)
    hi
    ```

    `create()` reuses a host that still has free capacity before booting a new one,
    so you can also grow on demand — one sandbox at a time — instead of warming a
    whole batch up front. Warm hosts are discovered via job labels, so reuse works
    **across processes** too (a fresh pool with the same `image`/`flavor`/`name`
    attaches to hosts an earlier run left behind):

    ```python
    >>> pool = SandboxPool(image="python:3.12")
    >>> sbx = pool.create()    # finds a warm host (here or in another process), else boots one
    ```
    """

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        *,
        flavor: str = "cpu-basic",
        sandboxes_per_host: int = DEFAULT_SANDBOXES_PER_HOST,
        max_hosts: int | None = None,
        name: str | None = None,
        timeout: int | float | str | None = None,
        idle_timeout: int | float | str | None = DEFAULT_IDLE_TIMEOUT,
        env: dict[str, Any] | None = None,
        secrets: dict[str, Any] | None = None,
        forward_hf_token: bool = False,
        namespace: str | None = None,
        server_source: str = "download",
        start_timeout: float = 120.0,
        token: str | None = None,
    ) -> None:
        """Configure a pool. No host job is started until the first `create()`.

        Args:
            image: Docker image for the hosts (needs `/bin/sh`). All sandboxes in the
                pool share this image.
            flavor: Hardware flavor for the host jobs (e.g. `"cpu-basic"`).
            sandboxes_per_host: How many sandboxes to pack per host (per VM density).
            max_hosts: Optional cap on the number of host jobs (a cost ceiling). When
                reached and all hosts are full, `create()` raises.
            name: Optional pool name. `create()` reuses running hosts (found via job
                labels, including from other processes) that match this pool's
                image/flavor/name before booting new ones. Reuse is scoped to the name,
                so distinct names keep separate pools from sharing hosts; `None` shares
                unnamed hosts.
            timeout: Max lifetime of each host job, e.g. `"1h"`.
            idle_timeout: Auto-shutdown a host after this much inactivity. Acts as a
                billing backstop if you forget to `close()`. Pass `None` to disable.
            env: Default environment variables for every sandbox.
            secrets: Default secret environment variables for every sandbox (delivered
                to the sandbox over TLS at creation; not stored as job secrets).
            forward_hf_token: If True, inject your HF token as `HF_TOKEN` in each sandbox.
            namespace: User or org namespace to run hosts under.
            server_source: `"download"` (default) or `"mount"`, see [`Sandbox.create`].
            start_timeout: Max seconds to wait for a host to become ready.
            token: HF token override.
        """
        if sandboxes_per_host < 1:
            raise ValueError("sandboxes_per_host must be >= 1.")
        self._api = HfApi(token=token)
        self.image = image
        self.flavor = flavor
        self.sandboxes_per_host = sandboxes_per_host
        self.max_hosts = max_hosts
        self.name = name
        self._timeout = timeout
        self._idle_timeout = idle_timeout
        self._namespace = namespace
        self._server_source = server_source
        self._start_timeout = start_timeout
        # Default per-sandbox env (merged into each create request). In shared mode
        # the sandbox process env is scrubbed (env_clear) of host job secrets, so user
        # env/secrets must be delivered per-sandbox rather than via the host job.
        self._default_env: dict[str, Any] = {**(env or {}), **(secrets or {})}
        if forward_hf_token:
            self._default_env["HF_TOKEN"] = _effective_token(self._api)
        self._hosts: List[_SandboxServer] = []
        self._lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------ public API

    def create(self, count: int = 1, *, env: dict[str, Any] | None = None) -> "Sandbox | List[Sandbox]":
        """Create `count` sandboxes, provisioning hosts as needed.

        Returns a single [`Sandbox`] when `count == 1`, else a list. Existing hosts
        with free capacity (this pool's, or warm hosts found via job labels) are
        filled first; only the shortfall boots new hosts, in
        parallel, with one batched create per host. So a single `create()` reuses a
        warm host in ~one round-trip, and a large fan-out costs ~one host cold start.

        Args:
            count: Number of sandboxes to create.
            env: Extra environment variables for these sandboxes (merged over the
                pool defaults).
        """
        if self._closed:
            raise SandboxError("This SandboxPool is closed.")
        if count < 1:
            raise ValueError("count must be >= 1.")
        sandbox_env = {**self._default_env, **(env or {})}

        # 1. Reserve slots on hosts we already track; if short, try to attach to warm
        #    hosts discovered via labels (incl. other processes') before booting new ones.
        reservations, remaining = self._reserve_on_existing(count)
        if remaining > 0:
            self._discover_hosts()
            more, remaining = self._reserve_on_existing(remaining)
            reservations.extend(more)
        new_hosts: List[_SandboxServer] = []
        try:
            if remaining > 0:
                num_new = -(-remaining // self.sandboxes_per_host)  # ceil
                new_hosts = self._provision_hosts(num_new)
                with self._lock:
                    for host in new_hosts:
                        if remaining <= 0:
                            break
                        take = min(host.capacity, remaining)
                        host.live += take
                        self._hosts.append(host)
                        reservations.append((host, take))
                        remaining -= take
            # 2. Create the sandboxes on each reserved host, in parallel across hosts.
            sandboxes = self._create_on_hosts(reservations, sandbox_env)
        except Exception:
            # Roll back optimistic slot reservations and discard freshly booted hosts.
            with self._lock:
                for host, take in reservations:
                    host.live = max(0, host.live - take)
                self._hosts = [h for h in self._hosts if h not in new_hosts]
            for host in new_hosts:
                try:
                    host.cancel_job()
                except Exception:
                    pass
                finally:
                    host.close()
            raise

        return sandboxes[0] if count == 1 else sandboxes

    @property
    def num_hosts(self) -> int:
        """Number of host jobs currently provisioned."""
        with self._lock:
            return len(self._hosts)

    @property
    def num_sandboxes(self) -> int:
        """Number of sandboxes currently handed out (across all hosts)."""
        with self._lock:
            return sum(host.live for host in self._hosts)

    @property
    def host_ids(self) -> List[str]:
        """Job ids of the provisioned hosts."""
        with self._lock:
            return [host.job_id for host in self._hosts]

    def close(self) -> None:
        """Terminate all host jobs (and therefore all their sandboxes). Idempotent."""
        with self._lock:
            hosts = self._hosts
            self._hosts = []
            self._closed = True
        for host in hosts:
            try:
                host.cancel_job()
            except Exception as e:
                logger.warning(f"Failed to cancel sandbox host {host.job_id}: {e}")
            finally:
                host.close()

    def __enter__(self) -> "SandboxPool":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------ internals

    def _reserve_on_existing(self, count: int) -> tuple[List[tuple[_SandboxServer, int]], int]:
        """Optimistically reserve up to `count` slots on existing hosts (under lock)."""
        reservations: List[tuple[_SandboxServer, int]] = []
        remaining = count
        with self._lock:
            for host in self._hosts:
                if remaining <= 0:
                    break
                free = host.capacity - host.live
                take = min(free, remaining)
                if take > 0:
                    host.live += take
                    reservations.append((host, take))
                    remaining -= take
        return reservations, remaining

    def _discover_hosts(self) -> None:
        """Attach to running host jobs that match this pool (image/flavor/name).

        Lets `create()` reuse a host warmed by an earlier call or another process
        instead of booting a new one. Hosts are found via job labels; each adopted
        host's free capacity is read from the server, so packing stays accurate.
        """
        known = {host.job_id for host in self._hosts}
        matches = []
        for job in self._api.list_jobs(namespace=self._namespace):
            labels = job.labels or {}
            if not labels.get(HOST_LABEL) or job.status.stage != "RUNNING" or job.id in known:
                continue
            if (job.docker_image or job.space_id) != self.image or job.flavor != self.flavor:
                continue
            if labels.get(POOL_LABEL) != self.name:  # None == unnamed pool
                continue
            matches.append(job)

        for job in matches:
            try:
                server = _connect_host(self._api, job.id, namespace=self._namespace)
                server.capacity = int((job.labels or {}).get(HOST_CAPACITY_LABEL, self.sandboxes_per_host))
                server.live = len(server.request("GET", "/v1/sandboxes").json())
            except SandboxError:
                continue  # host died or is still starting up; skip it
            with self._lock:
                if any(host.job_id == job.id for host in self._hosts):
                    server.close()  # adopted concurrently by another thread
                else:
                    self._hosts.append(server)

    def _provision_hosts(self, num_new: int) -> List[_SandboxServer]:
        """Boot `num_new` host jobs in parallel, respecting `max_hosts`."""
        with self._lock:
            current = len(self._hosts)
        if self.max_hosts is not None and current + num_new > self.max_hosts:
            allowed = self.max_hosts - current
            raise SandboxError(
                f"Pool needs {num_new} more host(s) but max_hosts={self.max_hosts} "
                f"allows only {max(0, allowed)} more. Raise max_hosts or kill some sandboxes."
            )
        with ThreadPoolExecutor(max_workers=min(num_new, 32)) as executor:
            return list(executor.map(lambda _: self._boot_host(), range(num_new)))

    def _boot_host(self) -> _SandboxServer:
        """Start one host job and wait until its server is ready."""
        hf_token = _effective_token(self._api)
        nonce = token_hex(16)
        sandbox_token = _derive_sandbox_token(hf_token, nonce)
        command, job_env, job_secrets, job_volumes = _bootstrap_job_spec(
            self._api,
            hf_token,
            env=None,  # host job carries no user env; sandboxes get env per create request
            secrets=None,
            volumes=None,
            idle_timeout=self._idle_timeout,
            forward_hf_token=False,
            server_source=self._server_source,
            sandbox_token=sandbox_token,
        )
        labels = {
            SANDBOX_LABEL: nonce,
            HOST_LABEL: "1",
            HOST_CAPACITY_LABEL: str(self.sandboxes_per_host),
        }
        if self.name is not None:
            labels[POOL_LABEL] = self.name
        job = self._api.run_job(
            image=self.image,
            command=command,
            env=job_env,
            secrets=job_secrets,
            flavor=self.flavor,
            timeout=self._timeout,
            labels=labels,
            volumes=job_volumes or None,
            expose=[SANDBOX_SERVER_PORT],
            namespace=self._namespace,
        )
        server = _SandboxServer(
            job=job,
            base_url=_find_server_url(job),
            sandbox_token=sandbox_token,
            api=self._api,
            max_connections=min(self.sandboxes_per_host + 8, 256),
            capacity=self.sandboxes_per_host,
        )
        try:
            server.wait_ready(self._start_timeout)
        except Exception:
            try:
                self._api.cancel_job(job_id=job.id, namespace=job.owner.name)
            except Exception as e:
                logger.warning(f"Failed to cancel sandbox host {job.id} after startup failure: {e}")
            server.close()
            raise
        return server

    def _create_on_hosts(self, reservations: List[tuple[_SandboxServer, int]], env: dict[str, Any]) -> List[Sandbox]:
        """Batch-create the reserved sandboxes on each host, in parallel across hosts."""

        def create_on(reservation: tuple[_SandboxServer, int]) -> List[Sandbox]:
            host, n = reservation
            body: dict[str, Any] = {"count": n}
            if env:
                body["env"] = env
            items = host.request("POST", "/v1/sandboxes", json=body).json()["sandboxes"]
            sandboxes = []
            for item in items:
                sandbox = Sandbox(
                    id=f"{host.job_id}{SHARED_ID_SEP}{item['id']}",
                    server=host,
                    local_id=item["id"],
                    owns_sandbox=True,
                    owns_server=False,
                )
                sandbox._on_kill = self._on_sandbox_killed
                sandboxes.append(sandbox)
            return sandboxes

        if len(reservations) == 1:
            return create_on(reservations[0])
        with ThreadPoolExecutor(max_workers=min(len(reservations), 32)) as executor:
            return [sbx for group in executor.map(create_on, reservations) for sbx in group]

    def _on_sandbox_killed(self, sandbox: Sandbox) -> None:
        """Free the packing slot of a shared sandbox that was killed."""
        with self._lock:
            sandbox._server.live = max(0, sandbox._server.live - 1)


def _effective_token(api: HfApi) -> str:
    token = api.token if isinstance(api.token, str) else get_token()
    if not token:
        raise SandboxError("A Hugging Face token is required to use sandboxes. Run `hf auth login` first.")
    return token


def _bootstrap_job_spec(
    api: HfApi,
    hf_token: str,
    *,
    env: dict[str, Any] | None,
    secrets: dict[str, Any] | None,
    volumes: List[Volume] | None,
    idle_timeout: int | float | str | None,
    forward_hf_token: bool,
    server_source: str,
    sandbox_token: str,
) -> tuple[list[str], dict[str, Any], dict[str, Any], List[Volume]]:
    """Build the (command, env, secrets, volumes) to launch a job running sbx-server.

    Shared by dedicated sandboxes and shared hosts: both download and exec the same
    unified `sbx-server` binary.
    """
    # Reserved SBX_* keys go last so user-provided env/secrets can't override them
    # (e.g. clobbering SBX_PORT would break the proxy, SBX_TOKEN would break auth).
    job_env: dict[str, Any] = {**(env or {}), "SBX_PORT": str(SANDBOX_SERVER_PORT)}
    job_secrets: dict[str, Any] = {**(secrets or {}), "SBX_TOKEN": sandbox_token}
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
    return command, job_env, job_secrets, job_volumes


def _connect_host(api: HfApi, host_job_id: str, *, namespace: str | None = None) -> _SandboxServer:
    """Reattach to a running host job and return its server transport."""
    job = api.inspect_job(job_id=host_job_id, namespace=namespace)
    nonce = (job.labels or {}).get(SANDBOX_LABEL)
    if nonce is None or not (job.labels or {}).get(HOST_LABEL):
        raise SandboxError(f"Job {host_job_id} is not a sandbox host.")
    if job.status.stage != "RUNNING":
        raise SandboxError(f"Sandbox host {host_job_id} is not running (status: {job.status.stage}).")
    return _SandboxServer(
        job=job,
        base_url=_find_server_url(job),
        sandbox_token=_derive_sandbox_token(_effective_token(api), nonce),
        api=api,
        max_connections=SandboxFiles.PARALLEL_MAX_WORKERS + 2,
    )


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
