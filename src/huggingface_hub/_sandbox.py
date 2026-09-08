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
import hashlib
import hmac
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from secrets import token_hex
from typing import Any, BinaryIO, Callable, Iterator, List, Literal, overload
from urllib.parse import urlparse

import httpx

from . import constants
from ._sandbox_cache import (
    HOST_TRUST_TTL,
    CacheContext,
    CachedHost,
    delete_pool_cache,
    read_pool_cache,
    save_pool_cache,
)
from ._space_api import Volume
from .errors import HfHubHTTPError, SandboxCommandError, SandboxError
from .hf_api import HfApi, JobInfo
from .utils import get_token, logging
from .utils._parsing import parse_duration


logger = logging.get_logger(__name__)

# Port the sandbox server listens on inside the job. Deliberately uncommon so that typical user ports stay free.
SANDBOX_SERVER_PORT = 49983

# Stable marker present on every sandbox job to ease filtering
SANDBOX_LABEL = "hf-sandbox"
# Sandbox mode: "dedicated" or "pool" to ease filtering
MODE_LABEL = "hf-sandbox-mode"
MODE_DEDICATED = "dedicated"
MODE_POOL = "pool"
# Pool name, to scope host reuse to a named group (see SandboxPool(name=...)). Present on pool
# host jobs only. Pool config (capacity, idle timeout) lives in the host's env vars, read via
# inspect_job — labels are kept for filtering/grouping only.
POOL_LABEL = "hf-sandbox-pool"
# Per-job public nonce the sandbox token is derived from (see _derive_sandbox_token), so
# `Sandbox.connect(id)` can recompute the token from any machine with no local state.
NONCE_LABEL = "hf-sandbox-nonce"

DEFAULT_IMAGE = "python:3.12"

# The exact sbx-server build every sandbox job downloads and runs as root, pinned by digest.
# The bucket object is fetched by its sha256 name and the download is verified before it is
# made executable (see `_BOOTSTRAP_DOWNLOAD`), so the bytes that run are the bytes that were
# reviewed for this release -- not whatever the mutable `sbx-server` alias points at today.
# Both constants move together on every server release; the digest is printed by the server
# repo's publish workflow.
SANDBOX_SERVER_VERSION = "0.6.0"
SANDBOX_SERVER_SHA256 = "501290eacb36a3bd8746b2f2ee20e85190ec3f697b4756c3e51a574384e61db4"

# The sbx-server wire contract this client drives, checked against `/health`'s `protocol` on
# startup. A pool host keeps the binary it downloaded at boot for up to 24h, so pinning a
# digest does not stop this client from meeting a server it did not pin -- the check does.
SANDBOX_SERVER_PROTOCOL = 2

DEFAULT_IDLE_TIMEOUT = 10 * 60  # 10 minutes
SANDBOX_MAX_LIFETIME = "24h"

DEFAULT_SANDBOXES_PER_HOST = 50

# How long `close()` waits for in-flight creates before giving up on them. A
# wedged create must not hang process exit, but returning immediately would let
# a host it is still booting escape teardown.
_CLOSE_DRAIN_TIMEOUT = 30.0

# Ceiling on the output `run()` will accumulate for its result. Without one, a
# runaway command (a loop printing to stdout, say) is an unbounded allocation in
# the caller's process -- and it happened even when a callback was already
# consuming the output. Raising beats being OOM-killed with no explanation.
MAX_CAPTURED_OUTPUT_CHARS = 64 * 1024 * 1024

# Which pool hosts `create()` may adopt from job labels. Labels are set by whoever
# creates a Job, so "any job carrying our pool's labels" is not an identity claim --
# see the `adopt_hosts` argument of [`SandboxPool`].
ADOPT_OWN = "own"
ADOPT_NAMESPACE = "namespace"
ADOPT_NEVER = "never"

SHARED_ID_SEP = "."

# Job stages in which a sandbox/host is finished and needs no teardown.
_TERMINAL_STAGES = ("COMPLETED", "ERROR", "DELETED", "CANCELED")

# Safety bound on create()'s pack-retry loop
_MAX_PACK_ROUNDS = 8

# hf-mount path where the server bucket is mounted on every sandbox job
_SERVER_MOUNT_PATH = "/.hf-sbx-server"

# Job startup script (needs only /bin/sh). The server bucket is public, so the download is
# unauthenticated: no HF credential is ever placed in the job environment (see `_derive_sandbox_token`).
#
# It fetches a public object and runs it as root, as PID 1, holding the sandbox token -- so it
# checks the bytes against `SANDBOX_SERVER_SHA256` and refuses to run them if they don't match.
# The check happens *before* `chmod +x`: an unverified file that is already executable is one
# slip away from being executed. Neither `sha256sum` nor `openssl` is guaranteed to exist in an
# arbitrary image, so both are tried; with neither, the default is to refuse rather than to run
# unverified code, and `SBX_ALLOW_UNVERIFIED_SERVER=1` (passed via `env=`) is the way out for an
# image that has to. It never bypasses a *failed* check, only a missing tool.
_BOOTSTRAP_DOWNLOAD = """\
set -e
d=/tmp/.sbx-server
if command -v wget >/dev/null 2>&1; then wget -q -O "$d" "$SBX_SERVER_URL"
elif command -v curl >/dev/null 2>&1; then curl -fsSL -o "$d" "$SBX_SERVER_URL"
else cp "$SBX_SERVER_MOUNT/sbx-server-$SBX_SERVER_SHA256" "$d"; fi
if command -v sha256sum >/dev/null 2>&1; then
  actual=$(sha256sum "$d" | cut -d' ' -f1)
elif command -v openssl >/dev/null 2>&1; then
  actual=$(openssl dgst -sha256 -r "$d" | cut -d' ' -f1)
elif [ "${SBX_ALLOW_UNVERIFIED_SERVER:-}" = 1 ]; then
  echo "sbx: SBX_ALLOW_UNVERIFIED_SERVER=1: running the sandbox server unverified" >&2
  actual=$SBX_SERVER_SHA256  # opted out, so there is nothing left to compare against
else
  echo "sbx: cannot verify the sandbox server digest: this image has neither sha256sum nor" >&2
  echo "sbx: openssl, so refusing to run it. To accept an unverified server, pass" >&2
  echo "sbx: env={'SBX_ALLOW_UNVERIFIED_SERVER': '1'} when creating the sandbox." >&2
  exit 1
fi
if [ "$actual" != "$SBX_SERVER_SHA256" ]; then
  echo "sbx: sandbox server digest mismatch: got $actual," >&2
  echo "sbx: expected $SBX_SERVER_SHA256." >&2
  echo "sbx: refusing to run it. Upgrade huggingface_hub if this client is pinned to a" >&2
  echo "sbx: digest that is no longer published." >&2
  exit 1
fi
chmod +x "$d"
unset SBX_SERVER_URL SBX_SERVER_MOUNT SBX_SERVER_SHA256 SBX_ALLOW_UNVERIFIED_SERVER
exec "$d"
"""


def _derive_sandbox_token(hf_token: str, nonce: str) -> str:
    """Derive the sandbox auth token from the user's HF token and the job's nonce.

    Stateless: any machine holding the same HF token can recompute it from the
    nonce stored in the job's labels, so `Sandbox.connect(job_id)` needs no local state.
    Only the derived token is passed to the sandbox server as a job secret; the HF token
    itself is not.

    Scope: one nonce is minted per *job*, not per sandbox. For a dedicated sandbox those are
    the same thing. In a pool this derives the *host* credential, which manages the pool and
    can recover per-sandbox tokens -- a leak there is host-wide. Each pooled sandbox also has
    its own random capability token, minted by the host server, which is what per-sandbox
    operations and `proxy_headers` use.

    This is not a hardened boundary in either direction. An untrusted image can own the
    sandbox port, so don't treat it as a guarantee that credentials stay out of the sandbox;
    and the token is derived for whichever job carries the matching nonce label, so don't
    treat holding it as proof of which job you are talking to.
    """
    return hmac.new(hf_token.encode(), f"hf-sandbox:{nonce}".encode(), hashlib.sha256).hexdigest()


def _duration_to_secs(duration: int | float | str) -> int:
    """Parse a duration like 300, "300s", "10m", "2h", "1d" into seconds."""
    if isinstance(duration, (int, float)):
        return int(duration)
    return parse_duration(duration)


@dataclass
class SandboxCommandResult:
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
        return f"SandboxCommandResult(exit_code={self.exit_code}, stdout={out!r}, duration_ms={self.duration_ms})"


@dataclass
class SandboxProcess:
    """A background process started in a sandbox with [`Sandbox.run`]`(..., background=True)`.

    List a sandbox's processes with [`Sandbox.processes`] and stop one with [`SandboxProcess.kill`].
    Recently completed processes stay in the listing (the server keeps a bounded number of
    them), so `running` and `exit_code` tell whether a process is still alive or already
    exited (as of when it was listed).
    """

    # Server-assigned handle, and the only identifier that addresses this process.
    # `pid` is the OS pid: useful for correlating with `ps` inside the sandbox, but
    # the OS may reuse it, so it is observational only. `None` for a process on a
    # host running a server that predates opaque ids.
    id: str | None
    pid: int
    cmd: str | List[str]
    # Back-reference to the sandbox, used by `kill()`. Excluded from repr/eq so a process
    # stays a plain data object.
    _sandbox: "Sandbox" = field(repr=False, compare=False)
    tag: str | None = None
    started_at_ms: int | None = None
    running: bool = True
    exit_code: int | None = None

    def kill(self) -> bool:
        """Terminate the background process. Idempotent.

        Returns whether this call is what stopped it: `False` means it had already
        exited or been terminated, which is not an error.

        Note that a descendant which detaches with `setsid()` leaves the signalled
        process group and outlives this call. Delete the sandbox to be certain
        everything it started is gone.
        """
        if self.id is None:
            raise SandboxError(
                "This process cannot be stopped individually: its sandbox runs a sandbox server "
                "that predates opaque process ids, so the server never issued one. Recycle the "
                "pool's hosts, or delete the sandbox to stop everything it started."
            )
        response = self._sandbox._request("DELETE", f"/processes/{self.id}")
        return bool(response.json().get("killed", False))


@dataclass
class FileEntry:
    """A file or directory inside a sandbox."""

    name: str
    path: str
    type: Literal["file", "dir", "symlink"]
    size: int
    mtime_ms: int | None = None
    mode: str = ""


@contextmanager
def _open_download_target(local_path: Path) -> Iterator[BinaryIO]:
    """Yield a writer whose bytes only appear at `local_path` once the download completes.

    Writing straight to the destination has three problems: an interrupted transfer leaves a
    truncated file at the final name; `open(path, "wb")` follows a symlink sitting there, so
    the sandbox's output can be redirected into any file this process can write; and a
    predictable temp name is a race another local process can win.

    So the bytes go to a temp file in the destination directory, created `O_EXCL` (nobody
    else's file) and `O_NOFOLLOW` (not a symlink) with mode `0600` and an unguessable name,
    then `os.replace`d into place. `rename` does not follow symlinks either, so a symlinked
    destination is *replaced* rather than written through, and a failure leaves nothing behind.
    """
    tmp = local_path.parent / f".{local_path.name}.{token_hex(8)}.part"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            yield f
        os.replace(tmp, local_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class SandboxFiles:
    """Filesystem operations inside a sandbox, available as [`Sandbox.files`].

    In shared (pool) mode, paths are rooted at the sandbox's private home — the
    only place its code can write — so a leading `/` is taken relative to that
    home. In dedicated mode, paths are absolute on the container filesystem.

    The rooting is lexical (`..` components are normalized away), so no path you pass can
    name a location outside the home. It is not a symlink-proof boundary: these operations
    run with the host server's privileges and currently follow symlinks, so a symlink that
    the sandbox's own code placed in its home can point one of them outside. See the
    "Known limitations" section of the sandbox conceptual guide.
    """

    # Above this size, transfers are split into ranged requests over parallel
    # connections: a single TCP stream through the jobs proxy is limited by the
    # bandwidth-delay product (~2 MiB/s at ~100ms RTT); parallel streams scale it.
    PARALLEL_THRESHOLD = 2 * 1024 * 1024
    PARALLEL_CHUNK_SIZE = 1 * 1024 * 1024
    PARALLEL_MAX_WORKERS = 16
    # Ceiling on what `read`/`read_text` will materialize in memory. A parallel
    # read used to collect every chunk into a list and *then* join it, so a 2 GB
    # file peaked at roughly twice its size before the caller saw a byte. Reading
    # a file into memory is inherently bounded by the file; this makes the bound
    # explicit and points at the streaming alternative instead of dying in an
    # allocator.
    MAX_READ_BYTES = 512 * 1024 * 1024

    def __init__(self, sandbox: "Sandbox") -> None:
        self._sandbox = sandbox

    def read(self, path: str) -> bytes:
        """Read a file from the sandbox and return its content as bytes.

        Raises [`SandboxError`] above `MAX_READ_BYTES`; use [`download`] for
        anything that large, which streams to disk instead of buffering.
        """
        size = self.stat(path).size
        if size > self.MAX_READ_BYTES:
            raise SandboxError(
                f"{path} is {size} bytes, over the {self.MAX_READ_BYTES}-byte limit for reading into "
                "memory. Use `files.download(path, local_path)`, which streams to disk."
            )
        if size > self.PARALLEL_THRESHOLD:
            # Write each range into one preallocated buffer as it arrives, rather
            # than collecting every chunk and joining: the join doubled peak memory.
            buffer = bytearray(size)
            for offset, part in self._read_ranges(path, size):
                buffer[offset : offset + len(part)] = part
            return bytes(buffer)
        response = self._sandbox._request("GET", "/files/read", params={"path": path})
        return response.content

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read a file from the sandbox and return its content as a string."""
        return self.read(path).decode(encoding)

    def write(self, path: str, data: str | bytes | BinaryIO, mode: str | None = None) -> None:
        """Write content to a file in the sandbox (parent directories are created)."""
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
        """Upload a local file to the sandbox.

        The file is opened once and both sized (`fstat`) and read through that same
        descriptor, so what is uploaded is the file that was measured -- rather than
        whatever the path resolves to a moment later.
        """
        with open(local_path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            if size > self.PARALLEL_THRESHOLD:
                # Each worker reads its own range straight off the descriptor, so
                # a large upload no longer means holding the whole file in memory
                # (`f.read()` on a 10 GB file was 10 GB of resident bytes).
                self._write_ranges_from_file(path, f, size, mode)
            else:
                self.write(path, f, mode=mode)

    def download(self, path: str, local_path: str | Path) -> None:
        """Download a file from the sandbox.

        The bytes land in a private temp file next to the destination and are renamed
        over it at the end, so an interrupted download leaves no truncated file and a
        symlink sitting at `local_path` is replaced rather than followed.
        """
        size = self.stat(path).size
        with _open_download_target(Path(local_path)) as f:
            if size > self.PARALLEL_THRESHOLD:
                # Seek-and-write per range as it completes, so peak memory is a
                # few chunks rather than the whole file.
                for offset, part in self._read_ranges(path, size):
                    f.seek(offset)
                    f.write(part)
                return
            with self._sandbox._stream("GET", "/files/read", params={"path": path}) as response:
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

    def _read_ranges(self, path: str, size: int) -> Iterator[tuple[int, bytes]]:
        """Fetch every range in parallel, yielding `(offset, bytes)` as they land.

        Yielding rather than returning a list is the point: the caller places each
        chunk and drops it, so peak memory is a few chunks instead of the whole
        file plus a copy.
        """

        def fetch(rng: tuple[int, int]) -> tuple[int, bytes]:
            offset, length = rng
            response = self._sandbox._request(
                "GET", "/files/read", params={"path": path, "offset": offset, "length": length}
            )
            return offset, response.content

        ranges = self._ranges(size)
        workers = min(self.PARALLEL_MAX_WORKERS, len(ranges))
        with ThreadPoolExecutor(workers) as executor:
            yield from executor.map(fetch, ranges)

    def _write_ranges(self, path: str, data: bytes, mode: str | None) -> None:
        def push(rng: tuple[int, int]) -> None:
            offset, length = rng
            self._put_range(path, offset, data[offset : offset + length], mode, total=len(data))

        self._parallel(self._ranges(len(data)), push)

    def _write_ranges_from_file(self, path: str, source: BinaryIO, size: int, mode: str | None) -> None:
        """Upload `source` in parallel ranges, reading each range on demand.

        `pread` rather than `seek`+`read`, because the workers share one
        descriptor and a seeking read would race with its siblings.
        """
        fileno = source.fileno()

        def push(rng: tuple[int, int]) -> None:
            offset, length = rng
            self._put_range(path, offset, os.pread(fileno, length, offset), mode, total=size)

        self._parallel(self._ranges(size), push)

    def _put_range(self, path: str, offset: int, chunk: bytes, mode: str | None, *, total: int) -> None:
        params: dict[str, Any] = {"path": path, "offset": offset}
        if mode is not None:
            params["mode"] = mode
        # Tell the server the intended final size. A ranged write does not
        # truncate, so overwriting a larger file with a smaller one used to leave
        # the old bytes past the new end.
        if offset + len(chunk) >= total:
            params["truncate_to"] = total
        self._sandbox._request("PUT", "/files/write", params=params, content=chunk)

    def list(self, path: str) -> List[FileEntry]:
        """List a directory in the sandbox.

        The server paginates, so this follows the cursor and returns everything --
        a caller sees one list regardless of how the directory is chunked on the
        wire.
        """
        entries: List[FileEntry] = []
        after: str | None = None
        while True:
            params: dict[str, Any] = {"path": path}
            if after is not None:
                params["after"] = after
            payload = self._sandbox._request("GET", "/files/list", params=params).json()
            entries.extend(FileEntry(**entry) for entry in payload["entries"])
            after = payload.get("next")
            if not after:
                return entries

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


def _exec_payload(cmd: str | List[str], shell: bool | None) -> dict[str, Any]:
    """Build the `cmd`/`shell` part of an `/exec` payload, validating their consistency."""
    if shell is True and not isinstance(cmd, str):
        raise ValueError("shell=True requires `cmd` to be a shell command string, not a list.")
    if shell is False and isinstance(cmd, str):
        raise ValueError("shell=False requires `cmd` to be an argv list (e.g. ['echo', 'hi']), not a string.")
    payload: dict[str, Any] = {"cmd": cmd}
    if shell is not None:
        payload["shell"] = shell
    return payload


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


def _check_server_protocol(response: httpx.Response, job_id: str) -> None:
    """Refuse a server whose wire contract this client cannot drive.

    Permissive in one direction only. A *newer* server is accepted: it declares its own
    backwards compatibility by keeping the protocol integer it still supports. An *older* one
    is refused, because the mismatches are things like the per-sandbox capability token, which
    surfaces as a 403 on an unrelated route several calls later instead of here.

    `version` is no use for this: it moves for a doc fix as readily as for a protocol break.
    """
    try:
        protocol = response.json().get("protocol")
    except Exception:
        protocol = None
    if isinstance(protocol, int) and not isinstance(protocol, bool) and protocol >= SANDBOX_SERVER_PROTOCOL:
        return
    # No field at all means a server published before protocols were declared, i.e. 1.
    reported = protocol if protocol is not None else "1 (declares no protocol)"
    raise SandboxError(
        f"Sandbox job {job_id} runs sbx-server protocol {reported}, but this version of "
        f"huggingface_hub needs protocol {SANDBOX_SERVER_PROTOCOL} or newer. A pool host keeps the "
        "server binary it downloaded at boot for up to 24h, so recycle this pool's hosts (kill them, "
        "or let them idle-time-out) to pick up the current server."
    )


class _SandboxAuth(httpx.Auth):
    """Attach the *current* HF bearer to every request.

    The bearer used to be captured once when the transport was built. A sandbox
    host lives up to 24h, so an OIDC/OAuth credential can expire or rotate inside
    that window, after which every request to the Jobs proxy failed even though
    the handle was still perfectly good. Reading it per request fixes that.

    The `X-Sandbox-Token` is deliberately *not* re-derived: the server holds the
    value derived from the bearer that created the job, so re-deriving from a
    rotated bearer would produce a token the server has never seen. The two
    credentials answer different gates -- the proxy wants a live bearer, the
    server wants the original capability -- and only the first needs refreshing.
    """

    def __init__(self, api: HfApi) -> None:
        self._api = api

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {_effective_token(self._api)}"
        yield request


class _SandboxServer:
    """HTTP transport to one `sbx-server` instance — a dedicated job or a shared host.

    Owns the `httpx.Client`, the base URL and the auth headers.
    In dedicated mode a server is paired 1:1 with its [`Sandbox`
    In pool mode one server (one host job) is shared by many sandboxes, and `live`/`capacity` track packing.
    """

    def __init__(
        self,
        *,
        job_id: str,
        owner: str,
        image: str | None,
        base_url: str,
        nonce: str,
        sandbox_token: str,
        api: HfApi,
        max_connections: int = 10,
        capacity: int = 0,
    ) -> None:
        self.job_id = job_id
        self.owner = owner
        self._image = image
        self.base_url = base_url
        # Public nonce the sandbox token is derived from; kept so a host can be persisted
        # to (and rebuilt from) the pool cache without re-reading the job labels.
        self.nonce = nonce
        self._api = api
        self._auth_token = _effective_token(api)
        self._sandbox_token = sandbox_token
        # Packing bookkeeping (shared mode only).
        self.capacity = capacity
        self.live = 0
        # False only for hosts rebuilt from the (best-effort) pool cache: their job may
        # be gone, so the first failed request drops them instead of failing the create.
        self.verified = True
        # Whether *this* process started this host job, and may therefore cancel
        # it. Ownership belongs to the host, not to the pool handle: a handle used
        # to cancel hosts it had merely discovered via labels, which may have been
        # started by another process -- or another member of the namespace -- and
        # may be serving their sandboxes.
        self.owned = False
        # httpx.Client is thread-safe, so a single client serves both sequential requests
        # and the concurrent workers used for parallel file transfers / many sandboxes.
        self._client = httpx.Client(
            headers={"X-Sandbox-Token": sandbox_token},
            auth=_SandboxAuth(api),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
            # Every request on this client carries the HF bearer and a sandbox token, and there
            # is no redirect in this protocol -- the Jobs proxy serves the in-job server
            # directly. Following one would hand both credentials to a destination that the
            # response, rather than the client, chose.
            follow_redirects=False,
        )

    @classmethod
    def from_job(
        cls,
        *,
        job: JobInfo,
        nonce: str,
        sandbox_token: str,
        api: HfApi,
        max_connections: int = 10,
        capacity: int = 0,
    ) -> "_SandboxServer":
        """Build a server from a freshly fetched job (reads its exposed server URL)."""
        return cls(
            job_id=job.id,
            owner=job.owner.name,
            image=job.docker_image or job.space_id,
            base_url=_find_server_url(job),
            nonce=nonce,
            sandbox_token=sandbox_token,
            api=api,
            max_connections=max_connections,
            capacity=capacity,
        )

    @property
    def image(self) -> str | None:
        return self._image

    def _with_token(self, kwargs: dict, sandbox_token: str | None) -> dict:
        """Override `X-Sandbox-Token` for one request.

        The client-level header carries the *host* credential, which manages the
        pool. A pooled sandbox's own operations should present that sandbox's
        capability token instead, so nothing wider than the sandbox is offered on
        a per-sandbox call.
        """
        if sandbox_token is not None:
            kwargs["headers"] = {**kwargs.get("headers", {}), "X-Sandbox-Token": sandbox_token}
        return kwargs

    def request(self, method: str, path: str, *, sandbox_token: str | None = None, **kwargs) -> httpx.Response:
        """Request to the in-job server. Raises SandboxError on API errors."""
        timeout = kwargs.pop("timeout", httpx.Timeout(60.0, connect=10.0))
        kwargs = self._with_token(kwargs, sandbox_token)
        response = self._client.request(method, self.base_url + path, timeout=timeout, **kwargs)
        if response.status_code >= 400:
            _raise_for_status(response)
        return response

    @contextmanager
    def stream(
        self, method: str, path: str, *, sandbox_token: str | None = None, **kwargs
    ) -> Iterator[httpx.Response]:
        """Streaming request to the in-job server. Raises SandboxError on API errors."""
        timeout = kwargs.pop("timeout", httpx.Timeout(70.0, connect=10.0))  # server pings every 15s
        kwargs = self._with_token(kwargs, sandbox_token)
        with self._client.stream(method, self.base_url + path, timeout=timeout, **kwargs) as response:
            if response.status_code >= 400:
                _raise_for_status(response)
            yield response

    def sandbox_token(self, local_id: str) -> str | None:
        """Recover a pooled sandbox's capability token, using the host credential.

        Keeps reconnection stateless: `Sandbox.connect` can reattach to a sandbox
        it never created. Returns `None` when the host runs a server that predates
        per-sandbox tokens, in which case the caller falls back to the host token.
        """
        try:
            return self.request("GET", f"/v1/sandboxes/{local_id}/token").json()["token"]
        except SandboxError as e:
            if e.status_code == 404:
                return None  # older sbx-server: no such route
            raise

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
                    _check_server_protocol(response, self.job_id)
                    return
            except httpx.RequestError:
                pass
            if time.time() - last_job_check > 2.0:
                last_job_check = time.time()
                job = self._api.inspect_job(job_id=self.job_id, namespace=self.owner)
                if job.status.stage in _TERMINAL_STAGES:
                    logs = _tail_job_logs(self._api, self.job_id, namespace=self.owner)
                    raise SandboxError(
                        f"Sandbox job {self.job_id} terminated during startup "
                        f"(status: {job.status.stage}, message: {job.status.message}).{logs}"
                    )
            time.sleep(0.15)
        raise SandboxError(f"Sandbox job {self.job_id} did not become ready within {start_timeout:.0f}s.")


class _KillMethod:
    """Lets `kill` work both as a classmethod and an instance method."""

    @overload
    def __get__(self, instance: None, owner: type) -> Callable[..., None]: ...
    @overload
    def __get__(self, instance: "Sandbox", owner: type) -> Callable[[], None]: ...
    def __get__(self, instance: "Sandbox | None", owner: type) -> Callable[..., None]:
        if instance is not None:
            return instance._kill

        def kill(sandbox_id: str, *, namespace: str | None = None, token: str | None = None) -> None:
            owner.connect(sandbox_id, namespace=namespace, token=token).kill()  # type: ignore[attr-defined]

        return kill


class Sandbox:
    """An isolated cloud machine running on Hugging Face Jobs.

    > [!NOTE]
    > The Sandbox API is experimental. Its API and behavior may change without notice.

    Create a dedicated one with [`Sandbox.create`] (one job per sandbox), or get many cheap shared ones from a [`SandboxPool`].
    Reattach to a running sandbox from anywhere with [`Sandbox.connect`]. Use as a context manager to terminate it on exit:

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
        sandbox_token: str | None = None,
    ) -> None:
        self.id = id
        self._server = server
        # Capability token for this sandbox alone (pool mode). Sent instead of the
        # host credential on every per-sandbox call, so a pooled sandbox's
        # operations -- and the headers handed to a port-proxy client -- confer no
        # authority over its siblings or over the pool. None in dedicated mode
        # (where the job is the sandbox, so the job token is already scoped to it)
        # and on hosts running a server that predates per-sandbox tokens.
        self._sandbox_token = sandbox_token
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
        idle_timeout: int | float | str | None = DEFAULT_IDLE_TIMEOUT,
        env: dict[str, Any] | None = None,
        secrets: dict[str, Any] | None = None,
        volumes: List[Volume] | None = None,
        namespace: str | None = None,
        forward_hf_token: bool = False,
        start_timeout: float = 120.0,
        token: str | None = None,
    ) -> "Sandbox":
        """Create a dedicated sandbox (one HF Job) and block until it is ready (~7s on cpu-basic).

        Each sandbox is a full isolated VM, so this is the right choice for GPU
        workloads or untrusted code. To fan out many cheap CPU sandboxes instead, use
        [`SandboxPool`].

        The job runs with a fixed 24h maximum lifetime; `idle_timeout` is the real
        keeper — an idle sandbox shuts itself down well before that.

        Args:
            image (`str`, *optional*, defaults to `"python:3.12"`):
                Any Docker image with `/bin/sh` (Docker Hub or `hf.co/spaces/...`).
            flavor (`str`, *optional*, defaults to `"cpu-basic"`):
                Hardware flavor, e.g. `"cpu-basic"`, `"a10g-small"`. See `hf jobs hardware`.
            idle_timeout (`int` or `float` or `str`, *optional*, defaults to `600`):
                Auto-shutdown after this much inactivity (no API calls, no running
                processes). Defaults to 10 minutes; pass `None` to disable. Note that a
                *foreground* command is not currently counted as activity, so a single
                `run()` that takes longer than this without other API traffic can have its
                sandbox shut down under it — raise the timeout (or pass `None`) for long
                single commands.
            env (`dict[str, Any]`, *optional*):
                Environment variables available in the sandbox.
            secrets (`dict[str, Any]`, *optional*):
                Secret environment variables (encrypted server-side).
            volumes (`List[Volume]`, *optional*):
                HF repos/buckets to mount, see [`Volume`].
            namespace (`str`, *optional*):
                User or org namespace to run under (defaults to current user).
            forward_hf_token (`bool`, *optional*, defaults to `False`):
                If True, your HF token is injected as `HF_TOKEN` (opt-in).
            start_timeout (`float`, *optional*, defaults to `120.0`):
                Max seconds to wait for the sandbox to become ready.
            token (`str`, *optional*):
                HF token override.

        The image only needs `/bin/sh`. The sandbox server is downloaded at startup with
        `wget`/`curl` if available, otherwise read off an always-mounted server bucket (which
        adds ~2-3s to cold start, so shipping `wget`/`curl` keeps it fast).
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
            sandbox_token=sandbox_token,
        )

        job = api.run_job(
            image=image,
            command=command,
            env=job_env,
            secrets=job_secrets,
            flavor=flavor,
            timeout=SANDBOX_MAX_LIFETIME,
            labels={SANDBOX_LABEL: "1", MODE_LABEL: MODE_DEDICATED, NONCE_LABEL: nonce},
            volumes=job_volumes or None,
            expose=[SANDBOX_SERVER_PORT],
            namespace=namespace,
        )
        server: "_SandboxServer | None" = None
        try:
            server = _SandboxServer.from_job(
                job=job,
                nonce=nonce,
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
        """Reattach to a running sandbox from anywhere, using only its id."""
        api = HfApi(token=token)
        sandbox_id, namespace = _split_sandbox_id(sandbox_id, namespace)
        if SHARED_ID_SEP in sandbox_id:
            host_job_id, local_id = sandbox_id.split(SHARED_ID_SEP, 1)
            server = _connect_host(api, host_job_id, namespace=namespace)
            try:
                existing = {item["id"] for item in server.request("GET", "/v1/sandboxes").json()}
                if local_id not in existing:
                    raise SandboxError(f"Sandbox {sandbox_id} no longer exists on host {host_job_id}.")
                # Recover the sandbox's capability token with the host credential, so
                # this handle operates with the narrow one from here on.
                sandbox_token = server.sandbox_token(local_id)
            except Exception:
                server.close()  # don't leak the HTTP client when the host is gone/unreachable
                raise
            return cls(
                id=sandbox_id,
                server=server,
                local_id=local_id,
                owns_sandbox=False,
                owns_server=True,
                sandbox_token=sandbox_token,
            )

        job = api.inspect_job(job_id=sandbox_id, namespace=namespace)
        labels = job.labels or {}
        nonce = labels.get(NONCE_LABEL)
        if labels.get(SANDBOX_LABEL) != "1" or nonce is None:
            raise SandboxError(f"Job {sandbox_id} is not a sandbox (missing '{SANDBOX_LABEL}' label).")
        if labels.get(MODE_LABEL) == MODE_POOL:
            raise SandboxError(
                f"Job {sandbox_id} is a sandbox host, not a single sandbox. Connect to one of its "
                f"sandboxes with id '<host_job_id>{SHARED_ID_SEP}<local_id>'."
            )
        if job.status.stage != "RUNNING":
            raise SandboxError(f"Sandbox {sandbox_id} is not running (status: {job.status.stage}).")
        sandbox_token = _derive_sandbox_token(_effective_token(api), nonce)
        server = _SandboxServer.from_job(
            job=job,
            nonce=nonce,
            sandbox_token=sandbox_token,
            api=api,
            max_connections=SandboxFiles.PARALLEL_MAX_WORKERS + 2,
        )
        return cls(id=job.id, server=server, local_id=None, owns_sandbox=False, owns_server=True)

    # `kill` is both a classmethod (`Sandbox.kill(id)`) and an instance method (`sbx.kill()`),
    # dispatched by the `_KillMethod` descriptor; `_kill` holds the instance behaviour.
    kill = _KillMethod()

    def _kill(self) -> None:
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
                self._server.request("DELETE", f"/v1/sandboxes/{self._local_id}", sandbox_token=self._sandbox_token)
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

    @overload
    def run(
        self,
        cmd: str | List[str],
        *,
        shell: bool | None = ...,
        env: dict[str, Any] | None = ...,
        cwd: str | None = ...,
        timeout: float | None = ...,
        stdin: str | None = ...,
        on_stdout: Callable[[str], None] | None = ...,
        on_stderr: Callable[[str], None] | None = ...,
        check: bool = ...,
        capture_output: bool = ...,
        background: Literal[False] = ...,
    ) -> SandboxCommandResult: ...

    @overload
    def run(
        self,
        cmd: str | List[str],
        *,
        shell: bool | None = ...,
        env: dict[str, Any] | None = ...,
        cwd: str | None = ...,
        background: Literal[True],
    ) -> SandboxProcess: ...

    def run(
        self,
        cmd: str | List[str],
        *,
        shell: bool | None = None,
        env: dict[str, Any] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        stdin: str | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
        check: bool = True,
        capture_output: bool = True,
        background: bool = False,
    ) -> SandboxCommandResult | SandboxProcess:
        """Run a command in the sandbox and wait for it, streaming output live.

        With `background=True` the command is started detached and `run` returns a
        [`SandboxProcess`] immediately, without waiting for it to finish — handy for
        servers and other long-running processes. List them later with [`Sandbox.processes`]
        and stop one with [`SandboxProcess.kill`]. The streaming/wait-only options
        (`timeout`, `stdin`, `on_stdout`, `on_stderr`, `check`) don't apply in that mode.

        Args:
            cmd (`str` or `List[str]`):
                A shell command string (run with `/bin/sh -c`) or an argv list (exec'd directly).
            shell (`bool`, *optional*):
                Force the execution mode instead of inferring it from the type of `cmd`.
                `True` runs through `/bin/sh -c` and requires `cmd` to be a string; `False`
                exec's `cmd` directly and requires it to be an argv list. `None` (default)
                infers from the type. Set it explicitly to avoid the type-driven footgun (e.g.
                `["echo hi"]` being exec'd as a single program named `"echo hi"`).
            env (`dict[str, Any]`, *optional*):
                Extra environment variables for this command.
            cwd (`str`, *optional*):
                Working directory.
            timeout (`float`, *optional*):
                Kill the command (whole process group) after this many seconds.
            stdin (`str`, *optional*):
                Data to write to the command's stdin.
            on_stdout (`Callable[[str], None]`, *optional*):
                Callback invoked with stdout chunks as they arrive.
            on_stderr (`Callable[[str], None]`, *optional*):
                Callback invoked with stderr chunks as they arrive.
            check (`bool`, *optional*, defaults to `True`):
                If True, raise [`SandboxCommandError`] on non-zero exit.
            capture_output (`bool`, *optional*, defaults to `True`):
                If True, accumulate stdout/stderr into the returned result. Pass `False`
                when you only want `on_stdout`/`on_stderr`: output is then handed to the
                callbacks and dropped, so a command producing gigabytes does not have to fit
                in memory. `result.stdout`/`result.stderr` are empty in that case.
            background (`bool`, *optional*, defaults to `False`):
                If True, start the command detached and return a [`SandboxProcess`] right
                away instead of waiting for it and returning a [`SandboxCommandResult`].

        Returns: a [`SandboxCommandResult`] (with `exit_code`, `stdout`, `stderr`,
        `duration_ms`), or a [`SandboxProcess`] when `background=True`.
        """
        payload = _exec_payload(cmd, shell)
        if env:
            payload["env"] = env
        if cwd:
            payload["cwd"] = cwd
        if background:
            data = self._request("POST", "/processes", json=payload).json()
            return SandboxProcess(
                # Absent on a server that predates opaque ids; `kill()` then says so
                # rather than sending a pid the server will reject.
                id=data.get("id"),
                pid=data["pid"],
                cmd=cmd,
                tag=data.get("tag"),
                _sandbox=self,
            )
        if timeout is not None:
            payload["timeout"] = timeout
        if stdin is not None:
            payload["stdin"] = stdin

        # Output was accumulated unconditionally, even when a callback was
        # consuming it -- so a streaming consumer still paid full memory for the
        # command's entire output. `capture_output=False` opts out.
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        captured = 0
        result: SandboxCommandResult | None = None

        def keep(parts: list[str], data: str) -> None:
            nonlocal captured
            if not capture_output:
                return
            if captured + len(data) > MAX_CAPTURED_OUTPUT_CHARS:
                raise SandboxError(
                    f"command produced more than {MAX_CAPTURED_OUTPUT_CHARS} characters of output. "
                    "Pass `capture_output=False` with `on_stdout`/`on_stderr` to stream it instead, "
                    "or redirect it to a file in the sandbox and download that."
                )
            captured += len(data)
            parts.append(data)

        with self._stream("POST", "/exec", json=payload) as response:
            for event in _iter_events(response):
                if event["event"] == "stdout":
                    keep(stdout_parts, event["data"])
                    if on_stdout is not None:
                        on_stdout(event["data"])
                elif event["event"] == "stderr":
                    keep(stderr_parts, event["data"])
                    if on_stderr is not None:
                        on_stderr(event["data"])
                elif event["event"] == "exit":
                    result = SandboxCommandResult(
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

    def processes(self) -> List[SandboxProcess]:
        """List the background processes of this sandbox.

        Returns the processes started with [`Sandbox.run`]`(..., background=True)`; stop one
        with [`SandboxProcess.kill`]. Recently completed processes stay listed (with
        `running=False` and their `exit_code`); the server keeps a bounded number of them, so a
        sandbox that has run thousands of short commands will not list them all.
        """
        data = self._request("GET", "/processes").json()
        return [
            SandboxProcess(
                id=p.get("id"),
                pid=p["pid"],
                cmd=p["cmd"],
                tag=p.get("tag"),
                started_at_ms=p.get("started_at_ms"),
                running=p["running"],
                exit_code=p.get("exit_code"),
                _sandbox=self,
            )
            for p in data
        ]

    # ------------------------------------------------------------------ misc

    @property
    def image(self) -> str | None:
        return self._server.image

    @property
    def host_id(self) -> str | None:
        """For a shared/pool sandbox, the job id of the host running it (else None)."""
        return self._server.job_id if self._local_id is not None else None

    # ------------------------------------------------------------------ port proxy

    def proxy_url_for(self, port: int | str, path: str = "/", *, scheme: str = "https://") -> str:
        """Public URL that proxies through to a server running *inside* this sandbox.

        Requests to the returned URL are forwarded by the in-job sandbox server to a
        server you started in the sandbox on `port`, including WebSocket (`ws(s)://`)
        upgrades and streamed responses. Pair it with [`proxy_headers`] for auth.

        How the sandbox must listen on `port`:

        - **Pool / shared sandbox**: it cannot bind a TCP port (Landlock), so bind a
          **unix socket** at `$SBX_PROXY_DIR/<port>.sock` (the `SBX_PROXY_DIR` env var
          is set in every sandbox). E.g. `uvicorn app:app --uds $SBX_PROXY_DIR/8000.sock`.
        - **Dedicated sandbox**: bind a normal TCP port on `127.0.0.1:<port>`. (You can
          also expose the port directly via the job proxy without going through here.)

        Args:
            port (`int` or `str`):
                The port (pool: the `<port>` of the unix socket) the inner server listens on.
            path (`str`, *optional*, defaults to `"/"`):
                Path on the inner server to point at, e.g. `"/ws"`.
            scheme (`str`, *optional*, defaults to `"https://"`):
                URL scheme to build the link with. Defaults to `"https://"`; pass
                `"wss://"` for a WebSocket client (the proxy is protocol-agnostic, so only
                the client-side scheme changes).

        Returns:
            `str`: a URL like `https://<job_id>--49983.hf.jobs/v1/.../proxy/8000/ws` (or
            `wss://...` with `scheme="wss://"`).

        Example:
            ```python
            >>> url = sandbox.proxy_url_for(8000, "/ws", scheme="wss://")
            >>> import websockets
            >>> async with websockets.connect(url, additional_headers=sandbox.proxy_headers) as ws:
            ...     await ws.send("hello")
            ```
        """
        # base_url is always https://...; swap in the requested scheme (e.g. wss://) for the client.
        host_and_rest = self._server.base_url.split("://", 1)[-1]
        path = path if path.startswith("/") else "/" + path
        return f"{scheme}{host_and_rest}{self._base_path}/proxy/{port}{path}"

    @property
    def proxy_headers(self) -> dict[str, str]:
        """Auth headers to send with [`proxy_url_for`] requests (HF token + sandbox token).

        Read these at the moment you use them: the HF token is resolved on access,
        so a long-lived handle hands out a current bearer rather than one captured
        when the sandbox was created.

        For a pooled sandbox this is the sandbox's own capability token, not the
        pool host's: these headers typically end up in a browser or WebSocket
        client, so they should confer access to this sandbox and nothing else.
        """
        return {
            "Authorization": f"Bearer {_effective_token(self._server._api)}",
            "X-Sandbox-Token": self._sandbox_token or self._server._sandbox_token,
        }

    def __repr__(self) -> str:
        return f"Sandbox(id={self.id!r}, image={self.image!r})"

    # ------------------------------------------------------------------ internals

    def _request(self, method: str, resource: str, **kwargs) -> httpx.Response:
        return self._server.request(method, self._base_path + resource, sandbox_token=self._sandbox_token, **kwargs)

    @contextmanager
    def _stream(self, method: str, resource: str, **kwargs) -> Iterator[httpx.Response]:
        with self._server.stream(
            method, self._base_path + resource, sandbox_token=self._sandbox_token, **kwargs
        ) as response:
            yield response


class SandboxPool:
    """A fleet of shared "host" jobs, each packing many landlock-isolated sandboxes.

    > [!WARNING]
    > The Sandbox API is experimental. Its API and behavior may change without notice.
    >
    > **Pooled sandboxes are for workloads inside one trust boundary.** A pooled sandbox is a uid plus a Landlock
    > ruleset inside a shared VM — not a VM of its own — and every sandbox on a host shares that host's auth token
    > and its privileged control plane. Use a pool to fan out *your own* code cheaply. For mutually distrusting
    > workloads, use [`Sandbox.create`], which gives each one its own VM. The specific gaps are listed under
    > "Known limitations" in the sandbox conceptual guide.

    One host is one billed HF Job (a VM); it runs the sandbox server and multiplexes
    up to `sandboxes_per_host` lightweight sandboxes, isolated from each other by
    uid + the Landlock LSM. This makes large fan-outs cheap (the VM cost is shared
    across all its sandboxes) and fast (creating a sandbox is ~one proxy round-trip
    once a host is warm). Best for many parallel CPU sandboxes such as RL rollouts;
    for GPU or strong VM-level isolation between mutually-distrusting workloads, use
    [`Sandbox.create`] instead.

    The constructor pre-provisions `warm_up` hosts (default 1) and blocks until they are
    ready; further hosts are then provisioned on demand as sandboxes are requested, and all
    are torn down on `close()` (or when idle, via `idle_timeout`). The user never manages jobs:

    ```python
    >>> from huggingface_hub import SandboxPool
    >>> with SandboxPool(image="python:3.12", flavor="cpu-basic", warm_up=2) as pool:
    ...     boxes = [pool.create() for _ in range(100)]   # packed across the warm hosts
    ...     print(boxes[0].run("echo hi").stdout)
    hi
    ```

    `create()` makes **one** sandbox at a time: it reuses a host that still has free
    capacity before booting a new one, so you grow on demand as work arrives. To avoid
    a cold start on the first few calls, pre-provision hosts with `warm_up` (or
    [`warm`]). Warm hosts are discovered via job labels, so reuse works **across
    processes** too (a fresh pool with the same `image`/`flavor`/`name` attaches to
    hosts an earlier run left behind):

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
        warm_up: int = 1,
        max_hosts: int | None = None,
        name: str | None = None,
        idle_timeout: int | float | str | None = DEFAULT_IDLE_TIMEOUT,
        namespace: str | None = None,
        start_timeout: float = 120.0,
        adopt_hosts: str = ADOPT_OWN,
        token: str | None = None,
        _connect_mode: bool = False,
    ) -> None:
        """Configure a pool and pre-provision `warm_up` hosts (blocks until they are ready).

        Env/secrets are *not* set here: they belong to each sandbox and are passed to
        `create(env=...)`, so sandboxes in the same pool can have different environments.

        Args:
            image (`str`, *optional*, defaults to `"python:3.12"`):
                Docker image for the hosts (needs `/bin/sh`). All sandboxes in the
                pool share this image.
            flavor (`str`, *optional*, defaults to `"cpu-basic"`):
                Hardware flavor for the host jobs (e.g. `"cpu-basic"`).
            sandboxes_per_host (`int`, *optional*, defaults to `50`):
                How many sandboxes to pack per host (per VM density).
            warm_up (`int`, *optional*, defaults to `1`):
                How many hosts to pre-provision in the constructor (which blocks
                until they are ready), so an initial burst of `create()` calls doesn't pay
                a host cold start each. Existing warm hosts (from the cache / other processes)
                count towards it, so only the shortfall is booted; capped by `max_hosts`.
                Defaults to 1 (a single host).
            max_hosts (`int`, *optional*):
                Optional cap on the number of host jobs (a cost ceiling). When
                reached and all hosts are full, `create()` raises. The cap is enforced
                per-process against the hosts this handle knows about, so two processes
                using the same pool concurrently can exceed it. (Per-host density,
                `sandboxes_per_host`, *is* enforced by the host itself.)
            name (`str`, *optional*):
                Pool name, used as the `hf-sandbox-pool` job label so the pool is
                discoverable (e.g. `hf sandbox pool ls`, `connect()`). `create()` reuses
                running hosts carrying this label (including from other processes) before
                booting new ones, so distinct names keep separate pools from sharing hosts.
                A random name is generated when omitted.
            idle_timeout (`int` or `float` or `str`, *optional*, defaults to `600`):
                Host idle timeout — a host shuts down once it has had no
                sandboxes for this long (a billing backstop). Each sandbox also has its
                own idle timeout, set at `create()`. Pass `None` to disable.
            namespace (`str`, *optional*):
                User or org namespace to run hosts under.
            start_timeout (`float`, *optional*, defaults to `120.0`):
                Max seconds to wait for a host to become ready.
            adopt_hosts (`str`, *optional*, defaults to `"own"`):
                Which already-running hosts `create()` may pack onto. Hosts are found by
                filtering Jobs on labels, and labels are set by whoever creates the Job — so a
                label match is a claim, not proof. This decides how much more than the claim is
                required:

                - `"own"` (default): only Jobs this principal started, per the Jobs API's
                  `initiator`. Safe in a shared namespace.
                - `"namespace"`: any Job in the namespace carrying the pool's labels, provided
                  its image, flavor, command and exposed URL all match. Restores cross-user
                  host sharing; only use it in a namespace whose members you trust, since any
                  of them can publish a Job your client will then send a token to.
                - `"never"`: no adoption at all — this handle only uses hosts it booted itself.
            token (`str`, *optional*):
                HF token override.
        """
        if adopt_hosts not in (ADOPT_OWN, ADOPT_NAMESPACE, ADOPT_NEVER):
            raise ValueError(
                f"adopt_hosts must be one of {ADOPT_OWN!r}, {ADOPT_NAMESPACE!r}, {ADOPT_NEVER!r}; got {adopt_hosts!r}."
            )
        if sandboxes_per_host < 1:
            raise ValueError("sandboxes_per_host must be >= 1.")
        if warm_up < 1:
            raise ValueError("warm_up must be >= 1.")
        self._api = HfApi(token=token)
        self.image = image
        self.flavor = flavor
        self.sandboxes_per_host = sandboxes_per_host
        self._warm_up = warm_up
        self.max_hosts = max_hosts
        self.name = name if name is not None else f"pool-{token_hex(6)}"
        self._idle_timeout = idle_timeout
        self._namespace = namespace
        # Security context the local cache is keyed by: entries written by another endpoint,
        # credential or namespace are a miss, not a hit (see `CacheContext`).
        self._cache_context = _cache_context(self._api, namespace)
        self._start_timeout = start_timeout
        self._adopt_hosts = adopt_hosts
        # Resolved lazily and only when `adopt_hosts="own"` needs it: whoami() is
        # heavily rate-limited, and a pool that never discovers never needs it.
        self._principal: str | None = None
        self._principal_resolved = False
        self._hosts: List[_SandboxServer] = []
        self._lock = threading.Lock()
        # Held across the whole one-time warm-up so concurrent first create() calls block until
        # it finishes (and see the warm hosts) instead of racing past a half-set flag.
        self._warmup_lock = threading.Lock()
        # Serializes on-demand host creation within the process: a burst of create() calls that
        # all find every host full boots one host at a time (each new host frees
        # `sandboxes_per_host` slots for the threads still queued behind the lock) instead of
        # one host per thread. The CLI is a single process per command, so this is a no-op there.
        self._boot_lock = threading.Lock()
        self._closed = False
        # Whether the one-time warm-up (seed cache + pre-provision `warm_up` hosts) ran.
        self._warmed_up = False
        # Set for connect()'d handles: the pool must already exist, so create() refuses to
        # silently resurrect it by booting a fresh host if discovery confirms every host is gone.
        # Such a handle also never boots during construction — it attaches lazily on create().
        self._require_live_host = _connect_mode
        # Whether close()/`__exit__` cancels the host jobs. True for a pool we created; False
        # for a connect()'d handle, which only releases its HTTP clients on exit — the shared
        # hosts (possibly serving other clients) are left running, like Sandbox.connect().
        # Kept as the default for hosts this handle boots (a connect()'d handle
        # boots nothing during construction), but teardown is decided per host.
        self._owns_hosts = not _connect_mode
        # In-flight create/boot operations. `close()` waits for these to drain, so
        # a host booted by a create() that was already past its closed check
        # cannot land in a list nobody will ever tear down.
        self._in_flight = 0
        self._drained = threading.Condition(self._lock)
        # Job ids of cached hosts we found dead this session; pruned from the cache on save.
        self._dead_host_ids: set[str] = set()

        # Pre-provision `warm_up` hosts now so the constructor returns only once the pool is
        # warm. A connect()'d handle skips this: it must never boot during construction (it can
        # only attach to existing hosts, lazily on the first create()).
        if not _connect_mode:
            self._ensure_warmed_up()

    # ------------------------------------------------------------------ public API

    @classmethod
    def connect(
        cls,
        pool_id: str,
        *,
        namespace: str | None = None,
        adopt_hosts: str = ADOPT_OWN,
        token: str | None = None,
    ) -> "SandboxPool":
        """Reattach to a running pool by id, from any machine — no local state needed.

        Finds a running host labelled with `pool_id` and rebuilds the pool's config
        (image/flavor/density/host-idle) from that host job's spec and env vars, returning
        a [`SandboxPool`] ready to `create()` more sandboxes — packing onto the running
        hosts, or booting a duplicate (same config) when they are full.

        Raises [`SandboxError`] if no running host is found (a pool stops existing once
        all of its hosts are gone — idle-timed-out or killed).

        Args:
            pool_id (`str`):
                The id returned when the pool was first created.
            namespace (`str`, *optional*):
                Namespace to search for the pool's hosts (defaults to yours).
            adopt_hosts (`str`, *optional*, defaults to `"own"`):
                Which hosts may be attached to. See [`SandboxPool`]. Reattaching to a pool whose
                hosts another member of the namespace started needs `"namespace"`.
            token (`str`, *optional*):
                HF token override.
        """
        api = HfApi(token=token)
        # Fast path: rebuild the pool from the local best-effort cache, with no HTTP at all.
        # Only an entry written by this endpoint, credential and namespace is a hit (see
        # `CacheContext`), so `namespace=` is honoured rather than being overridden by whatever
        # the file says -- a `connect(namespace="org-B")` can no longer be served org-A's hosts.
        # The cached hosts are seeded (and validated) lazily on the next create(); if they are
        # all stale, create() falls back to label discovery exactly like a cold connect would.
        cache = read_pool_cache(pool_id, _cache_context(api, namespace))
        if cache is not None:
            return cls(
                image=cache.image,
                flavor=cache.flavor,
                sandboxes_per_host=cache.sandboxes_per_host,
                max_hosts=cache.max_hosts,
                name=pool_id,
                idle_timeout=cache.idle_timeout,
                namespace=namespace,
                adopt_hosts=adopt_hosts,
                token=token,
                _connect_mode=True,  # attach to existing hosts; never boot during construction
            )

        # Cold path: find a running host via labels and rebuild the config from its job spec.
        job = _find_pool_host_job(api, pool_id, namespace=namespace, policy=adopt_hosts)
        env = _host_env(api, job, namespace=namespace)
        idle_raw = env.get("SBX_IDLE_TIMEOUT")
        max_hosts_raw = env.get("SBX_MAX_HOSTS")
        return cls(
            image=job.docker_image or job.space_id or DEFAULT_IMAGE,
            flavor=str(job.flavor) if job.flavor is not None else "cpu-basic",
            sandboxes_per_host=int(env.get("SBX_CAPACITY", DEFAULT_SANDBOXES_PER_HOST)),
            max_hosts=int(max_hosts_raw) if max_hosts_raw is not None else None,
            name=pool_id,
            idle_timeout=int(idle_raw) if idle_raw is not None else None,
            namespace=namespace,
            adopt_hosts=adopt_hosts,
            token=token,
            _connect_mode=True,  # attach to existing hosts; never boot during construction
        )

    def warm(self, num_hosts: int = 1) -> List[str]:
        """Ensure `num_hosts` empty host(s) are running and leave them running. Returns the
        pool's host job ids.

        Used to "create" a pool up front: the hosts carry the pool label and config (in
        their env vars), so a later `SandboxPool.connect(pool_id)` (even from another
        machine) finds them and spawns sandboxes without a cold start. The hosts keep
        billing until killed or idle.

        Adopts hosts already running for this pool (found via job labels) before booting,
        so a `warm()` after `connect()` — or a repeated `warm()` — tops up to `num_hosts`
        instead of duplicating live hosts and blowing past `max_hosts`.
        """
        if self._closed:
            raise SandboxError("This SandboxPool is closed.")
        self._discover_hosts()
        with self._lock:
            shortfall = num_hosts - len(self._hosts)
        if shortfall > 0:
            booted = self._provision_hosts(shortfall)
            with self._lock:
                self._hosts.extend(booted)
        with self._warmup_lock:
            self._warmed_up = True  # explicit warm-up satisfies create()'s one-time warm-up
        self._save_cache()
        return self.host_ids

    def create(
        self,
        *,
        env: dict[str, Any] | None = None,
        idle_timeout: int | float | str | None = DEFAULT_IDLE_TIMEOUT,
        forward_hf_token: bool = False,
    ) -> "Sandbox":
        """Create one sandbox, provisioning a host if needed.

        Reuses a host with free capacity (this pool's, or a warm host found via job labels
        / the local cache) before booting a new one, so a `create()` against a warm host
        costs ~one round-trip. Call it repeatedly to fan out; use `warm_up` (or [`warm`])
        to pre-provision hosts and avoid a cold start on the first calls. If a host fills
        up under us (another process packed it) or a cached host is gone, the sandbox is
        re-placed on another host (or a fresh one).

        Args:
            env (`dict[str, Any]`, *optional*):
                Environment variables for this sandbox (each sandbox gets its own).
            idle_timeout (`int` or `float` or `str`, *optional*, defaults to `600`):
                Per-sandbox idle timeout — a sandbox is evicted from its host
                after this much inactivity (no API calls, no running process). Distinct
                from the host idle timeout. Pass `None` to disable.
            forward_hf_token (`bool`, *optional*, defaults to `False`):
                If True, inject your HF token as `HF_TOKEN` in the sandbox
                (opt-in). Unlike a dedicated sandbox's `secrets`, a pooled sandbox's env is
                delivered to the host server at creation (never stored in the host job), so
                it doesn't appear in any job's metadata.
        """
        with self._lock:
            if self._closed:
                raise SandboxError("This SandboxPool is closed.")
            # Registered under the lock together with the closed check, so
            # `close()` cannot snapshot the host list between the two.
            self._in_flight += 1
        try:
            return self._create_locked(env, idle_timeout, forward_hf_token)
        finally:
            with self._lock:
                self._in_flight -= 1
                self._drained.notify_all()

    def _create_locked(
        self,
        env: dict[str, Any] | None,
        idle_timeout: int | float | str | None,
        forward_hf_token: bool,
    ) -> "Sandbox":
        sandbox_env = dict(env or {})
        if forward_hf_token:
            sandbox_env["HF_TOKEN"] = _effective_token(self._api)
        # None disables per-sandbox idle eviction (mirrors Sandbox.create): leave it out of
        # the create body entirely rather than sending 0, which the server reads as "evict now".
        idle_secs = _duration_to_secs(idle_timeout) if idle_timeout is not None else None

        new_hosts: List[_SandboxServer] = []
        discovered = False
        rounds = 0
        try:
            # One-time warm-up: seed from cache (no HTTP) and pre-provision `warm_up` hosts.
            # Warm-up hosts are pool-level (like `warm()`), so they survive a failed create().
            self._ensure_warmed_up()
            while True:
                # 1. Reserve a slot on a host we already track.
                host = self._reserve_one()
                # 2. If none free, attach (once) to warm hosts discovered via labels.
                if host is None and not discovered:
                    discovered = True
                    self._discover_hosts()
                    host = self._reserve_one()
                # 3. Still none? Boot one host (unless we must not resurrect a dead pool).
                if host is None:
                    if self._require_live_host and discovered and not self._hosts:
                        # connect()'d to a pool whose hosts are all gone (stale cache + nothing
                        # found via labels): don't resurrect it under the same id, just report it.
                        delete_pool_cache(self.name, self._cache_context)
                        raise SandboxError(
                            f"No running host found for pool '{self.name}'. The pool has stopped "
                            "(all its hosts were killed or idle-timed-out); create a new one."
                        )
                    host = self._boot_one_host(new_hosts)
                # 4. Create the sandbox on the reserved host. `None` means the host filled up
                #    (another process packed it) or a stale cached host is gone — retry.
                if host is not None:
                    try:
                        sandbox = self._create_one(host, sandbox_env, idle_secs)
                    except Exception:
                        # The create POST failed on a live host: release the slot we optimistically
                        # reserved so the host isn't left permanently one sandbox short.
                        with self._lock:
                            host.live = max(0, host.live - 1)
                        raise
                    if sandbox is not None:
                        self._save_cache()
                        return sandbox
                    # sandbox is None: the host is full (packed by another process) or gone.
                    # Either way it's no longer a host we exclusively own, so stop tracking it
                    # for teardown — cancelling it on a later round's failure would kill another
                    # tenant's sandboxes (full host) or hit an already-dead job (gone host).
                    if host in new_hosts:
                        new_hosts.remove(host)
                discovered = False  # re-scan for room / boot a host next round
                rounds += 1
                if rounds > _MAX_PACK_ROUNDS:
                    raise SandboxError(
                        "Could not place sandbox: hosts kept reporting full. Raise max_hosts or sandboxes_per_host."
                    )
        except Exception:
            # All-or-nothing: tear down hosts we booted in this call.
            with self._lock:
                self._hosts = [h for h in self._hosts if h not in new_hosts]
            for host in new_hosts:
                try:
                    host.cancel_job()
                except Exception:
                    pass
                finally:
                    host.close()
            raise

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
        """Release the pool. Idempotent.

        For a pool we created, this terminates all host jobs (and therefore all their
        sandboxes). For a `connect()`'d handle it only releases the local HTTP clients: the
        shared hosts may be serving other clients, so — like [`Sandbox.connect`] — leaving a
        `with` block must not tear them down. Terminate a connected pool's hosts explicitly
        with `hf sandbox pool delete <id>`.

        Only hosts *this handle started* are cancelled. A host discovered via labels may
        be serving another process' sandboxes, so it is released rather than terminated.

        Raises [`SandboxError`] if a host job could not be cancelled, naming the jobs that
        are still running — they keep billing, and their cache entries are kept so they stay
        discoverable. When `close()` is reached through `__exit__` with an exception already
        in flight, the failure is logged instead, so it cannot mask the original error.
        """
        with self._lock:
            self._closed = True
            # Wait for in-flight creates so a host booted by one that was already
            # past its closed check lands in `self._hosts` before the snapshot,
            # rather than being orphaned. Bounded, because a wedged create must
            # not hang process exit.
            if self._in_flight and not self._drained.wait_for(
                lambda: self._in_flight == 0, timeout=_CLOSE_DRAIN_TIMEOUT
            ):
                logger.warning(
                    f"Pool '{self.name}': {self._in_flight} operation(s) still in flight after "
                    f"{_CLOSE_DRAIN_TIMEOUT}s; any host they boot will keep running."
                )
            hosts = self._hosts
            self._hosts = []

        survivors: List[_SandboxServer] = []
        failures: List[str] = []
        for host in hosts:
            try:
                # Only what this handle started. A discovered host may be serving
                # someone else's sandboxes, and cancelling it kills them.
                if host.owned:
                    host.cancel_job()
            except Exception as e:
                # Keep it in the cache: that record is the user's only handle on a
                # job that is still running and still billing.
                survivors.append(host)
                failures.append(f"{host.job_id} ({e})")
                logger.warning(f"Failed to cancel sandbox host {host.job_id}: {e}")
            finally:
                host.close()

        if survivors:
            # Rewrite rather than delete, so the survivors stay discoverable.
            self._save_cache(hosts=survivors)
        elif self._owns_hosts:
            # The pool's hosts are gone; don't leave a stale entry.
            delete_pool_cache(self.name, self._cache_context)
        if failures:
            raise SandboxError(
                f"Pool '{self.name}' could not terminate host job(s): {'; '.join(failures)}. "
                "They are still running and billing; cancel them with `hf jobs cancel <id>`."
            )

    def __enter__(self) -> "SandboxPool":
        return self

    def __exit__(self, *exc_info) -> None:
        try:
            self.close()
        except SandboxError:
            # Never let a teardown failure mask the exception that is already
            # propagating out of the `with` block.
            if exc_info[0] is None:
                raise
            logger.warning(f"Pool '{self.name}' teardown failed while handling another error.", exc_info=True)

    # ------------------------------------------------------------------ internals

    def _ensure_warmed_up(self) -> None:
        """One-time: seed from the cache and pre-provision up to `warm_up` hosts.

        Cheap by default: seeding is local (no HTTP), and if the cache already gives at
        least `warm_up` hosts we trust them and skip discovery/booting (dead ones are
        pruned lazily by `create()`). Only when short do we list_jobs and boot the
        shortfall in parallel, capped by `max_hosts`.

        Runs at most once per pool, under `_warmup_lock`: concurrent first `create()` calls
        block here until the warm-up completes — and see its hosts — rather than racing past
        a half-set flag and each booting their own. `_warmed_up` is only set once the work
        succeeds, so a failed warm-up is retried (seeding/discovery dedupe by job id).

        Warm-up hosts are pool-level (like [`warm`]): they are not torn down if the
        triggering `create()` later fails — `close()` (or the `with` block) reclaims them.
        """
        with self._warmup_lock:
            if self._warmed_up:
                return
            self._seed_hosts_from_cache()
            target = self._warm_up if self.max_hosts is None else min(self._warm_up, self.max_hosts)
            with self._lock:
                shortfall = target - len(self._hosts)
            if shortfall > 0:
                self._discover_hosts()
                # A connect()'d pool must never boot during warm-up: it may only attach to
                # existing hosts. Booting is left to create()'s loop, guarded by
                # `_require_live_host`, so a stopped pool is reported instead of resurrected.
                if not self._require_live_host:
                    with self._lock:
                        shortfall = target - len(self._hosts)
                    if shortfall > 0:
                        booted = self._provision_hosts(shortfall)
                        with self._lock:
                            self._hosts.extend(booted)
            self._warmed_up = True
        # Persist the warmed hosts so the next process (e.g. `hf sandbox create --pool` after a
        # bare `hf sandbox pool create`) hits the cache fast path instead of re-discovering.
        if self._hosts:
            self._save_cache()

    def _principal_id(self) -> str | None:
        """Id of the principal this pool authenticates as, for the `"own"` policy."""
        if not self._principal_resolved:
            self._principal_resolved = True
            try:
                self._principal = self._api.whoami(cache=True).get("id")
            except Exception as e:
                # Leave it None: `_adoptable` then refuses rather than guessing.
                logger.warning(f"Could not resolve the current user to validate pool hosts: {e}")
        return self._principal

    def _adoptable(self, job: JobInfo) -> bool:
        """Whether `job` may be adopted as one of this pool's hosts."""
        reason = _host_rejection(
            job,
            policy=self._adopt_hosts,
            principal_id=self._principal_id() if self._adopt_hosts == ADOPT_OWN else None,
            namespace=self._namespace,
            image=self.image,
            flavor=self.flavor,
        )
        if reason is not None:
            logger.debug(f"Not adopting job {job.id} as a host for pool '{self.name}': {reason}.")
            return False
        return True

    def _adoptable_pending(self, job: JobInfo) -> bool:
        """Like `_adoptable`, minus the checks a not-yet-running job cannot pass."""
        reason = _host_rejection(
            job,
            policy=self._adopt_hosts,
            principal_id=self._principal_id() if self._adopt_hosts == ADOPT_OWN else None,
            namespace=self._namespace,
            image=self.image,
            flavor=self.flavor,
            check_url=False,
        )
        if reason is not None:
            logger.debug(f"Not waiting for job {job.id} as a host for pool '{self.name}': {reason}.")
            return False
        return True

    def _reserve_one(self) -> "_SandboxServer | None":
        """Reserve one slot on the first host with free capacity (under lock), else None."""
        with self._lock:
            for host in self._hosts:
                if host.capacity - host.live > 0:
                    host.live += 1
                    return host
        return None

    def _boot_one_host(self, new_hosts: List["_SandboxServer"]) -> "_SandboxServer | None":
        """Provision (or adopt) one host and reserve a slot on it. Returns None to retry.

        Held under `_boot_lock` so that, within a process, only one host is booted at a time:
        a burst of create() calls that all found every host full queue here, and each new host
        frees `sandboxes_per_host` slots for the threads still waiting — so they reuse it instead
        of each booting their own. Before booting, we reuse a slot freed by a concurrent boot and,
        failing that, adopt a host already SCHEDULING for this pool (here or in another process)
        rather than piling on a duplicate.
        """
        with self._boot_lock:
            # A concurrent boot (this process) may have added a host with free slots while we
            # waited for the lock — reuse it instead of booting another.
            host = self._reserve_one()
            if host is not None:
                return host
            # A host already coming up for this pool (ours, or another process'): wait for it
            # to start and adopt it instead of booting a duplicate.
            if self._adopt_pending_host():
                host = self._reserve_one()
                if host is not None:
                    return host
            booted = self._provision_hosts(1)
            new_hosts.extend(booted)
            with self._lock:
                self._hosts.extend(booted)
            return self._reserve_one()

    def _adopt_pending_host(self) -> bool:
        """Wait for and adopt a host already SCHEDULING for this pool, if any.

        Avoids over-provisioning when a host is already on its way up for this pool (started by
        another process, or an earlier create() in this one): rather than booting a duplicate,
        wait for it to reach RUNNING and adopt it via discovery. Returns True if a pending host
        was found (whether or not it eventually came up), False if none is scheduling.
        """
        known = {host.job_id for host in self._hosts}
        pending = next(
            (
                job
                for job in self._api.list_jobs(
                    status="SCHEDULING",
                    labels={MODE_LABEL: MODE_POOL, POOL_LABEL: self.name},
                    namespace=self._namespace,
                )
                # A SCHEDULING job has no exposed URL yet, so only the ownership
                # and spec checks apply here; discovery re-validates it in full
                # once it is RUNNING.
                if job.id not in known and self._adoptable_pending(job)
            ),
            None,
        )
        if pending is None:
            return False
        logger.debug(f"Pool '{self.name}' host {pending.id} is already SCHEDULING; waiting for it instead of booting.")
        deadline = time.time() + self._start_timeout
        while time.time() < deadline:
            stage = self._api.inspect_job(job_id=pending.id, namespace=self._namespace).status.stage
            if stage not in ("SCHEDULING", "RUNNING"):
                return False  # it died before starting up; let the caller boot a fresh one
            if stage == "RUNNING":
                # RUNNING precedes "server ready" (the host still has to fetch + exec sbx-server),
                # so discovery only adopts it once its server answers — keep polling until it does.
                self._discover_hosts()
                if any(host.job_id == pending.id for host in self._hosts):
                    return True
            time.sleep(1.0)
        return False

    def _discover_hosts(self) -> None:
        """Attach to running host jobs carrying this pool's labels.

        Lets `create()` reuse a host warmed by an earlier call or another process
        instead of booting a new one. Hosts are found via job labels; each adopted
        host's free capacity is read from the server, so packing stays accurate.

        Matching is on labels only — `hf-sandbox-mode=pool` plus `hf-sandbox-pool=<name>` in
        the target namespace. Job labels are set by whoever creates the job, so in a
        namespace whose members do not all trust each other, any of them can publish a job
        this will adopt. Adopted hosts are also not owned by this handle even though
        `close()` currently cancels them; see `close()`.
        """
        known = {host.job_id for host in self._hosts}
        matches = [
            job
            for job in self._api.list_jobs(
                status="RUNNING",
                labels={MODE_LABEL: MODE_POOL, POOL_LABEL: self.name},
                namespace=self._namespace,
            )
            if job.id not in known and self._adoptable(job)
        ]

        for job in matches:
            server = None
            try:
                server = _connect_host(self._api, job.id, namespace=self._namespace)
                # Capacity comes from the host's SBX_CAPACITY env; otherwise an adopted host
                # would silently pack at the pool default and mis-pack.
                env = _host_env(self._api, job, namespace=self._namespace)
                server.capacity = int(env.get("SBX_CAPACITY", self.sandboxes_per_host))
                server.live = len(server.request("GET", "/v1/sandboxes").json())
            except (SandboxError, httpx.HTTPError, HfHubHTTPError) as e:
                # Host died (e.g. deleted between list_jobs and inspect_job), is still starting
                # up, or is unreachable: skip it, closing the client if one was opened.
                logger.debug(f"Skipping host {job.id} during discovery: {e}")
                if server is not None:
                    server.close()
                continue
            with self._lock:
                if any(host.job_id == job.id for host in self._hosts):
                    server.close()  # adopted concurrently by another thread
                else:
                    self._hosts.append(server)

    def _live_host_count(self) -> int:
        """How many hosts are running for this pool, not just the ones we track.

        `max_hosts` is documented as a cost ceiling, but it was compared against
        `len(self._hosts)` -- an in-process count. Two processes using the same
        pool each counted only their own hosts, so the number of billed host jobs
        could quietly exceed the cap by a factor of however many processes were
        involved.

        Counting via labels narrows that considerably. It does not close it: two
        processes can still list simultaneously and both decide there is room. A
        hard cap has to be enforced where the jobs are created, not by the
        clients racing to count them.
        """
        tracked = {host.job_id for host in self._hosts}
        try:
            for job in self._api.list_jobs(
                status="RUNNING",
                labels={MODE_LABEL: MODE_POOL, POOL_LABEL: self.name},
                namespace=self._namespace,
            ):
                tracked.add(job.id)
        except Exception as e:
            # Fall back to the in-process count rather than refusing to boot.
            logger.debug(f"Could not count live hosts for pool '{self.name}': {e}")
        return len(tracked)

    def _provision_hosts(self, num_new: int) -> List[_SandboxServer]:
        """Boot `num_new` host jobs in parallel, respecting `max_hosts`."""
        if self.max_hosts is not None:
            with self._lock:
                current = self._live_host_count()
            if current + num_new > self.max_hosts:
                allowed = self.max_hosts - current
                raise SandboxError(
                    f"Pool needs {num_new} more host(s) but max_hosts={self.max_hosts} "
                    f"allows only {max(0, allowed)} more ({current} already running for this pool). "
                    "Raise max_hosts or kill some sandboxes."
                )
        # Boot in parallel, but collect every result before re-raising: if one boot fails,
        # the others may have already started billable host jobs, so cancel those instead of
        # leaking them (executor.map would surface the first error and drop the rest).
        with ThreadPoolExecutor(max_workers=min(num_new, 32)) as executor:
            futures = [executor.submit(self._boot_host) for _ in range(num_new)]
        booted: List[_SandboxServer] = []
        error: Exception | None = None
        for future in futures:
            try:
                booted.append(future.result())
            except Exception as e:
                error = e
        if error is not None:
            for server in booted:
                try:
                    server.cancel_job()
                except Exception:
                    pass
                finally:
                    server.close()
            raise error
        return booted

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
            sandbox_token=sandbox_token,
        )
        # Host mode: per-sandbox idle eviction + empty-host shutdown (vs the dedicated
        # whole-job watchdog).
        job_env["SBX_HOST_MODE"] = "1"
        # Server-authoritative packing cap: the host refuses creates past this (the client
        # then packs onto / boots another host). Avoids cross-process over-commit. Read
        # back (with SBX_IDLE_TIMEOUT) by `connect()` to rebuild the pool — labels stay for
        # filtering only.
        job_env["SBX_CAPACITY"] = str(self.sandboxes_per_host)
        # Persist the optional cost ceiling on the host too, so a later connect() (e.g. the CLI
        # `pool create --max-hosts N` then `create --pool <id>` flow) rebuilds the cap instead of
        # defaulting to unlimited and provisioning past it.
        if self.max_hosts is not None:
            job_env["SBX_MAX_HOSTS"] = str(self.max_hosts)
        labels = {SANDBOX_LABEL: "1", MODE_LABEL: MODE_POOL, POOL_LABEL: self.name, NONCE_LABEL: nonce}
        job = self._api.run_job(
            image=self.image,
            command=command,
            env=job_env,
            secrets=job_secrets,
            flavor=self.flavor,
            timeout=SANDBOX_MAX_LIFETIME,
            labels=labels,
            volumes=job_volumes or None,
            expose=[SANDBOX_SERVER_PORT],
            namespace=self._namespace,
        )
        server: "_SandboxServer | None" = None
        try:
            # from_job is inside the try: if it raises (e.g. the server port isn't exposed),
            # the already-started host job must still be cancelled so it doesn't linger billing.
            server = _SandboxServer.from_job(
                job=job,
                nonce=nonce,
                sandbox_token=sandbox_token,
                api=self._api,
                max_connections=min(self.sandboxes_per_host + 8, 256),
                capacity=self.sandboxes_per_host,
            )
            server.wait_ready(self._start_timeout)
        except Exception:
            try:
                self._api.cancel_job(job_id=job.id, namespace=job.owner.name)
            except Exception as e:
                logger.warning(f"Failed to cancel sandbox host {job.id} after startup failure: {e}")
            if server is not None:
                server.close()
            raise
        # We started this job, so this handle is the one that should end it.
        server.owned = self._owns_hosts
        return server

    def _create_one(self, host: "_SandboxServer", env: dict[str, Any], idle_secs: int | None) -> "Sandbox | None":
        """Create one sandbox on a reserved `host`. Returns the sandbox, or None to retry.

        None means either the host filled up between our reservation and the create
        (server-authoritative capacity — another client packed it, so we mark it full and
        place the sandbox elsewhere) or a host rebuilt from the cache is gone/unreachable
        (dropped and re-placed via discovery / a fresh boot).
        """
        body: dict[str, Any] = {"count": 1}
        if idle_secs is not None:
            body["idle_timeout_secs"] = idle_secs
        if env:
            body["env"] = env
        try:
            data = host.request("POST", "/v1/sandboxes", json=body).json()
        except (SandboxError, httpx.HTTPError) as e:
            if host.verified:
                raise  # a host we booted/discovered this session failing is a real error
            logger.debug(f"Dropping unreachable cached host {host.job_id}: {e}")
            self._drop_host(host)
            return None
        host.verified = True  # it answered, so it's a real live host now
        sandboxes = data.get("sandboxes") or []
        if int(data.get("rejected", 0)) or not sandboxes:
            with self._lock:
                host.live = host.capacity  # the host is full; stop reserving it
            return None
        item = sandboxes[0]
        sandbox = Sandbox(
            id=f"{host.job_id}{SHARED_ID_SEP}{item['id']}",
            server=host,
            local_id=item["id"],
            owns_sandbox=True,
            owns_server=False,
            # Absent on hosts running a server that predates per-sandbox tokens;
            # those keep using the host credential.
            sandbox_token=item.get("token"),
        )
        sandbox._on_kill = self._on_sandbox_killed
        return sandbox

    def _on_sandbox_killed(self, sandbox: Sandbox) -> None:
        """Free the packing slot of a shared sandbox that was killed."""
        with self._lock:
            sandbox._server.live = max(0, sandbox._server.live - 1)

    def _drop_host(self, host: _SandboxServer) -> None:
        """Forget a host found dead this session (and mark it for cache pruning)."""
        with self._lock:
            self._hosts = [h for h in self._hosts if h is not host]
            self._dead_host_ids.add(host.job_id)
        host.close()

    # ------------------------------------------------------------------ cache (best effort)

    def _seed_hosts_from_cache(self) -> None:
        """Adopt the pool's cached hosts without any HTTP (rebuilt from cached URL + nonce).

        The first thing that happens to a seeded host is an authenticated `POST`, so what the
        file says about *where* it lives is not taken on trust: the URL must be one only this
        job can have (`_cached_host_rejection`), and an entry older than `HOST_TRUST_TTL` is
        re-checked against the Jobs API first (`_confirm_cached_host`). A fresh entry — the
        `pool create` → `create --pool` case the cache exists for — is used as is.

        Liveness stays best-effort: a seeded host is confirmed by its first successful request
        and dropped on the first failure (see `_create_one`). Hosts the cache believes are
        full are skipped — label discovery re-checks them with fresh counts if the seeded ones
        don't satisfy the request, so a stale-full entry never blocks a create.
        """
        cache = read_pool_cache(self.name, self._cache_context)
        if cache is None:
            return
        hf_token = _effective_token(self._api)
        now = time.time()
        for ch in cache.hosts:
            with self._lock:
                known = {host.job_id for host in self._hosts}
            if ch.job_id in known or ch.live >= ch.capacity:
                continue
            reason = _cached_host_rejection(ch)
            if reason is not None:
                logger.debug(f"Dropping cached host {ch.job_id} of pool '{self.name}': it {reason}.")
                with self._lock:
                    self._dead_host_ids.add(ch.job_id)  # an entry we can never use is litter
                continue
            if now - ch.updated_at > HOST_TRUST_TTL and not self._confirm_cached_host(ch):
                continue
            server = _SandboxServer(
                job_id=ch.job_id,
                owner=ch.owner,
                image=self.image,
                base_url=ch.base_url,
                nonce=ch.nonce,
                sandbox_token=_derive_sandbox_token(hf_token, ch.nonce),
                api=self._api,
                max_connections=min(self.sandboxes_per_host + 8, 256),
                capacity=ch.capacity,
            )
            server.live = ch.live
            server.verified = False
            with self._lock:
                if any(host.job_id == ch.job_id for host in self._hosts):
                    server.close()  # seeded concurrently by another thread
                else:
                    self._hosts.append(server)

    def _confirm_cached_host(self, ch: CachedHost) -> bool:
        """Confirm an aged-out cache entry against the Jobs API before crediting it.

        Freshness is what the no-HTTP path rests on: the entry was written by this same
        principal minutes ago, so the job it names is almost certainly still the host it
        described. Past `HOST_TRUST_TTL` that argument weakens, and the entry becomes a hint
        that has to be checked — the job must still be running, still be labelled for *this*
        pool with the nonce the entry derives its token from, still pass the pool's adoption
        policy, and still expose exactly the cached URL. One round-trip, and only for entries
        the fast path would no longer be honest about.

        A definitive "no" prunes the entry; a transient failure only skips it this time.
        """
        try:
            job = self._api.inspect_job(job_id=ch.job_id, namespace=ch.owner)
            labels = job.labels or {}
            if job.status.stage != "RUNNING":
                reason = f"its job is not running (status: {job.status.stage})"
            elif labels.get(MODE_LABEL) != MODE_POOL or labels.get(POOL_LABEL) != self.name:
                reason = f"its job is not labelled as a host of pool '{self.name}'"
            elif labels.get(NONCE_LABEL) != ch.nonce:
                reason = "its job carries a different nonce"
            elif not self._adoptable(job):
                reason = "its job is not one this pool may adopt"
            elif (exposed := _find_server_url(job)) != ch.base_url:
                reason = f"its job exposes {exposed!r}, not the cached URL"
            else:
                return True
        except Exception as e:
            logger.debug(f"Could not confirm cached host {ch.job_id} of pool '{self.name}': {e}")
            return False
        logger.debug(f"Dropping stale cached host {ch.job_id} of pool '{self.name}': {reason}.")
        with self._lock:
            self._dead_host_ids.add(ch.job_id)
        return False

    def _save_cache(self, hosts: "List[_SandboxServer] | None" = None) -> None:
        """Persist the pool config + hosts (with their live counts) for next time.

        `hosts` overrides the tracked list, for the one case where they differ:
        `close()` persisting the hosts it could *not* cancel, since that record is
        the caller's only handle on a job that is still billing.
        """
        with self._lock:
            source = self._hosts if hosts is None else hosts
            entries = [
                CachedHost(
                    job_id=host.job_id,
                    owner=host.owner,
                    base_url=host.base_url,
                    nonce=host.nonce,
                    capacity=host.capacity,
                    live=host.live,
                    updated_at=time.time(),
                )
                for host in source
            ]
            dead = set(self._dead_host_ids)
        save_pool_cache(
            self.name,
            context=self._cache_context,
            image=self.image,
            flavor=self.flavor,
            sandboxes_per_host=self.sandboxes_per_host,
            max_hosts=self.max_hosts,
            idle_timeout=_duration_to_secs(self._idle_timeout) if self._idle_timeout is not None else None,
            hosts=entries,
            dead_host_ids=dead,
        )


def _normalize_image(image: str | None) -> str | None:
    """Compare Docker images leniently: registries and tags get normalized
    server-side, and a mis-parse here must not stop a pool from working."""
    if image is None:
        return None
    image = image.strip().lower()
    for prefix in ("docker.io/", "index.docker.io/", "library/"):
        if image.startswith(prefix):
            image = image[len(prefix) :]
    return image


def _server_host_rejection(url: str, *, job_id: str) -> str | None:
    """Why `url` is not a URL we should send a sandbox job's credentials to.

    The hostname must be derived from the job id we already hold, so nothing but the
    client's own knowledge of which job it is talking to can decide where the credentials
    go. The domain is not hard-coded, so staging endpoints keep working.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return f"uses scheme {parsed.scheme!r}, not https"
    # `urlparse` lowercases the hostname, so compare case-insensitively.
    if not (parsed.hostname or "").startswith(f"{job_id}--{SANDBOX_SERVER_PORT}.".lower()):
        return f"is not job {job_id}'s port-{SANDBOX_SERVER_PORT} URL"
    return None


def _server_url_rejection(job: JobInfo) -> str | None:
    """Why this Job's exposed URL is not one we should send credentials to."""
    urls = list(job.status.expose_urls or []) if job.status is not None else []
    if len(urls) != 1:
        return f"exposes {len(urls)} URL(s), expected exactly one for port {SANDBOX_SERVER_PORT}"
    reason = _server_host_rejection(urls[0], job_id=job.id)
    return f"exposes {urls[0]!r}, which {reason}" if reason is not None else None


def _cached_host_rejection(host: CachedHost) -> str | None:
    """Why a cached host entry must not be rebuilt into a credentialed transport.

    Same reasoning as `_server_url_rejection`, for a URL that comes off the local disk
    instead of the Jobs API. Without this, the cache file gets to choose where the HF bearer
    and the derived host token are sent -- `http://127.0.0.1:1234` included -- and it gets to
    choose it *before* any API call could have contradicted it. Deriving the expected
    hostname from the entry's own `job_id` leaves it able to name only a job this client can
    independently identify, which is the same bar a discovered host has to clear.
    """
    reason = _server_host_rejection(host.base_url, job_id=host.job_id)
    return f"names {host.base_url!r}, which {reason}" if reason is not None else None


def _cache_context(api: HfApi, namespace: str | None) -> CacheContext:
    """The local pool cache's security context for this endpoint/credential/namespace."""
    return CacheContext.for_credential(endpoint=api.endpoint, token=_effective_token(api), namespace=namespace)


def _host_rejection(
    job: JobInfo,
    *,
    policy: str,
    principal_id: str | None,
    namespace: str | None,
    image: str | None,
    flavor: str | None,
    check_url: bool = True,
) -> str | None:
    """Why this Job must not be adopted as one of our pool hosts, or `None`.

    Hosts are found by filtering Jobs on labels, and labels are set by whoever
    creates the Job -- so a label match says "this Job claims to be one of our
    hosts", not "this Job is one of our hosts". Anyone who can create a Job in
    the namespace can make that claim, copy the public nonce from a real host's
    labels, and receive the token derived for that host.

    Everything checked here is asserted by the backend rather than by the Job's
    creator: the initiator, the owner, the image, the flavor, the command, and
    the exposed URL. `initiator` is the load-bearing one -- there is no
    client-side way to set it -- and it is what makes `policy="own"` meaningful.
    """
    if policy == ADOPT_NEVER:
        return "host adoption is disabled (adopt_hosts='never')"

    if policy == ADOPT_OWN:
        initiator = job.initiator
        if initiator is None or initiator.id is None:
            return "the Jobs API did not say who started it, so it cannot be confirmed as ours"
        if principal_id is None:
            return "could not resolve the current principal to compare against its initiator"
        if initiator.id != principal_id:
            return f"started by a different principal ({initiator.name or initiator.id})"

    if namespace is not None and job.owner is not None and job.owner.name != namespace:
        return f"runs under namespace {job.owner.name!r}, not {namespace!r}"

    job_image = job.docker_image or job.space_id
    if image is not None and _normalize_image(job_image) != _normalize_image(image):
        return f"runs image {job_image!r}, not {image!r}"

    if flavor is not None and job.flavor is not None and str(job.flavor) != flavor:
        return f"runs flavor {str(job.flavor)!r}, not {flavor!r}"

    # A sandbox host runs the bootstrap script and nothing else. A Job running
    # anything else is not a host, whatever its labels say.
    if job.command is not None and list(job.command) != _bootstrap_command():
        return "does not run the sandbox bootstrap command"

    # A SCHEDULING job has no exposed URL yet; the caller re-checks once it runs.
    return _server_url_rejection(job) if check_url else None


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
    sandbox_token: str,
) -> tuple[list[str], dict[str, Any], dict[str, Any], List[Volume]]:
    """Build the (command, env, secrets, volumes) to launch a job running sbx-server.

    Shared by dedicated sandboxes and shared hosts: both fetch and exec the same unified
    `sbx-server` binary at startup (via `/bin/sh`), downloading it with wget/curl, or
    reading it off the always-mounted server bucket when the image ships neither. Either
    way the download is pinned to `SANDBOX_SERVER_SHA256` and verified before it runs.
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

    # The digest-named object, not the mutable `sbx-server` alias: a fetch that names the
    # content it wants cannot be answered with different content, and the bootstrap re-checks
    # the digest anyway so a rewritten object is caught rather than executed.
    server_object = f"sbx-server-{SANDBOX_SERVER_SHA256}"
    job_env["SBX_SERVER_URL"] = f"{api.endpoint}/buckets/{constants.SANDBOX_SERVER_BUCKET}/resolve/{server_object}"
    job_env["SBX_SERVER_SHA256"] = SANDBOX_SERVER_SHA256
    # Always mount the server bucket as a transparent fallback for images without wget/curl.
    # It's only read (paying the ~2-3s FUSE cost) when the bootstrap script can't download.
    job_env["SBX_SERVER_MOUNT"] = _SERVER_MOUNT_PATH
    job_volumes.append(
        Volume(type="bucket", source=constants.SANDBOX_SERVER_BUCKET, mount_path=_SERVER_MOUNT_PATH, read_only=True)
    )
    return _bootstrap_command(), job_env, job_secrets, job_volumes


def _bootstrap_command() -> list[str]:
    """The Job command every sandbox host and dedicated sandbox runs."""
    return ["/bin/sh", "-c", _BOOTSTRAP_DOWNLOAD]


def _host_env(api: HfApi, job: JobInfo, *, namespace: str | None) -> dict[str, Any]:
    """Return a host job's env vars (where pool config lives)"""
    env = job.environment if isinstance(job.environment, dict) else {}
    if "SBX_CAPACITY" not in env:
        env = api.inspect_job(job_id=job.id, namespace=namespace).environment or {}
        env = env if isinstance(env, dict) else {}
    return env


def _find_pool_host_job(api: HfApi, pool_id: str, *, namespace: str | None = None, policy: str = ADOPT_OWN) -> JobInfo:
    """Return a running host job belonging to `pool_id`, found via the pool label.

    This one feeds `SandboxPool.connect`'s cold path, which rebuilds the pool's
    whole configuration (image, flavor, density, caps) from the job it returns --
    so an unvalidated match would let a Job in the namespace choose the image
    that the pool's *next* hosts boot. Image and flavor are unknown here (they
    are what we are about to learn), so this checks ownership and shape.
    """
    principal = None
    if policy == ADOPT_OWN:
        try:
            principal = api.whoami(cache=True).get("id")
        except Exception as e:
            logger.warning(f"Could not resolve the current user to validate pool hosts: {e}")
    rejected = []
    for job in api.list_jobs(
        status="RUNNING", labels={MODE_LABEL: MODE_POOL, POOL_LABEL: pool_id}, namespace=namespace
    ):
        reason = _host_rejection(
            job, policy=policy, principal_id=principal, namespace=namespace, image=None, flavor=None
        )
        if reason is None:
            return job
        rejected.append(f"{job.id} ({reason})")
    if rejected:
        raise SandboxError(
            f"Found host job(s) labelled for pool '{pool_id}' but none usable: {'; '.join(rejected)}. "
            "Pass adopt_hosts='namespace' to SandboxPool if you intend to share hosts with other "
            "members of this namespace."
        )
    raise SandboxError(
        f"No running host found for pool '{pool_id}'. The pool has stopped "
        "(all its hosts were killed or idle-timed-out); create a new one."
    )


def _connect_host(api: HfApi, host_job_id: str, *, namespace: str | None = None) -> _SandboxServer:
    """Reattach to a running host job and return its server transport."""
    job = api.inspect_job(job_id=host_job_id, namespace=namespace)
    labels = job.labels or {}
    nonce = labels.get(NONCE_LABEL)
    if nonce is None or labels.get(MODE_LABEL) != MODE_POOL:
        raise SandboxError(f"Job {host_job_id} is not a sandbox host.")
    if job.status.stage != "RUNNING":
        raise SandboxError(f"Sandbox host {host_job_id} is not running (status: {job.status.stage}).")
    # The caller named this job explicitly, so who started it is their call --
    # but the credentials below must still only reach this job's own URL, over
    # HTTPS, on a job that actually runs the sandbox bootstrap.
    reason = _host_rejection(
        job, policy=ADOPT_NAMESPACE, principal_id=None, namespace=namespace, image=None, flavor=None
    )
    if reason is not None:
        raise SandboxError(f"Job {host_job_id} does not look like a usable sandbox host: {reason}.")
    return _SandboxServer.from_job(
        job=job,
        nonce=nonce,
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
