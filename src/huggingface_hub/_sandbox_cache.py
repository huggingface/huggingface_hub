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
"""Best-effort local cache for SandboxPool hosts (host/pool mode)."""

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List

from . import constants
from .utils import WeakFileLock, logging


logger = logging.get_logger(__name__)

# Bump if the on-disk layout changes incompatibly; older/newer files are ignored on read.
_CACHE_VERSION = 2

# A write should never block a sandbox creation for long: the cache is best-effort, so we
# rather skip persisting than wait on a stuck lock.
_LOCK_TIMEOUT = 5.0

# How long a cached host may be turned into a credentialed transport on the strength of the
# file alone. Within it, the `pool create` -> `create --pool` sequence stays at zero extra
# round-trips (the point of the cache); past it, the caller re-checks the host against the
# Jobs API before sending it anything.
HOST_TRUST_TTL = 15 * 60

# The cache holds reusable host URLs and the public nonces the host tokens derive from, so
# both the directories and the files are private to the user.
_DIR_MODE = 0o700

# Domain separation, so a digest from this cache is not also a digest of the same material
# computed for some other purpose (and so a version bump renames every directory).
_DIGEST_PREFIX = f"hf-sandbox-pool-cache/v{_CACHE_VERSION}"

# Plausibility bounds for the schema check below. Generous on purpose: this is here to turn a
# wrong-typed or absurd value into a cache miss, not to re-specify the pool's own limits.
_MAX_COUNT = 1_000_000
_MAX_SECONDS = 10 * 365 * 24 * 3600
_MAX_TIMESTAMP = 4_000_000_000  # epoch seconds, some time in 2096
_MAX_STR = 2048
# Job ids are backend-assigned; the id also feeds a URL and an `inspect_job` call, so keep it
# to characters that cannot change the meaning of either.
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
# Nonces are minted by `token_hex(16)`; the bounds are loose so that changing that length
# would not quietly turn every cache entry into a miss.
_NONCE_RE = re.compile(r"^[0-9a-fA-F]{16,128}$")


def _digest(*parts: str) -> str:
    """Short, stable digest of `parts` (which are joined unambiguously)."""
    material = "\x00".join((_DIGEST_PREFIX, *parts))
    return hashlib.sha256(material.encode()).hexdigest()


@dataclass(frozen=True)
class CacheContext:
    """The security context a cache entry belongs to.

    A cached entry is not just data: the fast path rebuilds a host transport from it and
    sends it the HF bearer plus the derived host token. So it may only ever be read back by
    the same *endpoint*, *credential* and *namespace* that wrote it -- anything else is a
    plain cache miss, which is what the docs promise and what keeps a `connect(namespace=...)`
    from being served hosts that belong to another namespace.

    `principal` is an opaque, stable id for the credential; the token itself is never stored
    nor used as a key (see `for_credential`).
    """

    endpoint: str
    principal: str
    namespace: str | None

    @classmethod
    def for_credential(cls, *, endpoint: str, token: str, namespace: str | None) -> "CacheContext":
        """Build the context of a given HF token, identified by its fingerprint.

        The fingerprint is a local computation, which is what keeps a cache hit at zero
        round-trips -- resolving the credential to a user id would cost a `whoami` and defeat
        the purpose of the cache. It is derived from the token but does not contain it: only
        the digest is ever written to disk.

        A rotated or swapped credential therefore reads as a different principal and gets a
        cache miss (the cold path, which always works), rather than someone else's hosts.
        """
        return cls(endpoint=endpoint, principal=_digest("principal", token)[:32], namespace=namespace)

    @property
    def key(self) -> str:
        """Digest of the whole context; the name of the cache subdirectory it owns."""
        return _digest("context", self.endpoint, self.principal, self.namespace or "")[:16]


@dataclass
class CachedHost:
    """A single host Job of a pool, as last seen by some process.

    `base_url` + `nonce` are everything needed to rebuild the in-job server transport
    (`_SandboxServer`) without an `inspect_job` round-trip: the per-sandbox auth token is
    re-derived from the user's HF token and `nonce` (see `_derive_sandbox_token`).
    """

    job_id: str
    owner: str  # namespace the host job runs under (for cancel/inspect)
    base_url: str  # exposed sbx-server URL (does not change while the job lives)
    nonce: str  # public nonce from the job label; derives the sandbox auth token
    capacity: int  # SBX_CAPACITY: max sandboxes the host packs
    live: int  # sandboxes last observed on the host (best-effort, may be stale)
    updated_at: float = 0.0


@dataclass
class PoolCache:
    """Cached view of one pool: its config (to boot new hosts) + its known hosts."""

    pool_id: str
    image: str
    flavor: str
    sandboxes_per_host: int
    max_hosts: int | None
    idle_timeout: int | None
    namespace: str | None
    # Context the entry was written in, so a file that is moved (or copied from elsewhere)
    # into a directory it was not written for is still rejected. See `CacheContext`.
    endpoint: str = ""
    context: str = ""
    hosts: List[CachedHost] = field(default_factory=list)
    version: int = _CACHE_VERSION
    updated_at: float = 0.0


def _pools_dir() -> Path:
    return Path(constants.HF_HOME) / "sandbox" / "pools"


def _check_pool_id(pool_id: str) -> str:
    """Return `pool_id` if it is safe to use as a file name, else raise."""
    if any(c in pool_id for c in ("/", "\\", "\x00")) or pool_id in (".", ".."):
        raise ValueError(f"Invalid pool id: {pool_id!r}")
    return pool_id


def pool_cache_path(pool_id: str, context: CacheContext) -> Path:
    """Path of the cache file for `pool_id` in `context` (no I/O)."""
    return _pools_dir() / context.key / f"{_check_pool_id(pool_id)}.json"


def read_pool_cache(pool_id: str, context: CacheContext) -> PoolCache | None:
    """Return the cached view of `pool_id` for `context`, or `None`.

    Missing, corrupt, incompatible, mistyped and foreign-context files are all the same thing
    here: a cache miss, logged at debug level. The caller's cold path is always correct, so
    this never raises -- a bad cache costs latency, never an error mid-`create()`.
    """
    try:
        path = pool_cache_path(pool_id, context)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != _CACHE_VERSION:
            return None
        hosts = [CachedHost(**h) for h in data.pop("hosts", [])]
        cache = PoolCache(**data, hosts=hosts)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.debug(f"Ignoring unreadable sandbox pool cache for {pool_id!r}: {e}")
        return None
    reason = _cache_rejection(cache, pool_id=pool_id, context=context)
    if reason is not None:
        logger.debug(f"Ignoring sandbox pool cache for {pool_id!r}: {reason}.")
        return None
    return cache


def save_pool_cache(
    pool_id: str,
    *,
    context: CacheContext,
    image: str,
    flavor: str,
    sandboxes_per_host: int,
    max_hosts: int | None,
    idle_timeout: int | None,
    hosts: List[CachedHost],
    dead_host_ids: set[str] | None = None,
) -> None:
    """Merge `hosts` into the cache for `pool_id` in `context` (best-effort, never raises).

    Concurrency-safe: under a file lock, the on-disk hosts are read, then `hosts` are
    upserted by `job_id` and `dead_host_ids` removed, so a process only adds/updates what
    it learned and never drops hosts another process discovered. The result is written
    atomically. The pool config is refreshed from the arguments.
    """
    dead = dead_host_ids or set()
    try:
        path = pool_cache_path(pool_id, context)
        _ensure_private_dir(path.parent)
        with WeakFileLock(str(path) + ".lock", timeout=_LOCK_TIMEOUT):
            existing = read_pool_cache(pool_id, context)
            merged = {h.job_id: h for h in (existing.hosts if existing else [])}
            for host in hosts:
                merged[host.job_id] = host
            for job_id in dead:
                merged.pop(job_id, None)
            cache = PoolCache(
                pool_id=pool_id,
                image=image,
                flavor=flavor,
                sandboxes_per_host=sandboxes_per_host,
                max_hosts=max_hosts,
                idle_timeout=idle_timeout,
                namespace=context.namespace,
                endpoint=context.endpoint,
                context=context.key,
                hosts=list(merged.values()),
                updated_at=time.time(),
            )
            _atomic_write(path, cache)
    except Exception as e:
        logger.debug(f"Could not write sandbox pool cache for {pool_id!r}: {e}")


def delete_pool_cache(pool_id: str, context: CacheContext | None = None) -> None:
    """Remove the cache file(s) for `pool_id` (best-effort, never raises).

    Without a `context`, the pool's entry is removed from *every* context: deleting a pool is
    the one operation for which over-deleting is the safe direction (the cache is disposable),
    and the caller may not know which credential wrote the entry.
    """
    try:
        if context is not None:
            pool_cache_path(pool_id, context).unlink(missing_ok=True)
            return
        for path in _pools_dir().glob(f"*/{_check_pool_id(pool_id)}.json"):
            path.unlink(missing_ok=True)
    except Exception as e:
        logger.debug(f"Could not delete sandbox pool cache for {pool_id}: {e}")


def _ensure_private_dir(path: Path) -> None:
    """Create the cache directory tree, readable by this user only.

    `mkdir`'s mode is masked by the umask and the tree may predate this code, so the modes
    are also set explicitly, from the leaf up to (and including) the sandbox directory.
    """
    path.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    root = Path(constants.HF_HOME) / "sandbox"
    for directory in (path, *path.parents):
        if directory != root and root not in directory.parents:
            break  # $HF_HOME itself (and anything above it) is not ours to re-mode
        os.chmod(directory, _DIR_MODE)


def _atomic_write(path: Path, cache: PoolCache) -> None:
    """Write the cache via a temp file + `os.replace` so readers never see a partial file.

    `mkstemp` gives the temp file an unpredictable name and mode `0600` regardless of the
    umask (which `os.replace` then carries over to the final file), so no other user can read
    -- or pre-create and win a race on -- the host URLs and nonces on their way to disk.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(asdict(cache), f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _is_count(value: Any, *, minimum: int = 0, maximum: int = _MAX_COUNT) -> bool:
    """Whether `value` is a plausible count. `bool` is an `int`, but not a count."""
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _is_timestamp(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= _MAX_TIMESTAMP


def _is_text(value: Any, *, allow_empty: bool = False) -> bool:
    return isinstance(value, str) and (allow_empty or len(value) > 0) and len(value) <= _MAX_STR


def _cache_rejection(cache: PoolCache, *, pool_id: str, context: CacheContext) -> str | None:
    """Why this cache file must not be used, or `None` if it may be.

    Two jobs. First, the entry has to belong to the caller's security context -- same
    endpoint, same credential, same namespace -- which is what makes a cross-namespace or
    cross-principal hit a miss instead of a confidentiality bug.

    Second, a schema check. `dataclass` does not enforce types, so a well-shaped file with
    `"capacity": "50"` parses happily and only blows up much later, on `host.capacity -
    host.live`, as an uncaught `TypeError` in the middle of `create()`. Validating here turns
    that into the documented cache miss.
    """
    if cache.context != context.key:
        return "it was written for a different endpoint, credential or namespace"
    if cache.endpoint != context.endpoint:
        return f"it names endpoint {cache.endpoint!r}, not {context.endpoint!r}"
    if cache.namespace != context.namespace:
        return f"it names namespace {cache.namespace!r}, not {context.namespace!r}"
    if cache.pool_id != pool_id:
        return f"it names pool {cache.pool_id!r}, not {pool_id!r}"

    if not _is_text(cache.image) or not _is_text(cache.flavor):
        return "its image/flavor are not strings"
    if not _is_count(cache.sandboxes_per_host, minimum=1):
        return f"its sandboxes_per_host is not a positive count: {cache.sandboxes_per_host!r}"
    if cache.max_hosts is not None and not _is_count(cache.max_hosts, minimum=1):
        return f"its max_hosts is not a positive count: {cache.max_hosts!r}"
    if cache.idle_timeout is not None and not _is_count(cache.idle_timeout, maximum=_MAX_SECONDS):
        return f"its idle_timeout is not a duration in seconds: {cache.idle_timeout!r}"
    if not _is_timestamp(cache.updated_at):
        return f"its updated_at is not a timestamp: {cache.updated_at!r}"

    if not isinstance(cache.hosts, list):
        return "its hosts are not a list"
    for host in cache.hosts:
        if not _JOB_ID_RE.match(host.job_id if isinstance(host.job_id, str) else ""):
            return f"host {host.job_id!r} is not a plausible job id"
        if not _JOB_ID_RE.match(host.owner if isinstance(host.owner, str) else ""):
            return f"host {host.job_id} names a non-namespace owner: {host.owner!r}"
        if not _is_text(host.base_url):
            return f"host {host.job_id} has no URL"
        if not isinstance(host.nonce, str) or not _NONCE_RE.match(host.nonce):
            return f"host {host.job_id} has no usable nonce"
        if not _is_count(host.capacity) or not _is_count(host.live):
            return f"host {host.job_id} has a non-numeric capacity/live count"
        if not _is_timestamp(host.updated_at):
            return f"host {host.job_id} has no usable timestamp"
    return None
