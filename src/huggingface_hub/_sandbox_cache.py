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
"""Best-effort local cache for [`SandboxPool`] hosts (host/pool mode).

Why this exists
---------------
A pool has **no authoritative local state** — it *is* its running host Jobs, found via
job labels (see `_sandbox.py`). That makes pools discoverable from any machine, but it
also means a cold `hf sandbox create --pool <id>` must, every single time:

1. `list_jobs()` to find the pool's hosts (scans *all* the namespace's jobs),
2. `inspect_job()` each host to rebuild its server URL + auth nonce,
3. `GET /v1/sandboxes` on each to learn how full it is,

before it can finally `POST /v1/sandboxes` to spawn the sandbox. Even with a warm host
that is several round-trips of pure overhead per CLI invocation.

This module persists what a process learned about a pool to
`$HF_HOME/sandbox/pools/<pool-id>.json` so the *next* process can skip straight to the
`POST`: it caches the pool config (enough to boot a fresh host) and, per host, the URL +
nonce + last-seen free slots needed to talk to it directly.

It is **purely an optimization and never a source of truth**:

- The in-job server is authoritative on capacity (it replies `{"rejected": N}` when full),
  so a stale `live` count only ever costs a wasted request, never correctness.
- Every read tolerates a missing / corrupt / foreign-machine file by returning `None`;
  the caller then falls back to the label-discovery path.
- Writes are merged under a [`WeakFileLock`] and committed atomically (temp file +
  `os.replace`), so concurrent `create` processes don't clobber each other and lock-free
  readers never see a half-written file.

Deleting the file at any time is safe.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

from . import constants
from .utils import WeakFileLock, logging


logger = logging.get_logger(__name__)

# Bump if the on-disk layout changes incompatibly; older/newer files are ignored on read.
_CACHE_VERSION = 3

# A write should never block a sandbox creation for long: the cache is best-effort, so we
# rather skip persisting than wait on a stuck lock.
_LOCK_TIMEOUT = 5.0


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
    hosts: List[CachedHost] = field(default_factory=list)
    version: int = _CACHE_VERSION
    updated_at: float = 0.0


def _pools_dir() -> Path:
    return Path(constants.HF_HOME) / "sandbox" / "pools"


def pool_cache_path(pool_id: str) -> Path:
    """Path of the cache file for `pool_id` (no I/O)."""
    return _pools_dir() / f"{pool_id}.json"


def read_pool_cache(pool_id: str) -> PoolCache | None:
    """Return the cached view of `pool_id`, or `None` if missing/corrupt/incompatible.

    Never raises: the cache is best-effort, so any problem reading it is treated as a
    cache miss and the caller falls back to label discovery.
    """
    path = pool_cache_path(pool_id)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != _CACHE_VERSION:
            return None
        hosts = [CachedHost(**h) for h in data.pop("hosts", [])]
        return PoolCache(**data, hosts=hosts)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.debug(f"Ignoring unreadable sandbox pool cache {path}: {e}")
        return None


def save_pool_cache(
    pool_id: str,
    *,
    image: str,
    flavor: str,
    sandboxes_per_host: int,
    max_hosts: int | None,
    idle_timeout: int | None,
    namespace: str | None,
    hosts: List[CachedHost],
    dead_host_ids: set[str] | None = None,
) -> None:
    """Merge `hosts` into the cache for `pool_id` (best-effort, never raises).

    Concurrency-safe: under a file lock, the on-disk hosts are read, then `hosts` are
    upserted by `job_id` and `dead_host_ids` removed, so a process only adds/updates what
    it learned and never drops hosts another process discovered. The result is written
    atomically. The pool config is refreshed from the arguments.
    """
    path = pool_cache_path(pool_id)
    dead = dead_host_ids or set()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with WeakFileLock(str(path) + ".lock", timeout=_LOCK_TIMEOUT):
            existing = read_pool_cache(pool_id)
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
                namespace=namespace,
                hosts=list(merged.values()),
                updated_at=time.time(),
            )
            _atomic_write(path, cache)
    except Exception as e:
        logger.debug(f"Could not write sandbox pool cache {path}: {e}")


def delete_pool_cache(pool_id: str) -> None:
    """Remove the cache file for `pool_id` (best-effort, never raises)."""
    try:
        pool_cache_path(pool_id).unlink(missing_ok=True)
    except Exception as e:
        logger.debug(f"Could not delete sandbox pool cache for {pool_id}: {e}")


def _atomic_write(path: Path, cache: PoolCache) -> None:
    """Write the cache via a temp file + `os.replace` so readers never see a partial file."""
    tmp = path.parent / f"{path.name}.{os.getpid()}.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(asdict(cache), f, indent=2)
    os.replace(tmp, path)
