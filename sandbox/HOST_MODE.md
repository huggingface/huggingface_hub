# Host mode — many lightweight sandboxes inside one Job (`SandboxPool`)

> Goal: cheap, fast fan-out to hundreds/thousands of CPU sandboxes (RL rollouts,
> batch tool execution) without paying one VM cold start per sandbox.
> **Result: 1000 sandboxes created + exec'd + killed in 15.8s, $0.0009 total.**

## The problem with one-Job-per-sandbox

`Sandbox.create` maps **one HF Job to one sandbox**. A Job is a real 2-vCPU VM, so
every sandbox pays a ~6s VM cold start and holds a whole machine even though an RL
environment typically needs a few MB of RAM and one core for a few seconds. At
100–1000 parallel environments this is both slow (a 1000-VM scheduling burst) and
wasteful (1000 VMs billed).

## The idea

Run **one Job as a host** and multiplex many sandboxes inside it. A sandbox is not
a VM or a nested container — it is the classic Unix multi-user primitive:

- a **dedicated uid** (≥ 20000),
- a **private `0700` home** owned by that uid,
- commands `exec`'d with that uid/gid, a **scrubbed environment** (`env_clear`, so
  the host Job's secrets never leak in), `NO_NEW_PRIVS`, per-uid/process **rlimits**,
  and a per-sandbox **Landlock LSM ruleset**.

Creating a sandbox is `mkdir + chown + build ruleset` ≈ **1ms server-side**. There is
no second cold start: the only client-visible latency is the Jobs proxy round-trip
(~100–150ms), and N sandboxes are created in **one batched request per host**.

This ships as [`SandboxPool`]: it provisions host Jobs lazily, packs
`sandboxes_per_host` sandboxes per host, scales out as needed, and tears everything
down on `close()`. Every sandbox it hands out is a normal `Sandbox` (`run`, `spawn`,
`files`, `connect`, `kill`).

`pool.create()` makes one sandbox at a time: it reuses a host with free capacity before
booting a new one, so you grow on demand as work arrives. Pre-provision hosts with
`SandboxPool(warm_up=N)` (or `pool.warm(N)`) to skip the cold start on the first calls.
Warm hosts are discovered via the `hf-sandbox-host` + `hf-sandbox-pool` job labels so reuse
works **across processes** — a fresh pool, or `hf sandbox create --pool <id>`, attaches to a
host an earlier run left running.

### Pools have no local state

A pool is **not** a local config file — it's a set of running host VMs sharing a
`hf-sandbox-pool=<id>` job label. This keeps pools consistent with the rest of the
sandbox API (everything is discoverable from labels, reattachable from any machine):

```bash
hf sandbox pool create --image python:3.12 --flavor cpu-basic   # warms 1 host (bills) -> pool id
hf sandbox create --pool <pool_id>                              # pack onto a host, or boot a duplicate
hf sandbox create --pool <pool_id> --secrets K=v                # env/secrets are per-sandbox
hf sandbox pool ls
hf sandbox pool delete <pool_id>                                # terminate the pool's hosts
```

- `pool create` warms one host carrying the pool's config (image/flavor/`sandboxes_per_host`/
  idle timeout) in its **job env vars** — labels are kept for filtering only.
- `hf sandbox create --pool` (or `SandboxPool.connect(id)`) finds the pool's hosts by label, and
  when it must boot a duplicate it **reads that config back from a running host job** (via
  `inspect_job` env vars). Env/secrets are *not* pool-level: each sandbox gets its own, passed at
  create time — so no secret is ever stored on a host or kept locally.
- Capacity is **server-authoritative**: a host refuses creates past `sandboxes_per_host`
  (replying `{"rejected": N}`), and the client packs the overflow onto another host or
  boots a duplicate. This makes packing exact even when several processes create at once.
- **Idle eviction is two-level**: each sandbox is evicted after its own `idle_timeout` of
  inactivity (unless it still has a running process); once a host has had no sandboxes for the
  host idle timeout, it shuts itself down. A pool **stops existing once all its hosts are gone**.

### A best-effort cache keeps `create --pool` fast

Having no local state is great for correctness but costs latency: every cold
`hf sandbox create --pool <id>` (a fresh CLI process) must `list_jobs` + `inspect_job` each
host + `GET /v1/sandboxes` before it can `POST` to spawn a sandbox. To avoid paying that on
every call, a process records what it learned to `$HF_HOME/sandbox/pools/<pool-id>.json`: the
pool config plus, per host, its proxy URL, auth nonce and last-seen free slots. The next
process rebuilds the host transport straight from that file (no HTTP) and goes directly to the
`POST`. It is **purely an optimization** — the in-job server stays authoritative, a stale/gone
host is dropped and re-discovered on the first failed request, writes are merged under a file
lock and committed atomically, and a missing/corrupt/foreign-machine file is just a cache miss
that falls back to label discovery. Worst case = today's behavior; warm case = one round-trip.
See [`ARCHITECTURE.md`](ARCHITECTURE.md#the-pool-cache--why-create---pool-is-fast) and
`src/huggingface_hub/_sandbox_cache.py`.

## Isolation: uid (DAC) + Landlock LSM, both unprivileged

On a stock Job the container runs **as root inside a user namespace mapping only uids
0..65535**, with seccomp filtering on and no `CAP_SYS_ADMIN/NET_ADMIN/NET_RAW`. That
rules out nested namespaces, mounts and cgroup delegation (`unshare`, `mount`,
`mkdir /sys/fs/cgroup/...` all fail). But the kernel ships **Landlock ABI 6**, which
any process can use to restrict *itself* and its `execve`'d children — exactly the
per-sandbox boundary we need. The server builds one ruleset per sandbox; the exec
child applies `NO_NEW_PRIVS` → `landlock_restrict_self` → rlimits → setuid/gid before
running the command.

Verified live against a hostile sandbox A attacking victim B:

- ✅ A cannot read PID 1 / any process `environ` → **HF & sandbox tokens never leak**.
- ✅ A cannot `SIGKILL`/`ptrace`/read `/proc/<pid>/mem` of B's processes, nor `setuid`
  into B, nor read B's `0700` home (distinct uids).
- ✅ `/tmp` & `/dev/shm`: write/list/symlink-squat all denied — each sandbox is Landlock-
  confined to its own home (`TMPDIR` points into `$HOME/.tmp`).
- ✅ A cannot `bind` a TCP port → no inter-sandbox localhost service (outbound `connect`
  stays allowed, so internet works).
- ✅ Cross-sandbox abstract unix sockets blocked (`LANDLOCK_SCOPED_ABSTRACT_UNIX_SOCKET`,
  ABI 6 — uid isolation alone does *not* block these).

### Residual gaps (acceptable for the same-user trust model)

- **Resource DoS**: no cgroup delegation, so CPU/total-RAM/disk are unpartitioned.
  `RLIMIT_NPROC`/`RLIMIT_AS` bound per-uid/per-process usage, but an aggressive sandbox
  can still starve others or trip the global OOM killer.
- **PID metadata**: a sandbox can *see* other processes via `/proc` (names, cmdlines)
  but cannot read or signal them (no PID-namespace hiding without `unshare`).
- **SysV/POSIX IPC** namespace is shared (Landlock scoping covers abstract unix sockets).

**Net:** confidentiality and integrity between sandboxes are enforced; only availability
(DoS) and process-list metadata remain shared. That's the right boundary for one user's
own parallel rollouts. For **mutually-hostile untrusted code** or GPU, use
`Sandbox.create` (a separate VM per sandbox).

## File model in host mode

A sandbox's only writable area is its home (Landlock), and its default cwd is the home.
So the file API roots every path at the home: `files.write("data/in.txt", ...)` →
`$HOME/data/in.txt`, a leading `/` is taken relative to the home, and `..` cannot escape
it. Files written via the API are `chown`ed to the sandbox uid so the sandbox's own code
can read/write them. This gives a clean "filesystem rooted at the sandbox" model that
matches what code inside the sandbox can touch.

## Numbers (cpu-basic, client on a laptop over the proxy)

| N sandboxes | hosts (50/host) | provision + create | exec echo (all) | kill all | TOTAL |
|---|---|---|---|---|---|
| 100  | 2  | 6.1s | 1.5s | 0.6s | **8.2s** |
| 1000 | 20 | 7.4s | 4.2s (1000/1000) | 4.2s | **15.8s** |

Server-side create/exec/delete are each ~1ms; the budget is network. The host VM's
~6s startup is a one-time cost amortized across all its sandboxes. See `BENCHMARKS.md`.

## Implementation

- **Server** (`sandbox-server`, single unified `sbx-server` binary): a `sandboxes`
  module + a `landlock` module (raw syscalls, zero deps). The dedicated routes
  (`/v1/exec`, `/v1/files/*`, `/v1/procs/*`) and the host routes (`/v1/sandboxes/*`,
  including per-sandbox `exec`/`files`/`procs`) live in the same ~672KB binary. Host mode
  also enforces `SBX_CAPACITY` (atomic slot reservation; replies `{"rejected": N}` when
  full) and runs the two-level idle watchdog (per-sandbox eviction + empty-host shutdown).
- **Client** (`huggingface_hub`): `SandboxPool` provisions/packs/scales hosts; a single
  `Sandbox` class serves both backends via a path prefix (`/v1/*` vs
  `/v1/sandboxes/<id>/*`) and a per-mode kill strategy (cancel Job vs `DELETE` sandbox).
  `SandboxPool.connect(pool_id)` rebuilds a pool from a running host job's env vars, and
  `create()` retries any `rejected` sandboxes on another (or a fresh) host.

## Reproduce

```bash
python sandbox/experiments/exp10_pool_scale.py --num 1000 --per-host 50 --flavor cpu-basic
```
