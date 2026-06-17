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

You can grow on demand instead of warming a batch: `pool.create()` (count 1) reuses a
host with free capacity before booting a new one, and warm hosts are discovered via
job labels (`hf-sandbox-host` + `hf-sandbox-capacity` + optional `hf-sandbox-pool`
name) so reuse works **across processes** — a fresh pool, or `hf sandbox create
--shared`, attaches to a host an earlier run left running.

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
  including per-sandbox `exec`/`files`/`procs`) live in the same ~672KB binary.
- **Client** (`huggingface_hub`): `SandboxPool` provisions/packs/scales hosts; a single
  `Sandbox` class serves both backends via a path prefix (`/v1/*` vs
  `/v1/sandboxes/<id>/*`) and a per-mode kill strategy (cancel Job vs `DELETE` sandbox).

## Reproduce

```bash
python sandbox/experiments/exp10_pool_scale.py --num 1000 --per-host 50 --flavor cpu-basic
```
