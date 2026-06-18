<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->
# Sandboxes

A **sandbox** is an isolated cloud machine you can spin up in seconds, run commands in with
live-streamed output, and move files in and out of — all from Python or the CLI. Sandboxes are
built on top of [Jobs](./jobs): under the hood, a sandbox is just a Job running a tiny server that
exposes command execution and file transfer over HTTP.

They are a good fit whenever you need to run code somewhere other than your own machine:

- **Running untrusted or AI-generated code** — let an agent execute arbitrary code without giving
  it access to your filesystem.
- **Reproducible builds and experiments** — run on a clean, well-defined image, on CPU or GPU.
- **Fanning out work** — launch hundreds of parallel environments (RL rollouts, evaluation,
  batch tool execution) cheaply.

Any Docker image with `/bin/sh` works — no Python, pip, or agent needs to be preinstalled (a small
static server binary is injected at startup).

> [!TIP]
> Like Jobs, sandboxes are available to [Pro users](https://huggingface.co/pro) and
> [Team or Enterprise organizations](https://huggingface.co/enterprise). You pay only for the
> seconds the sandbox is alive (a `cpu-basic` sandbox costs $0.01/hour).

> [!TIP]
> Curious how this works under the hood — the in-job server, the stateless auth, or how a single
> Job hosts many isolated sandboxes? See the [Sandboxes conceptual guide](../concepts/sandbox).

## The two kinds of sandbox

There are two ways to get a sandbox. Both hand you the **same** [`Sandbox`] object — same
`run`, `spawn`, `files`, `connect`, `kill` — they differ only in how the underlying machine is
allocated:

| | [`Sandbox.create`] — **dedicated** | [`SandboxPool`] — **shared / pool** |
|---|---|---|
| mapping | one Job = **one sandbox** (a whole VM) | one Job = **many sandboxes** (one VM, packed) |
| isolation | full VM | uid + [Landlock](https://docs.kernel.org/userspace-api/landlock.html) (same-user trust) |
| cold start | ~6s per sandbox | ~6s for the first host, then ~1 round-trip each |
| cost | one VM per sandbox | one VM per **host**, amortized across many sandboxes |
| GPU | ✅ | ❌ (CPU only) |
| best for | a single sandbox, GPU workloads, untrusted code | many cheap CPU sandboxes (RL rollouts, fan-out) |

Rule of thumb: **need a GPU or to run mutually-untrusted code → dedicated. Need hundreds of cheap
CPU sandboxes → a pool.**

## Quickstart

```python
>>> from huggingface_hub import Sandbox

>>> with Sandbox.create() as sbx:                      # ready in ~6s
...     result = sbx.run("python -c 'print(40 + 2)'")  # ~100ms per command
...     print(result.stdout)
42
```

Pick any image and hardware [flavor](./jobs#select-the-hardware):

```python
>>> sbx = Sandbox.create(image="alpine:3.20")
>>> sbx = Sandbox.create(image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel", flavor="a10g-small")
```

## Running commands

[`Sandbox.run`] executes a command and waits for it. Pass a shell string or an argv list:

```python
>>> sbx.run("pip install -q numpy")                       # runs through /bin/sh -c
>>> sbx.run(["python", "-c", "import numpy; print(numpy.__version__)"])

# Live output streaming, plus env, cwd, timeout, stdin
>>> sbx.run("make -j4", cwd="/app", env={"CC": "gcc"}, timeout=600, on_stdout=print, on_stderr=print)
```

A command that exits non-zero raises [`SandboxCommandError`] (with `stdout`, `stderr` and
`exit_code` attached). Pass `check=False` to get the [`CommandResult`] back instead of raising:

```python
>>> result = sbx.run("test -f /tmp/missing", check=False)
>>> result.exit_code
1
```

Start background processes with [`Sandbox.spawn`]:

```python
>>> server = sbx.spawn("python -m http.server 8080", tag="web")
>>> server.pid, server.running
(112, True)
>>> for stream, data in server.logs(follow=True):   # tail logs live
...     print(stream, data)
>>> server.kill()
```

## Files

```python
>>> sbx.files.write("/app/script.py", "print('hi')")     # str | bytes | file-like
>>> sbx.files.read_text("/app/script.py")
"print('hi')"
>>> sbx.files.upload("local_data.csv", "/data/data.csv")  # local -> sandbox
>>> sbx.files.download("/data/results.bin", "results.bin")  # sandbox -> local
>>> sbx.files.list("/data")
[FileEntry(name='data.csv', path='/data/data.csv', type='file', size=5324, ...)]
```

Other helpers: `stat`, `exists`, `mkdir`, `delete`. Transfers above 8 MiB automatically use
parallel ranged requests (several hundred MiB/s from a well-connected machine).

## Lifecycle

A sandbox outlives the process that created it — you can create it now and reconnect later, from
any machine that holds the same HF token, with no state to copy around:

```python
>>> sbx = Sandbox.create()
>>> sbx.id
'687f911eaea852de79c4a50a'

# Later, from anywhere:
>>> sbx = Sandbox.connect("687f911eaea852de79c4a50a")

>>> Sandbox.list()       # your running sandboxes
>>> sbx.kill()           # terminate now
```

- `idle_timeout` (default 10 minutes) is the real keeper: it shuts the sandbox down once no API
  call is made and no process is running, so abandoned sandboxes stop billing. Set it at create
  time (`Sandbox.create(idle_timeout="30m")`) or pass `None` to disable.
- The job also has a fixed 24h maximum lifetime as a hard backstop (not configurable).
- Your HF token is **never** sent into the sandbox unless you opt in with `forward_hf_token=True`.

## Many sandboxes at once: SandboxPool

When you need *many* sandboxes (parallel RL rollouts, fan-out evaluation, batch tool execution),
one Job per sandbox is wasteful: each pays a full VM cold start and holds a whole machine for a
workload that needs a few MB of RAM. [`SandboxPool`] packs many lightweight sandboxes into a few
shared **host** Jobs instead — one billed VM serves dozens of sandboxes, so per-sandbox cost drops
by that factor and per-sandbox cold start is ~one network round-trip.

```python
>>> from huggingface_hub import SandboxPool

>>> with SandboxPool(image="python:3.12", flavor="cpu-basic", warm_up=2) as pool:
...     boxes = [pool.create() for _ in range(100)]   # packed across the 2 warm host VMs
...     print(boxes[0].run("echo hi").stdout)          # each box is a normal Sandbox
hi
```

Each `create()` returns **one** full [`Sandbox`]; call it repeatedly to fan out. The pool boots host
Jobs as needed, packs `sandboxes_per_host` sandboxes per host, and terminates everything on
`close()` (or when a host goes idle, as a billing backstop). The typical fan-out pattern:

```python
>>> from concurrent.futures import ThreadPoolExecutor
>>> with SandboxPool(image="python:3.12", warm_up=4) as pool:
...     boxes = [pool.create() for _ in tasks]
...     with ThreadPoolExecutor(32) as ex:
...         outputs = list(ex.map(lambda b, t: b.run(t.cmd).stdout, boxes, tasks))
```

Env, secrets and `idle_timeout` are **per-sandbox** (they belong to `create()`, not the pool), so
sandboxes in one pool can have different environments:

```python
>>> sbx = pool.create(env={"SEED": "42"}, idle_timeout="5m")
```

### Growing on demand and pre-warming

`pool.create()` makes one sandbox at a time, **reusing a host that still has free capacity** before
booting a new one — so you can spawn sandboxes as work arrives and they pack themselves onto warm
hosts:

```python
>>> pool = SandboxPool(image="python:3.12", flavor="cpu-basic")
>>> sbx = pool.create()    # boots the first host (~6s)
>>> sbx = pool.create()    # packs onto the same warm host (~one round-trip, no new VM)
```

To avoid the host cold start on the first few calls, pre-provision hosts with `warm_up=N` (booted on
the first `create()`) or by calling `pool.warm(N)` up front.

### Reusing pools across processes and machines

Warm hosts are discovered through job labels, so reuse works **across processes**: a brand-new
`SandboxPool` with the same `image`/`flavor`/`name` attaches to hosts an earlier run left running
instead of booting its own. Pass a `name=` to keep separate pools from sharing hosts.

To reattach from another machine with no local state, reconnect by pool id with
[`SandboxPool.connect`] — it finds a running host, rebuilds the pool's config (image, flavor, packing
density) from that host job, and is ready to `create()` more:

```python
>>> pool = SandboxPool.connect("pool-ae9f7efe0bc7")   # from anywhere, no config needed
>>> sbx = pool.create()
```

> [!WARNING]
> Sandboxes within a host are isolated from each other by distinct uids plus a per-sandbox Landlock
> ruleset — they cannot read, signal, or write each other's files, and each is confined to its own
> private home. This is the right boundary for *one user's own* parallel workloads. For
> mutually-hostile untrusted code, or for GPU, use [`Sandbox.create`] (a separate VM per sandbox).
> The trade-offs are detailed in the [conceptual guide](../concepts/sandbox#isolation-in-a-pool-uid--landlock).

## From the CLI

The `hf sandbox` command mirrors the Python API. A dedicated sandbox:

```bash
>>> hf sandbox create
✓ Sandbox ready id=687f911eaea852de79c4a50a image=python:3.12 elapsed=6.0s

>>> hf sandbox exec 687f911eaea852de79c4a50a -- python -c "print('hi')"
hi

>>> hf sandbox cp data.csv 687f911eaea852de79c4a50a:/data/data.csv
>>> hf sandbox ls
>>> hf sandbox ps 687f911eaea852de79c4a50a       # processes running inside the sandbox
>>> hf sandbox kill 687f911eaea852de79c4a50a
```

`hf sandbox exec` streams output live and exits with the command's exit code, so it composes in
scripts:

```bash
hf sandbox exec $ID -- pytest && echo "tests passed"
```

For many cheap shared sandboxes, warm a pool once and then create into it on demand:

```bash
# Warm a pool -> prints a pool id (billing starts: a host VM is now running)
>>> hf sandbox pool create --image python:3.12 --flavor cpu-basic
✓ Pool created id=pool-ae9f7efe0bc7 image=python:3.12 flavor=cpu-basic host=687f... elapsed=5.7s

# Each create packs onto a host with room (found by the pool id, from any machine);
# only when every host is full does it boot a duplicate. Env/secrets are per-sandbox.
>>> hf sandbox create --pool pool-ae9f7efe0bc7 --secrets OPENAI_API_KEY=sk-...
>>> hf sandbox create --pool pool-ae9f7efe0bc7

>>> hf sandbox pool ls
>>> hf sandbox pool delete pool-ae9f7efe0bc7    # terminate the pool's hosts (and their sandboxes)
```

`hf sandbox create --pool` produces a shared sandbox; its id looks like
`<host_job_id>.<local_id>` and works everywhere a dedicated id does (`exec`, `cp`, `kill`). A pool
has **no local state** — it is just its running host VMs, found by the pool id — so it works from any
machine and stops existing once all its hosts are gone (killed or idle-timed-out).
