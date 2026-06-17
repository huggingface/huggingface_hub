<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->
# Sandboxes

Sandboxes are isolated cloud machines built on top of [Jobs](./jobs): spin one up in seconds, run
commands with live-streamed output, move files in and out, expose ports publicly — from Python or
the CLI. They are ideal for running untrusted or AI-generated code, reproducible builds, or quick
experiments on any hardware (CPUs, GPUs).

There are two ways to get a sandbox, sharing the exact same `Sandbox` surface (`run`, `spawn`,
`files`, ...):

- [`Sandbox.create`] — **one dedicated Job per sandbox**: a full isolated VM. Best for a single
  sandbox, GPU workloads, untrusted code, or public port forwarding. ~6s cold start.
- [`SandboxPool`] — **many lightweight sandboxes packed into shared "host" Jobs**, isolated from
  each other by uid + the Landlock LSM. Best for fanning out many cheap CPU sandboxes (e.g. RL
  rollouts): the cost of one VM is amortized across dozens of sandboxes and per-sandbox cold start
  is ~one network round-trip. See [Many sandboxes at once](#many-sandboxes-at-once-sandboxpool).

> [!TIP]
> Like Jobs, sandboxes are available to [Pro users](https://huggingface.co/pro) and
> [Team or Enterprise organizations](https://huggingface.co/enterprise). You pay only for the
> seconds the sandbox is alive (a `cpu-basic` sandbox costs $0.01/hour).

## Quickstart

```python
>>> from huggingface_hub import Sandbox

>>> with Sandbox.create() as sbx:                      # ready in ~6s
...     result = sbx.run("python -c 'print(40 + 2)'")  # ~100ms per command
...     print(result.stdout)
42
```

Any Docker image with `/bin/sh` works — no Python, pip, or agent preinstalled in the image is
required (a tiny static server binary is injected at startup):

```python
>>> sbx = Sandbox.create(image="alpine:3.20")
>>> sbx = Sandbox.create(image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel", flavor="a10g-small")
```

## Running commands

[`Sandbox.run`] executes a command and waits for it. Pass a shell string or an argv list:

```python
>>> sbx.run("pip install -q numpy")                       # /bin/sh -c
>>> sbx.run(["python", "-c", "import numpy; print(numpy.__version__)"])

# Live output streaming, env, cwd, timeout, stdin
>>> sbx.run("make -j4", cwd="/app", env={"CC": "gcc"}, timeout=600, on_stdout=print, on_stderr=print)
```

A command that exits with a non-zero code raises [`SandboxCommandError`] (with `stdout`, `stderr`
and `exit_code` attached); pass `check=False` to get the [`CommandResult`] instead.

Start background processes with [`Sandbox.spawn`]:

```python
>>> server = sbx.spawn("python -m http.server 8080", tag="web")
>>> server.pid, server.running
(112, True)
>>> for stream, data in server.logs(follow=True):
...     print(stream, data)
>>> server.kill()
```

## Files

```python
>>> sbx.files.write("/app/script.py", "print('hi')")
>>> sbx.files.read_text("/app/script.py")
"print('hi')"
>>> sbx.files.upload("local_data.csv", "/data/data.csv")
>>> sbx.files.download("/data/results.bin", "results.bin")
>>> sbx.files.list("/data")
[FileEntry(name='data.csv', path='/data/data.csv', type='file', size=5324, ...)]
```

Transfers above 8 MiB automatically use parallel ranged requests (several hundred MiB/s from a
well-connected machine).

## Exposing ports

Expose extra ports at creation and get their public URLs with [`Sandbox.url`]. Requests to these
URLs require an HF token with read access to the sandbox's namespace:

```python
>>> sbx = Sandbox.create(expose=[8080])
>>> sbx.spawn("python -m http.server 8080")
>>> sbx.url(8080)
'https://<sandbox_id>--8080.hf.jobs'
```

## Lifecycle

```python
>>> sbx = Sandbox.create(timeout="1h", idle_timeout="10m")
>>> sbx.id
'687f911eaea852de79c4a50a'

# From any machine with the same HF token (no state to copy around):
>>> sbx = Sandbox.connect("687f911eaea852de79c4a50a")

>>> Sandbox.list()       # running sandboxes
>>> sbx.kill()           # terminate now
```

- `timeout` is the maximum lifetime (a Jobs timeout; 30 minutes by default).
- `idle_timeout` (default 10 minutes) terminates the sandbox automatically when no API call is made
  and no process is running — abandoned sandboxes don't keep billing.
- Your HF token is never sent to the sandbox unless you pass `forward_hf_token=True`.

## Many sandboxes at once: SandboxPool

When you need *many* sandboxes (parallel RL rollouts, fan-out evaluation, batch tool execution),
creating one Job per sandbox is wasteful: each pays a full VM cold start and holds a whole machine
for a workload that needs a few MB of RAM. [`SandboxPool`] packs many lightweight sandboxes into a
few shared **host** Jobs instead — one billed VM serves dozens of sandboxes, so the per-sandbox
cost drops by that factor and per-sandbox cold start is ~one round-trip.

```python
>>> from huggingface_hub import SandboxPool

>>> with SandboxPool(image="python:3.12", flavor="cpu-basic") as pool:
...     boxes = pool.create(count=100)          # ~2 host VMs (50 sandboxes each), booted in parallel
...     print(boxes[0].run("echo hi").stdout)    # each box is a normal Sandbox
hi
```

Each returned object is a full [`Sandbox`] (`run`, `spawn`, `files`, `connect`, `kill`). The pool
provisions host Jobs lazily as sandboxes are requested, packs `sandboxes_per_host` per host, and
terminates everything on `close()` (or when a host goes idle, as a billing backstop). The typical
fan-out pattern:

```python
>>> from concurrent.futures import ThreadPoolExecutor
>>> with SandboxPool(image="python:3.12") as pool:
...     boxes = pool.create(count=len(tasks))
...     with ThreadPoolExecutor(32) as ex:
...         outputs = list(ex.map(lambda b, t: b.run(t.cmd).stdout, boxes, tasks))
```

### Grow on demand (instead of warming a batch up front)

You don't have to ask for all your sandboxes at once. `pool.create()` (count 1) **reuses a host
that still has free capacity** before booting a new one, so you can spawn sandboxes one at a time
as work arrives and they pack themselves onto warm hosts:

```python
>>> pool = SandboxPool(image="python:3.12", flavor="cpu-basic")
>>> sbx = pool.create()    # boots the first host
>>> sbx = pool.create()    # packs onto the same warm host (~one round-trip, no new VM)
```

Warm hosts are discovered through job labels, so this works **across processes** too: a brand-new
`SandboxPool` (same `image`/`flavor`/`name`) — or a fresh `hf sandbox create --shared` — attaches
to hosts an earlier run left running, rather than booting its own. Pass a `name=` to keep separate
pools from sharing hosts, or `discover=False` to only use hosts a given pool created.

**Isolation & trust model.** Sandboxes within a host are isolated from each other by distinct uids
plus a per-sandbox Landlock ruleset: they cannot read, signal, or write each other's files, and
each is confined to its own private home (a leading `/` in a file path is taken relative to that
home). This is the right boundary for *one user's own* parallel workloads; for mutually-hostile
untrusted code, or for GPU, use [`Sandbox.create`] (a separate VM per sandbox). Per-sandbox exposed
ports are not available in shared mode (use [`Sandbox.create`] for [`Sandbox.url`]).

Shared sandbox ids look like `<host_job_id>.<local_id>` and work everywhere a dedicated id does
(`Sandbox.connect`, `hf sandbox exec/cp/kill`).

## From the CLI

```bash
# One dedicated sandbox
>>> hf sandbox create
✓ Sandbox ready id=687f911eaea852de79c4a50a image=python:3.12 elapsed=6.0s

>>> hf sandbox exec 687f911eaea852de79c4a50a -- python -c "print('hi')"
hi

>>> hf sandbox cp data.csv 687f911eaea852de79c4a50a:/data/data.csv
>>> hf sandbox ls
>>> hf sandbox kill 687f911eaea852de79c4a50a

# One shared sandbox on demand — reuses a warm host, or boots one if none has room
>>> hf sandbox create --shared
>>> hf sandbox create --shared     # packs onto the same host as the previous call

# Or a whole batch at once (implies shared mode)
>>> hf sandbox create -n 100
>>> hf sandbox kill --all          # tear down every sandbox and host
```

`hf sandbox exec` streams output live and exits with the command's exit code, so it composes in
scripts: `hf sandbox exec $ID -- pytest && echo green`.

## How it works

`Sandbox.create` starts a Job with the requested image whose command downloads a small (<1MB)
static server binary and executes it. The server exposes command execution and file transfer over
HTTP on a port published through the Jobs proxy (`https://<job_id>--<port>.hf.jobs`). Two layers
protect it: the proxy requires an HF token with read access to the namespace, and the server
itself requires a per-sandbox token (delivered via Job secrets) that only the creator can derive.
