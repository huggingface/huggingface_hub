<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->
# Sandboxes

Sandboxes are isolated cloud machines built on top of [Jobs](./jobs): spin one up in seconds, run
commands with live-streamed output, move files in and out, expose ports publicly — from Python or
the CLI. They are ideal for running untrusted or AI-generated code, reproducible builds, or quick
experiments on any hardware (CPUs, GPUs).

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

## From the CLI

```bash
>>> hf sandbox create
✓ Sandbox ready id=687f911eaea852de79c4a50a image=python:3.12 elapsed=6.0s

>>> hf sandbox exec 687f911eaea852de79c4a50a -- python -c "print('hi')"
hi

>>> hf sandbox cp data.csv 687f911eaea852de79c4a50a:/data/data.csv
>>> hf sandbox ls
>>> hf sandbox kill 687f911eaea852de79c4a50a
```

`hf sandbox exec` streams output live and exits with the command's exit code, so it composes in
scripts: `hf sandbox exec $ID -- pytest && echo green`.

## How it works

`Sandbox.create` starts a Job with the requested image whose command downloads a small (<1MB)
static server binary and executes it. The server exposes command execution and file transfer over
HTTP on a port published through the Jobs proxy (`https://<job_id>--<port>.hf.jobs`). Two layers
protect it: the proxy requires an HF token with read access to the namespace, and the server
itself requires a per-sandbox token (delivered via Job secrets) that only the creator can derive.
