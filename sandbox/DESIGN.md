# `hf sandbox` — Design

Sandboxes on Hugging Face Jobs: spin up an isolated cloud machine in ~7s, run commands with
live-streamed output, move files in/out, expose ports — from Python or the CLI.
Inspired by E2B and Modal Sandboxes; built only on existing Jobs primitives
(`run_job`, `expose=`, `secrets=`, `labels=`).

## Architecture

```
┌──────────────┐   HTTPS (Bearer HF token ── proxy gate)    ┌─────────────────────────┐
│ Python client │ ─────────────────────────────────────────▶ │ https://<job>--8000.hf.jobs │
│  / hf CLI     │   X-Sandbox-Token (app gate)               │   HF Jobs proxy          │
└──────────────┘                                             └───────────┬─────────────┘
                                                                          ▼
                                                             ┌─────────────────────────┐
                                                             │ job container (any image)│
                                                             │  sbx-server (641KB Rust  │
                                                             │  static musl binary)     │
                                                             └─────────────────────────┘
```

- **sbx-server**: hand-rolled HTTP/1.1 server (no framework; tiny_http buffers chunked
  responses → unusable for live streaming, verified). NDJSON event streams for exec,
  raw bodies for files, explicit flush per chunk. Static musl binary → runs in *any*
  x86_64 Linux image, no Python/pip needed.
- **Bootstrap** (cold start measured at ~6.7s end-to-end): job command is a `/bin/sh -c`
  script that downloads the binary from a HF model repo (private; auth via `SBX_DL_TOKEN`
  job secret) using whichever of `wget` / `curl` / `python3` exists, `chmod +x`, `exec`.
  Covers alpine/busybox (wget), most images (curl), `python:*-slim` (python3).
  - Fallback `server_source="mount"` for images with `sh` but no downloader: mount the
    repo as a volume + `cp && chmod && exec` (costs ~3s extra; exec bit is NOT preserved
    on mounted volumes — verified, hence the cp).
  - Images without `/bin/sh` (distroless): unsupported v1; documented.

## Measured constraints (see experiments/NOTES.md)

| fact | consequence |
|---|---|
| cold start floor ≈ 6.5s (job scheduling) | poll `/health` directly at 150ms interval; no blind sleeps |
| proxy: ~105ms RTT keep-alive, ~456ms fresh conn | one persistent `Session` per Sandbox |
| proxy streams chunked responses incrementally | exec output = NDJSON over chunked HTTP |
| proxy requires Bearer HF token (401) | tunnel already namespace-gated |
| WebSocket works (~95ms RTT) | reserved for future PTY/interactive shell |
| volume mounts drop exec bit | download bootstrap is the default |
| job timeout is fixed at creation | sandbox max lifetime = `timeout`; `idle_timeout` watchdog in server exits early to stop billing |

## Auth model (two layers)

1. **Proxy gate**: HF token with read access to the job's namespace (enforced by hf.jobs).
2. **App gate**: `X-Sandbox-Token` checked by sbx-server (constant-time). Defense in depth —
   read-only namespace members can reach the proxy but must not exec.

Token derivation — **stateless reconnection** (no local files, no secrets in labels):

```
nonce  = random 128-bit hex                 # stored in job label "sandbox"
token  = HMAC-SHA256(key=user_hf_token, msg="hf-sandbox:" + nonce)
```

- `SBX_TOKEN` delivered to the server via job `secrets=` (encrypted; never in command/logs).
- `Sandbox.connect(job_id)` from any machine: read nonce from labels → recompute token.
- Per-sandbox isolation (unique nonce); a leaked sandbox token compromises one sandbox only.
- Other namespace members can't derive it (different HF token); they also can't read job secrets.
- HF token never enters the sandbox unless `forward_hf_token=True` (opt-in, separate secret).

## Python API

```python
from huggingface_hub import Sandbox

with Sandbox.create(image="python:3.12", flavor="cpu-basic", timeout="30m",
                    idle_timeout="10m", env=None, secrets=None, volumes=None,
                    expose=None, namespace=None, forward_hf_token=False) as sbx:

    sbx.id, sbx.image, sbx.url(8080)            # public URL of an extra exposed port

    # foreground exec — streams internally; raises SandboxCommandError on rc != 0
    r = sbx.run("pip install -q numpy")                       # shell string → /bin/sh -c
    r = sbx.run(["python", "-c", "print(40+2)"])              # argv form
    r.stdout, r.stderr, r.exit_code, r.duration                # CommandResult
    sbx.run("make -j", on_stdout=print, on_stderr=print, timeout=600, env={...}, cwd="/app", check=False)

    # background processes
    proc = sbx.spawn("python -m http.server 8080", tag="web") # → SandboxProcess
    proc.pid; proc.running; proc.wait(); proc.kill()
    for line in proc.logs(follow=True): ...
    proc.send_stdin("data\n", eof=False)
    sbx.processes()                                            # list[SandboxProcessInfo]

    # files
    sbx.files.write("/app/main.py", "print('hi')")            # str | bytes | IO | Path
    sbx.files.read("/data/out.bin")                            # bytes
    sbx.files.read_text("/etc/os-release")
    sbx.files.upload("local.csv", "/data/local.csv")
    sbx.files.download("/data/out.bin", "out.bin")
    sbx.files.list("/app"); sbx.files.stat(p); sbx.files.exists(p)
    sbx.files.mkdir(p); sbx.files.delete(p, recursive=False)

# lifecycle from anywhere
sbx = Sandbox.connect(job_id)         # stateless (HMAC token recomputed)
Sandbox.list()                        # jobs labelled as sandboxes, RUNNING only by default
sbx.kill()                            # cancel_job; context manager calls it too
```

Decisions:
- `run()` **raises** on non-zero exit by default (E2B-style; best DX for "run code, see error"),
  `check=False` opts out. `CommandResult` is returned either way.
- `run(str)` → `sh -c`; `run(list)` → argv. Both supported (E2B does str, Modal does argv;
  both are natural in different cases).
- Foreground `run()` has no default timeout (job timeout bounds it); the server emits
  keepalive pings every 15s so the proxy never sees an idle connection.
- Sync-only v1 (matches `HfApi`; async later if demanded).
- `create()` blocks until `/health` answers; fails fast with job logs if the job ERRORs.

## CLI

```
hf sandbox create [-i IMAGE] [--flavor F] [--timeout T] [--idle-timeout T]
                  [-e KEY=VAL]... [--expose PORT]... [--volume ...]... [--namespace NS]
hf sandbox ls
hf sandbox info <id>
hf sandbox exec <id> [-w CWD] [-e KEY=VAL]... -- CMD [ARGS...]   # streams, propagates exit code
hf sandbox ps <id>                                                # processes inside the sandbox
hf sandbox cp <local> <id>:<remote>   |   <id>:<remote> <local>   # docker-style
hf sandbox url <id> <port>
hf sandbox kill <id> [-y]
```

- `exec` prints stdout→stdout / stderr→stderr live and exits with the remote exit code
  (scripting-friendly: `hf sandbox exec $ID -- pytest && ...`).
- No `hf sandbox shell` in v1 (needs PTY over WebSocket; the pieces are proven, future work).
- Implementation: `src/huggingface_hub/cli/sandbox.py` following repo CLI conventions.

## Server HTTP protocol (v1)

```
GET  /health                              → {"status","version","uptime_ms"}   (no auth)
POST /v1/exec        {cmd, env?, cwd?, timeout?, stdin?, background?, tag?}
                     foreground → NDJSON stream: start / stdout / stderr / ping / exit
                     background → {"pid", "tag"}
GET  /v1/procs                            → [{pid,tag,cmd,running,exit_code,started_at_ms}]
GET  /v1/procs/{pid}/logs?follow=         → NDJSON replay (+live if follow)
GET  /v1/procs/{pid}/wait                 → NDJSON pings until {"event":"exit",...}
POST /v1/procs/{pid}/kill  {signal?}      → default SIGKILL, to the process group
POST /v1/procs/{pid}/stdin?eof=           → raw body to stdin
GET  /v1/files/read?path=                 → raw bytes
PUT  /v1/files/write?path=&mode=&mkdir=   → raw body to file (parents created)
GET  /v1/files/list?path=  /stat?path=
DELETE /v1/files/delete?path=&recursive=
POST /v1/files/mkdir?path=
```

Env config: `SBX_PORT` (8000), `SBX_TOKEN` (removed from env before any child spawns),
`SBX_IDLE_TIMEOUT` (secs; exits when idle and no running process).

## Cost guardrails

- `idle_timeout` default **10m**: abandoned sandboxes self-terminate even if the client vanished.
- Context manager + atexit kill on the client side.
- Default job `timeout` 30m (Jobs default), explicit override for long sessions.

## Non-goals v1 (documented future work)

PTY/interactive shell (WebSocket proven), file watch, pause/resume (no Jobs support),
async client, sliding timeout extension (no Jobs API), Windows containers, arm64 binary.
