# hf sandbox — final report

**Status: working end-to-end.** Sandboxes on HF Jobs: ~6s cold start, ~105ms command
latency, live output streaming, parallel file transfer, public port forwarding, works
in any Docker image with `/bin/sh` (no Python/pip required). Try it:

```bash
.venv/bin/python sandbox/demo.py        # full tour, ~20s, costs ~$0.001
hf sandbox create                        # CLI
```

## What was built

| piece | where | what |
|---|---|---|
| Rust server (`sbx-server`) | [github.com/Wauplin/sandbox-server](https://github.com/Wauplin/sandbox-server) | 641KB static musl binary: exec with NDJSON streaming, background procs, raw-body file API, idle watchdog. Hand-rolled HTTP/1.1 (tiny_http buffers chunked responses — verified — so no framework). Binary hosted at `Wauplin/sbx-server` on the Hub (private). |
| Python API (`Sandbox` + `SandboxPool`) | `src/huggingface_hub/_sandbox.py` | `Sandbox.create/connect/list/kill`, `run/spawn/processes`, `files.*`, `url(port)`, context manager. `SandboxPool` packs many landlock sandboxes per host Job (shared mode, see [HOST_MODE.md](HOST_MODE.md)). Exported from `huggingface_hub`. |
| CLI (`hf sandbox`) | `src/huggingface_hub/cli/sandbox.py` | `create [-n N] / ls / exec / ps / cp / url / kill`, follows repo CLI conventions. |
| Tests | `tests/test_sandbox.py` | 31 tests (helpers + dedicated client + shared client + pool, against an in-process fake server). All pass; `make quality` clean. |
| Docs | `docs/source/en/guides/sandbox.md`, `package_reference/sandbox.md`, sections in `guides/cli.md` + `guides/jobs.md` | guide + API reference, registered in toctree. |
| Reports | `DESIGN.md`, `BENCHMARKS.md`, `experiments/` | design rationale, measured numbers, all probe/benchmark scripts. |

## The 30-second pitch

```python
from huggingface_hub import Sandbox

with Sandbox.create() as sbx:                          # any image, ready in ~6s
    sbx.files.write("/app/main.py", code)
    result = sbx.run("python /app/main.py")            # raises with stderr on failure
    server = sbx.spawn("python -m http.server 8080")   # background process
    sbx = Sandbox.connect(sandbox_id)                  # from any machine, stateless

# Fan out cheaply: many landlock sandboxes packed into shared host VMs
from huggingface_hub import SandboxPool
with SandboxPool(image="python:3.12") as pool:         # see HOST_MODE.md
    boxes = pool.create(count=1000)                    # ~20 cpu-basic hosts, ~16s end-to-end
```

## Key design decisions (and why)

1. **Rust static binary instead of FastAPI bootstrap** — works in any image, no pip,
   cold start 6s instead of 30–90s for the old hf-sandbox prototype.
2. **Hand-rolled HTTP/1.1 server** — tiny_http (and most minimal frameworks) buffer
   chunked responses; live streaming requires flushing each chunk. Verified by test.
3. **Download bootstrap (wget→curl→python3) over volume mount** — volume mounts drop
   the exec bit (verified: `permission denied`) and add ~3s; downloading the 641KB
   binary from the HF CDN inside the DC is free. Mount kept as `server_source="mount"`
   fallback; the binary repo stays private (token passed as job secret).
4. **Stateless auth**: sandbox token = `HMAC(hf_token, nonce)`, nonce in job labels,
   token delivered to the server via job secrets. `connect(id)` needs no local state;
   the HF token never enters the sandbox (unless `forward_hf_token=True`). Two layers:
   the hf.jobs proxy already enforces namespace read access (401 otherwise).
5. **Server on port 49983** so user ports (3000/8000/8080...) stay free.
6. **`run()` raises on non-zero exit** (E2B-style, `check=False` to opt out) — errors
   surface with stderr in the message, the best DX for run-code-see-error loops.
7. **Idle watchdog (default 10m) instead of client-side atexit kill** — persistent
   sandboxes are a feature (create now, connect later); leaked ones still die.
8. **Parallel ranged file transfers** above 8 MiB (offset/length params server-side).

## Numbers (see BENCHMARKS.md)

- Cold start: **5.8s median**; 4 parallel sandboxes in 6.1s wall.
- Exec: **p50 110ms** (proxy RTT floor is ~105ms; client overhead ≈0).
- Streaming: ticks emitted every 300ms arrive with median gap 300ms (zero buffering).
- Files (cloud-to-cloud): **340 MiB/s down / 441 MiB/s up** with parallel ranges.
- WebSocket also works through the proxy (~95ms RTT) — reserved for a future PTY shell.

## Future work (deliberately not in v1)

- `hf sandbox shell` — interactive PTY over WebSocket (proxy support verified).
- Official binary hosting (e.g. `huggingface/sbx-server` repo, versioned, arm64 build)
  instead of `Wauplin/sbx-server`; pin client↔server protocol versions.
- Async client; pause/resume (needs Jobs support); sliding timeout extension (needs
  Jobs API); directory upload (tar); `files.watch`.
- Telemetry on create/kill like the old prototype.

## Caveats

- Images without `/bin/sh` (distroless) are unsupported.
- Sandbox lifetime is capped by the job `timeout` (default 30m) — there is no way to
  extend a running job today.
- `connect()` requires the *same* HF token that created the sandbox (HMAC-derived).
  Other org members can't connect (by design, for now).
