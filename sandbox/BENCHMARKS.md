# Benchmarks (2026-06-11, cpu-basic, namespace Wauplin)

All numbers measured against real HF Jobs with the scripts in `experiments/`.

## Cold start (`Sandbox.create()` returns, server answering)

| scenario | time |
|---|---|
| `python:3.12`, download bootstrap (default) | **median 5.8s** (5.3–6.1, n=5) |
| `alpine:3.20`, download bootstrap | 6.0s |
| `alpine:3.20`, `server_source="mount"` | 8.8s (volume mount adds ~3s) |
| 4 sandboxes created in parallel | **6.1s wall total** (perfect parallelism) |

Breakdown of a typical run: `run_job` API call 0.3s → pod RUNNING ~5.5s → binary
download (641KB from HF CDN, in-DC) + server boot + first /health ≈ 0.3s.
The floor is job scheduling; bootstrap overhead is ≈0.5s. The old FastAPI-based
hf-sandbox was 30–90s (pip install + uvicorn + 15s blind sleep).

## Command execution

| metric | value |
|---|---|
| sequential `run("true")` | **p50 110ms**, p90 169ms, min 103ms |
| raw proxy RTT (keep-alive) | ~105ms → client overhead ≈ 0 |
| fresh connection (no keep-alive) | ~456ms → why the client keeps one session |
| 16 parallel `sleep 1` on one sandbox | 2.2s wall |
| streaming liveness (ticks every 300ms) | gaps median 300ms, max 339ms — zero buffering |

## File transfer through the proxy

Measured cloud-to-cloud (sandbox B ↔ sandbox A); a single TCP stream from a
residential connection is limited by the local link, not the proxy.

| path | throughput |
|---|---|
| download, single stream | 137 MiB/s |
| download, 8 parallel ranged requests | **340 MiB/s** |
| upload, single stream | 278 MiB/s |
| upload, 8 parallel ranged requests | **441 MiB/s** |
| in-sandbox loopback (server ceiling) | 715 MiB/s |

The client automatically switches to parallel ranged transfers above 8 MiB
(`offset`/`length` params on the server's read/write endpoints).

## Optimizations applied along the way

1. **Dropped tiny_http** — it buffers chunked responses until completion (verified:
   all chunks arrived at once). Hand-rolled HTTP/1.1 with per-chunk flush.
2. **Persistent HTTP session** in the client (saves ~350ms/request).
3. **Download bootstrap over volume mount** as default (saves ~3s cold start);
   mount kept as fallback for images without wget/curl/python3.
4. **Server port 49983** (not 8000) so common dev ports stay free for users.
5. **Parallel ranged file transfers** above 8 MiB (2.5× download, 1.6× upload, more
   on high-BDP paths).
6. **15s keepalive pings** in all streaming responses so the proxy never kills an
   idle connection mid-command.
7. **Readiness polling at 150ms** directly against /health (no blind sleep), with
   job-status checks every 2s to fail fast with logs when the image is broken.

## SandboxPool — shared/host mode at scale (2026-06-17, cpu-basic)

`SandboxPool` packs many landlock-isolated sandboxes into shared host VMs. Measured
with `experiments/exp10_pool_scale.py` against real HF Jobs (client on a laptop,
all traffic through the Jobs proxy):

| N sandboxes | hosts (50/host) | provision + create all | exec `echo` in all | kill all | TOTAL |
|---|---|---|---|---|---|
| 100  | 2  | 6.1s | 1.5s (93/100, no retry) | 0.6s | **8.2s** |
| 1000 | 20 | 7.4s | 4.2s (1000/1000, 23 needed 1 proxy retry) | 4.2s | **15.8s** |

- **Provision + create** boots `ceil(N/per_host)` hosts in parallel and creates all N
  sandboxes with one batched request per host — so 1000 sandboxes cost roughly one host
  cold start (~6s), not N. Server-side `create` is ~1ms/sandbox (mkdir + chown + Landlock
  ruleset), dwarfed by the proxy round-trip.
- **exec** fans out ≤200 concurrent `run("echo")` calls over the proxy: p50 738ms, p95
  1054ms per call (proxy-RTT-bound, not server-bound). Under heavy fan-out the proxy
  occasionally drops a connection (transient `RemoteProtocolError`); a single retry brought
  1000/1000 to success.
- **Cost**: 1000 sandboxes for ~16s used 20 × cpu-basic ≈ 20 × $0.01/h × (16/3600) ≈
  **$0.0009 total** — vs ~$0.06 for 1000 one-job-per-sandbox VMs alive the same duration,
  and without the 1000-VM scheduling burst.

## Cost reference

cpu-basic is $0.01/hour. A dedicated sandbox that cold-starts, runs a command and is killed
costs ~$0.00002. The 10-minute default idle watchdog bounds the cost of leaked sandboxes to
~$0.002. In shared mode you pay per *host* VM, not per sandbox, so packing `sandboxes_per_host`
sandboxes divides the per-sandbox cost by that factor.
