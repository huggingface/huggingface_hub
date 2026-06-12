# Experiment notes

## Exp 01 — cold start baseline (2026-06-11)

Job: `python:3.12`, `python -m http.server 8000`, `expose=[8000]`, cpu-basic, namespace Wauplin.

| metric | value |
|---|---|
| `run_job()` API call | 0.27s |
| status == RUNNING | 6.1s |
| exposed URL serving (real server response) | **6.7s** |

- `expose_urls` available on JobInfo as soon as job is created: `https://<job_id>--<port>.hf.jobs`.
- Proxy auth: requires `Authorization: Bearer <hf_token>` with read access to namespace. 401 otherwise (no auth or bad token). → the tunnel is already access-controlled; still want an app-level sandbox token as 2nd layer (proxy gate = whole namespace, incl. read-only members).
- **Proxy request latency: ~105ms median with keep-alive connection, ~456ms with fresh connection.** → client MUST use a persistent session. TLS+conn setup is ~350ms.

## Exp 02 — proxy capabilities (2026-06-11)

Job: `python:3.12` + `pip install aiohttp` + aiohttp app, `expose=[8000, 8001]`.

- **Chunked streaming: WORKS, fully incremental.** Chunks emitted 0.5s apart arrive 0.5s apart (no buffering in proxy). → live exec output streaming over plain HTTP works.
- **SSE: works.**
- **WebSocket: works** through proxy (`wss://`), ~95ms RTT — same as HTTP keep-alive. No latency benefit over HTTP; keep for future PTY/interactive shell.
- **Multi-port expose works** (`expose=[8000, 8001]` → two URLs).
- RUNNING at +5.6s; pip install aiohttp + app boot → ready at +8.4s (pip adds ~2.5s for a small wheel; fastapi+uvicorn would be worse).

## Decisions so far

- Transport: HTTP/1.1 keep-alive + chunked streaming responses. No websocket needed for v1 exec path.
- Auth: proxy Bearer token (automatic) + `X-Sandbox-Token` secret header checked by in-job server (delivered via job `secrets=`).
- Server: static Rust binary (musl) — no pip, no Python requirement in image.
- Open question → exp03: how to get the binary into ANY image. Options: (a) mount HF model repo volume containing binary (does exec bit survive? mount overhead?), (b) curl download at startup (needs curl/wget in image).

## hf-sandbox (old prototype) takeaways (from repo review)

- 30–90s cold start: pod scheduling + `pip install fastapi uvicorn` + uvicorn boot + hardcoded 15s blind sleep before first health poll. We're at 6.7s without pip.
- Keep: dual-layer auth via job secrets; `exec()` → familiar result shape; atexit cleanup; opt-in `forward_hf_token`.
- Fix: no streaming, base64-in-JSON file transfer, no reconnect/list, no CLI, Python+pip required in image, URL string-templating (now JobInfo.status.expose_urls exists).
