# HF Sandbox vs E2B vs Modal

Feature-by-feature comparison with equivalent code. `sbx`/`sb` is a created sandbox object.

| Capability | **HF Sandbox** | **E2B** | **Modal** |
|---|---|---|---|
| Create | `Sandbox.create(image="python:3.12", flavor="a10g-small")` — any Docker Hub / `hf.co/spaces` image with `/bin/sh` | `Sandbox.create(template="base")` — pre-built E2B Template (custom images need their build toolchain) | `modal.Sandbox.create(app=app, image=modal.Image.from_registry(...), gpu="A10G")` — requires an `App` |
| Reconnect by id | `Sandbox.connect(id)` — stateless (HMAC-derived token) | `Sandbox.connect(id)` | `Sandbox.from_id(id)` / `from_name(app, name)` |
| List | `Sandbox.list()` | `Sandbox.list(query=...)` → paginator | `Sandbox.list(app_id=..., tags=...)` → generator |
| Terminate | `sbx.kill()` | `sbx.kill()` | `sb.terminate()` |
| Run command (wait) | `sbx.run("pip install x")` or `sbx.run(["python", "-c", ...])` → `CommandResult`, raises on non-zero | `sbx.commands.run("pip install x")` (shell string only) → `CommandResult`, raises | `p = sb.exec("pip", "install", "x")` (argv only) → `ContainerProcess`, then `p.wait()`; no raise |
| Background process | `proc = sbx.spawn(cmd, tag="web")` → pid, `wait/kill/logs/send_stdin` | `handle = commands.run(cmd, background=True)`; reattach via `commands.connect(pid)` | `sb.exec(...)` without `wait()`; no reattach-by-pid |
| Live output streaming | `sbx.run(cmd, on_stdout=..., on_stderr=...)`; `proc.logs(follow=True)` | `on_stdout`/`on_stderr` callbacks; iterate handle | iterate `p.stdout` (`StreamReader`); no callbacks |
| stdin | `run(stdin=...)`; `proc.send_stdin(data, eof=True)` | `commands.send_stdin(pid, data)` | `p.stdin.write(...)` + `write_eof()` |
| Interactive PTY | ✗ (roadmap — WebSocket through the proxy verified working) | `sbx.pty.create/resize/send_stdin` | `exec(..., pty=True)` |
| Per-command env / cwd / timeout | `run(env=..., cwd=..., timeout=...)` | `run(envs=..., cwd=..., timeout=..., user=...)` | `exec(..., env=..., workdir=..., timeout=...)` |
| Files: read/write | `sbx.files.read/read_text/write` (raw bytes, parallel ranged >8 MiB) | `files.read/write/write_files` | `sb.filesystem.read_bytes/write_bytes` |
| Files: local transfer | `files.upload(local, remote)` / `files.download(remote, local)` | via `files.write(path, open(f, "rb"))` / `files.read` | `filesystem.copy_from_local/copy_to_local` |
| Files: management | `files.list/stat/exists/delete/mkdir` | `files.list/exists/remove/rename/make_dir/get_info` | `filesystem.list_files/stat/remove/make_directory` |
| File watching | ✗ | `files.watch_dir(path)` | `filesystem.watch(path)` |
| Expose ports | `Sandbox.create(expose=[8080])` → `sbx.url(8080)`; **URL requires an HF token** (namespace-gated) | every port automatic: `sbx.get_host(3000)` → public URL | declared `encrypted_ports=[8000]` → `sb.tunnels()[8000].url`, public random URL |
| Lifetime | job `timeout` (fixed at create) + `idle_timeout` watchdog | sliding `set_timeout()`; max 1h/24h continuous | `timeout` (max 24h) + `idle_timeout` |
| Pause / snapshot | ✗ (no Jobs support today) | `pause()` + `connect()` auto-resume (disk+memory) | `snapshot_filesystem()`; experimental memory snapshots |
| Hardware | any Jobs flavor: CPU → T4/A10G/A100/H200 GPUs | CPU only | `cpu=`, `memory=`, `gpu="H100"` |
| CLI | `hf sandbox create/ls/exec/ps/cp/url/kill` | `e2b sandbox list/kill`, template tooling | `modal shell`, app-centric CLI |
| Transport | HTTPS through the Jobs proxy → static Rust server in-sandbox | REST (lifecycle) + Connect RPC to `envd` in-VM | gRPC to Modal control plane (never direct) |
| Isolation unit | Docker container on HF Jobs infra | Firecracker microVM | gVisor container |
| Cold start | ~6s (job scheduling floor) | ~150ms | <1s (cached images) |
| Billing | per-second Jobs pricing (cpu-basic $0.01/h) | per-second vCPU/RAM | per-second CPU/GPU/RAM |

## Takeaways

- **API shape**: HF Sandbox deliberately blends the two — E2B's lifecycle (`create`/`connect`/`kill`, raising `run`) with support for both shell strings (E2B-style) and argv lists (Modal-style). Naming note: the Python API uses `run()` (matches `subprocess.run`/E2B); the CLI uses `exec` (matches `docker exec` — running in an *existing* environment, vs `hf jobs run` which creates one).
- **Where we win**: zero infra lock-in (any Docker image, no template build step), GPUs up to H200, token-gated port URLs by default, stateless reconnection, per-second Jobs billing.
- **Where they win**: cold start (Firecracker/gVisor pools vs job scheduling), pause/resume & snapshots, PTY + file watching (both on our roadmap), E2B's sub-second boots for bursty agent workloads.
