# Review of the sandbox network policy PRs

Reviewed on 2026-11-04:

- sandbox-server pre-draft `3b4a51c2-9800-4a62-b48a-2ba73b47ab6a`, head `369ad10fb7551dea49db0e78de23a916d01e3aec` (base `74ad0f1b92955faa613221b7e0348d0d9dcc1732`).
- huggingface_hub pre-draft `8fdfe73b-bafc-4188-864f-0c111c39afdd`, implementation head `dd0787bb2b2808304052ca24aa685c93b8e2d6d8` (base `bd7e65e442c5baeecaa0c946b9a9eb9f31d93fae`).

## Findings

### [P1] Socket replacement breaks existing epoll registrations and duplicates

`sandbox-server/src/network.rs:689-696`: `SECCOMP_IOCTL_NOTIF_ADDFD` replaces the caller's descriptor with a different open file description. Any epoll registration created before `connect()` still refers to the old socket. Nonblocking clients that register before connecting therefore miss connection completion, even for an explicitly permitted destination. This is more than the documented loss of socket options.

Reproduced against a listening local HTTP service, using the built musl server in a disposable Docker container. With no policy, the following returns an EPOLLOUT event. With `SBX_NETWORK='{"allow":["127.0.0.1:18080"]}'`, `connect_ex` returns EINPROGRESS, `ep.poll(1)` returns `[]`, and `select.select([], [s], [], 1)` confirms the replacement socket is writable:

```python
import select, socket
s = socket.socket()
s.setblocking(False)
ep = select.epoll()
ep.register(s, select.EPOLLOUT | select.EPOLLET)
ep.poll(0)  # Drain any initial readiness of the unconnected socket.
print(s.connect_ex(("127.0.0.1", 18080)))
print(ep.poll(1))
print(select.select([], [s], [], 1)[1])
```

The same replacement also breaks descriptor sharing: `other = socket.socket(fileno=os.dup(s.fileno()))` before a blocking `s.connect(...)` leaves `other.getpeername()` raising ENOTCONN. Without a policy both descriptors report the connected peer. A second connect to an already-connected socket also incorrectly succeeds instead of returning EISCONN. Preserve socket identity or change the mediation design before relying on allowlists with general-purpose clients; add regression coverage for pre-registered epoll sockets and duplicate descriptors.

### [P1] Blocking connects leave unbounded threads in the privileged server

`sandbox-server/src/network.rs:599-601, 683-684`: each connect notification creates an unrestricted server thread, which may block in `libc::connect()` with no application deadline or notification-cancellation handling. Killing or timing out the sandbox command does not interrupt that connect in the server. Repeated calls can accumulate threads and sockets outside the sandbox's RLIMIT_NPROC/RLIMIT_AS limits, eventually exhausting the shared host; `std::thread::spawn` panics on resource exhaustion (the release profile aborts on panic).

Reproduced without exhausting resources: a local TCP listener with backlog 0 was filled by one connection, then a sandbox command launched eight blocking connections to its allowlisted address. The exec timeout killed the command after one second (`signal: 9`, `timed_out: true`), but server thread count rose from 2 to 10 and remained 10 three seconds after timeout. All test processes were then terminated. Use bounded concurrency, nonblocking connections with a deadline, and cancellation when the seccomp notification becomes invalid. Test timeout/deletion while connects are pending.

## Client outcome and integration dependency

No additional actionable bug found in the huggingface_hub implementation. Its fail-closed checks are present in both dedicated and pooled creation paths. The client still pins sandbox-server 0.7.0, so the new `network=` functionality remains unavailable until a corrected server is released and the version/digest are updated, as already documented in the PR. This review did not rerun paid HF Jobs end-to-end tests or establish the absence of all possible network-policy bypasses.

## Checks performed

- `cargo test`: 57 passed.
- `cargo build --target x86_64-unknown-linux-musl`: passed.
- `scripts/network-regression.sh` in a disposable Python 3.12 Docker container using that musl binary: all checks passed, including dedicated none/allowlist and mixed host-mode policies. Kernel reported Landlock ABI 4.
- `PYTHONPATH=src /home/wauplin/projects/huggingface_hub/.venv/bin/python -m pytest tests/test_sandbox.py -q`: 170 passed.
- `make style`, then `make quality`, using the existing shared virtualenv and this worktree's source: passed; no generated source changes.
- Additional Docker probes reproduced both findings above; the existing suites do not cover them.

This commit records review findings only. No implementation fixes were made. Nothing was pushed or published, and both worktrees are retained.
