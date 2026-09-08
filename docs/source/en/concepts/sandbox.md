<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->
# Sandboxes under the hood

This guide explains how [Sandboxes](../guides/sandbox) work internally, and why they are built the way they are and what their limitations are. If you only want to use sandboxes, the [Sandboxes guide](../guides/sandbox) is enough; read on if you want to understand the mechanism, evaluate the trust model, or debug something.

> [!NOTE]
> The Sandbox API is experimental. Its API and behavior may change without notice.

## There is no "sandbox service"

The first thing to understand is that there is no dedicated sandbox backend. A sandbox is just an [HF Job](../guides/jobs) (a VM) running a single small static binary `sbx-server` that speaks HTTP. The client talks to that server through the Jobs proxy (the `*.hf.jobs` URL the Job exposes). Everything else — authentication, discovery, packing many sandboxes into one Job — is built out of existing Jobs primitives: labels, environment variables, and secrets.

```mermaid
flowchart LR
    subgraph local["Your machine"]
        C["huggingface_hub<br/>Sandbox / SandboxPool<br/>(or hf sandbox CLI)"]
    end
    subgraph hf["Hugging Face"]
        P["Jobs proxy<br/>(namespace auth)"]
        subgraph job["HF Job (a VM)"]
            S["sbx-server<br/>(static binary, port 49983)"]
        end
    end
    C -- "HTTPS + X-Sandbox-Token" --> P
    P --> S
    S -- "exec / files / procs" --> S
```

This "no new infrastructure" design is the reason a sandbox works in any Docker image and inherits Jobs' billing, hardware flavors, and namespace permissions for free.

### Bootstrapping the server

At job startup, the Job's command is a small `/bin/sh -c` script that fetches the `sbx-server` binary, makes it executable, and `exec`s it. The binary is a ~640KB static [musl](https://musl.libc.org/) build with zero runtime dependencies, so it runs in any `x86_64` Linux image.

A few decisions worth calling out:

- **Download, with a mount fallback.** The fast path downloads the binary from the HF CDN with `wget` or `curl` (every common base image ships one), which is fast and free. As a safety net, the server's Hub repo is also mounted on every job as a volume: if the image has neither `wget` nor `curl`, the script copies the binary off that mount instead. The mount is transparent — it costs nothing unless actually read — but reading it adds ~2-3s to cold start, hence why it's not the default. The only hard requirement on the image is `/bin/sh`.
- **A hand-rolled HTTP/1.1 server, no framework.** Live output streaming requires flushing each chunk as it is produced. Common minimal Rust HTTP servers (e.g. `tiny_http`) buffer chunked responses until the response completes, which breaks streaming. The server therefore implements HTTP/1.1 by hand: NDJSON event streams for `exec`, raw bodies for files, an explicit flush per chunk.
- **Port 49983.** The server listens on a deliberately uncommon port so that the common dev ports (3000, 8000, 8080, …) stay free for your own code.

> [!TIP]
> The server is open source at [github.com/huggingface/sandbox-server](https://github.com/huggingface/sandbox-server).

## Authentication is stateless

Two independent layers protect a sandbox:

1. **The proxy gate.** The Jobs proxy only forwards requests carrying an HF token with read access to the job's namespace. A random member of the internet cannot reach the URL.
2. **The application gate.** `sbx-server` additionally checks an `X-Sandbox-Token` header on every request except `/health`. This is defense in depth: a read-only namespace member who can reach the proxy still cannot execute commands.

The sandbox token is derived, not stored:

```text
nonce  = random 128-bit hex                       # stored in the job label "hf-sandbox-nonce"
token  = HMAC-SHA256(key=your_hf_token, msg="hf-sandbox:" + nonce)
```

The nonce is public and the HMAC key is your HF token, so any machine holding that token can recompute the sandbox token — that is exactly what makes reconnection stateless. It also means the token is a *capability* derived from your credential, not an identity check on the caller: the server verifies that the caller knows the token, not who they are.

Every sandbox job also carries two stable labels for discovery — `hf-sandbox=1` (on all of them) and `hf-sandbox-mode=dedicated` or `hf-sandbox-mode=pool` — so you can list or filter them server-side, e.g. `hf jobs ps --label hf-sandbox=1`.

The token is delivered to the server via a Job secret. The client re-derives it on demand from the public nonce in the label. This has some nice consequences:

- **Stateless reconnection.** [`Sandbox.connect(id)`] works from any machine that holds the same HF token — read the nonce from the label, recompute the token. No local files, no state to copy.
- **The HF token is not passed to the sandbox as an environment variable or job secret** (unless you opt in with `forward_hf_token=True`). This is not a hard guarantee that your credentials stay out of reach: the process listening on the sandbox port is whatever the image starts first, so an untrusted image may be able to observe the requests the client sends — including their `Authorization` header. Treat credentials reachable from a sandbox as potentially exposed to it.

### Token scope

One nonce is minted per **Job**, so the derived token is a *job* credential. In dedicated mode the job is the sandbox, so the two coincide. In pool mode the job is the host, and a host holds many sandboxes — so a pooled sandbox gets a second, narrower credential of its own:

| credential | derived how | authorizes |
| --- | --- | --- |
| **dedicated sandbox token** | `HMAC(hf_token, nonce)` from the job's label | that sandbox (which is the whole job) |
| **pool host token** | `HMAC(hf_token, nonce)` from the host job's label | pool management: create, list and delete sandboxes on that host, and recover their tokens |
| **pooled sandbox token** | random 256 bits, minted by the host server per sandbox | that one sandbox — not a sibling, not the pool |

The client uses the narrow one automatically: `pool.create()` receives it in the create response, `Sandbox.connect("<host>.<id>")` recovers it with the host token, and [`proxy_headers`] hands out the sandbox's token rather than the host's (resolving the HF bearer at the moment you read it, so a long-lived handle does not hand out a stale one) — those headers usually end up in a browser or WebSocket client, so they should confer access to one sandbox and nothing more.

What this means for a leak: a **pooled sandbox token** compromises that sandbox. A **host token** compromises the host — every sandbox on it, current and future, plus its management routes — so treat it as the pool's admin credential. Members of your namespace hold a different HF token and cannot derive yours, but see [Known limitations](#known-limitations) for how a token can be *delivered* to the wrong place.

> [!NOTE]
> During the rollout the host server still accepts the host token on per-sandbox routes, so
> clients that predate per-sandbox tokens keep working. That is a management credential
> reaching the sandboxes it created; a sandbox credential can never reach a sibling.

## Dedicated sandboxes (`Sandbox.create`)

The straightforward model: **one Job per sandbox**. A Job is a real VM, so this gives the strongest isolation (VM-level), supports any hardware flavor including GPUs, and is the right choice for mutually-untrusted code. Its API routes live at `/v1/*` and file paths are absolute on the container filesystem. `kill()` simply cancels the Job.

The boundary a dedicated sandbox gives you is between the **Job and everything outside it**. Inside the VM, your code and `sbx-server` both run as root in the same container: `sbx-server` does not sandbox the command it runs for you, it only exposes it over HTTP. So once code you don't trust has executed in a dedicated sandbox, treat that VM — including its control plane — as belonging to that code: it can kill or replace the server, and anything you do in that sandbox afterwards runs at its mercy. Use one dedicated sandbox per untrusted workload and `kill()` it when done, rather than reusing one across trust boundaries.

```mermaid
sequenceDiagram
    participant U as Sandbox.create()
    participant J as Jobs API
    participant S as sbx-server (in Job)
    U->>J: run_job(image, expose 49983, labels hf-sandbox=1 + hf-sandbox-nonce, secret token)
    J-->>U: job_id + proxy URL
    U->>S: poll /health until ready (~6s VM boot)
    U->>S: POST /v1/exec  (run a command, stream NDJSON back)
    U->>J: cancel_job  (on kill / context-manager exit)
```

The cost is right there in the diagram: every sandbox pays a full ~6s VM cold start and bills a whole machine. For a single sandbox or a GPU workload that is exactly what you want. For 100–1000 short CPU tasks it is wasteful — which is what pools are for.

## Pools: many sandboxes in one Job (`SandboxPool`)

> [!WARNING]
> **Pools are for workloads inside one trust boundary.** A pooled sandbox is a uid + a Landlock
> ruleset inside a shared VM, not a VM of its own, and its host runs a privileged control plane
> that all of its sandboxes talk to. Use pools to fan out *your own* code cheaply. For mutually
> distrusting workloads — anything you would not let read your other sandboxes' files — use
> [`Sandbox.create`], which gives each workload its own VM. See
> [Known limitations](#known-limitations) for the specific gaps.

A typical RL rollout or tool-execution sandbox needs a few MB of RAM and one core for a few seconds. Paying a 2-vCPU VM and a 6s cold start each — and triggering a 1000-VM scheduling burst — is the wrong trade. So [`SandboxPool`] runs one Job as a host and multiplexes many sandboxes inside it.

A pooled sandbox is not a nested VM or container. It is the classic Unix multi-user primitive:

- a **dedicated uid** (≥ 20000),
- a **private `0700` home** owned by that uid,
- commands `exec`'d as that uid with a **scrubbed environment** (`env_clear`, so the host's secrets never leak in), `NO_NEW_PRIVS`, per-process **rlimits**, and a per-sandbox **Landlock ruleset**.

Creating a sandbox is therefore `mkdir + chown + build ruleset` ≈ 1ms server-side — no second VM boot. The only client-visible latency is the proxy round-trip.

```mermaid
flowchart TB
    Pool["SandboxPool<br/>image, flavor, sandboxes_per_host"]
    subgraph h1["Host Job #1 (one VM)"]
        S1["sbx-server"]
        b1["sbx · uid 20001 · ~/ 0700 · landlock"]
        b2["sbx · uid 20002 · ~/ 0700 · landlock"]
        b3["sbx · uid 20003 · ~/ 0700 · landlock"]
    end
    subgraph h2["Host Job #2 (one VM)"]
        S2["sbx-server"]
        b4["sbx · uid 20001 · landlock"]
        b5["sbx · uid 20002 · landlock"]
    end
    Pool --> h1
    Pool --> h2
```

A pooled sandbox's public id is `<host_job_id>.<local_id>`, so `connect`/`exec`/`kill` work statelessly just like dedicated ones. `kill()` on a pooled sandbox sends a `DELETE` to its host (freeing a slot); the host keeps running.

### Isolation in a pool: uid + Landlock

This is the crux of the pool design, so it is worth being precise about what is and isn't isolated.

A stock Job runs as root inside a user namespace that maps only uids 0..65535, with a seccomp filter on and without `CAP_SYS_ADMIN` / `CAP_NET_ADMIN` / `CAP_NET_RAW`. That rules out the usual heavyweight isolation tools: no nested namespaces, no new mounts, no cgroup delegation (`unshare`, `mount`, writing to `/sys/fs/cgroup/...` all fail). What the kernel does offer is [**Landlock**](https://docs.kernel.org/userspace-api/landlock.html) (ABI 6), a Linux Security Module that lets any unprivileged process restrict itself and its children — exactly the per-sandbox boundary we need. For each sandbox the server builds a ruleset; the exec child applies `NO_NEW_PRIVS` → `landlock_restrict_self` → rlimits → `setuid/setgid` before running the command.

Combining distinct uids (discretionary access control) with Landlock is designed and tested to provide:

- ✅ A cannot read another process's `environ`, preventing direct access to HF and sandbox tokens between sandboxes.
- ✅ A cannot `SIGKILL` / `ptrace` / read the memory of B's processes, `setuid` into B, or read B's
  `0700` home.
- ✅ `/tmp` and `/dev/shm` access is denied — each sandbox is Landlock-confined to its own home (its
  `TMPDIR` points inside `$HOME`).
- ✅ A cannot `bind` a TCP port, so there is no inter-sandbox localhost service (outbound `connect`
  stays allowed, so the internet works).
- ✅ Cross-sandbox abstract unix sockets are blocked (`LANDLOCK_SCOPED_ABSTRACT_UNIX_SOCKET`; uid
  isolation alone does *not* block these).

Read that list precisely: it describes what code running **inside** a sandbox can do *directly*. It says nothing about the privileged control plane. `sbx-server` runs as root, outside every sandbox's Landlock domain, and some of its endpoints act on paths and socket names that the sandbox itself controls — so a sandbox that cannot reach its neighbour directly may still be able to get the server to do it on its behalf. Those are tracked in [Known limitations](#known-limitations).

> [!NOTE]
> **Why this is not a substitute for a VM.** Landlock and uid isolation are intended for workloads within the same
> trust boundary. Because pooled sandboxes share a kernel, a VM and a control plane, protection from every
> cross-sandbox attack is not guaranteed; resources and some process-list metadata also remain shared. For mutually
> untrusted code, or for GPU, use [`Sandbox.create`], which gives each sandbox its own VM.

### The file model in a pool

Because a pooled sandbox's only writable area is its Landlock-confined home (which is also its default working directory), the file API roots every path at that home: `files.write("data/in.txt", ...)` writes to `$HOME/data/in.txt`, a leading `/` is taken relative to the home, and `..` cannot escape it. Files written through the API are `chown`ed to the sandbox's uid so the sandbox's own code can read them. This gives a clean "filesystem rooted at the sandbox" model that matches exactly what code inside the sandbox can touch — and differs from dedicated sandboxes, where paths are absolute on the container filesystem.

The `..` guarantee is lexical: path components are normalized before the operation, so no request can *name* a path outside the home. **Symlinks are a different question** — see [Known limitations](#known-limitations). The file API runs as root, so a symlink placed inside the home by the sandbox's own code can currently point the operation at a target outside it.

### Pools have no authoritative local state

A pool is deliberately not a local config file. A pool is its set of running host Jobs, all sharing an `hf-sandbox-pool=<id>` label. This keeps pools consistent with the rest of the sandbox API (everything is discoverable from labels and reattachable from any machine), and it means a pool simply stops existing once its last host is gone.

- A host carries the pool's config (image, flavor, `sandboxes_per_host`, idle timeout) in its job env vars — labels are used only for filtering. When a client must boot a duplicate host, it reads that config back from a running host (`inspect_job`), so all hosts in a pool stay consistent without a central record.
- Env and secrets are per-sandbox, passed at create time — never pool-level. A pooled sandbox's environment is held in the host server's memory for the sandbox's lifetime (it is re-applied to every command you run); it is never written to disk and never appears in any Job's metadata. There is no encrypted-secrets channel for pooled sandboxes — use `env` and treat those values as "not at rest, but not encrypted either". For values that need the encrypted store, use a dedicated sandbox's `secrets`.
- Capacity is server-authoritative. A host refuses creates beyond `sandboxes_per_host` (replying `{"rejected": N}`); the client packs the overflow onto another host or boots a duplicate. This keeps packing exact even when several processes create into the same pool concurrently.
- Idle eviction is two-level. Each sandbox is evicted after its own `idle_timeout` of inactivity (unless it still has a running process); once a host has had no sandboxes for the host idle timeout, it shuts itself down — a billing backstop even if every client disappears.

### A best-effort cache keeps `create --pool` fast

Having no authoritative local state is great for correctness but costs latency. A cold `hf sandbox create --pool <id>` (a fresh CLI process) would otherwise have to rediscover everything over the network before it can create a sandbox: `list_jobs` to scan the namespace → `inspect_job` each host to rebuild its URL and nonce → `GET /v1/sandboxes` to see how full each is → finally `POST` to create. Several round-trips of pure overhead, on every call.

A best-effort cache at `$HF_HOME/sandbox/pools/<context>/<pool-id>.json` removes that. After any create/warm, a process records the pool config plus, per host, its proxy URL, auth nonce, and last-seen free slots. The next process rebuilds the host transport straight from the file (no HTTP) and goes directly to the `POST`.

`<context>` is a digest of the endpoint, the credential and the namespace the entry was written for — the credential as a fingerprint; your token itself is never written to the cache. Both the directory name and the file carry it, and a read that doesn't match on all three is a plain cache miss, so another user of the machine, another endpoint, or a `connect(pool, namespace="other-org")` gets the cold path instead of hosts that were cached for something else. Files are created `0600`, directories `0700`.

```mermaid
flowchart TD
    start["hf sandbox create --pool ID"] --> rc{"cache hit?<br/>pools/&lt;context&gt;/ID.json<br/>(endpoint + credential + namespace)"}
    rc -- "yes, written &lt;15 min ago" --> seed["rebuild host transport<br/>from cached URL + nonce<br/>(no HTTP)"]
    rc -- "yes, older" --> check["inspect_job: still a running host<br/>of this pool, on that URL?"]
    check -- "yes" --> seed
    check -- "no" --> slow
    seed --> post["POST /v1/sandboxes"]
    post -- "ok" --> done["sandbox ready<br/>(~1 round-trip)"]
    post -- "host gone / full" --> slow
    rc -- "no / corrupt / another context" --> slow["fallback: list_jobs +<br/>inspect_job + GET (the cold path)"]
    slow --> post2["POST /v1/sandboxes"] --> done2["sandbox ready<br/>+ refresh cache"]
```

What the cache is and isn't trusted for:

- **Not authoritative on capacity.** The in-job server stays authoritative, so a stale `live` count only ever costs a wasted request, never a mis-packed host.
- **Not trusted for where a host lives.** A cached URL is only used if it is the HTTPS jobs-proxy URL of the job that same entry names, so a cache file cannot choose the destination your HF bearer and sandbox token are sent to; an entry naming anything else is dropped from the file. Past 15 minutes an entry is no longer credited on its own either: `inspect_job` has to confirm the job is still running, still labelled for this pool, and still on that URL before the transport is built.
- **Not trusted for its shape.** Types and ranges are checked on read, so a hand-edited, truncated or stringly-typed file is a cache miss rather than an error part-way through `create()`.
- **Self-healing.** A cached host that is gone is dropped on the first failed request and pruned from the file; the create transparently falls back to label discovery.
- **Concurrency-safe.** Writes merge under a file lock (keyed by `job_id`) and commit atomically, so parallel `create` processes don't clobber each other and readers never see a half-written file.
- **Disposable.** Delete it and you get a cache miss: the cold path runs and everything still works. It is never shared across machines.

Two things to keep in mind. Treat `$HF_HOME` as sensitive: an entry is still a reusable host URL plus the public nonce its token derives from, and while a file written under a different credential is ignored, one written under *yours* is credited for the 15-minute window above. And because the namespace is part of the key, a pool cached under an explicit `namespace=` is only found again when the same `namespace=` is passed — `hf sandbox create --pool <id>` wants the same `--namespace` the pool was created with, or it takes the cold path and looks in your own namespace.

## Known limitations

This section is deliberately exhaustive rather than reassuring: if you are deciding whether a sandbox is a strong enough boundary for a given workload, you need the gaps, not the highlights. Nothing here is a substitute for the rule of thumb — **pools for your own code, dedicated sandboxes for code you don't trust.**

### Threat model at a glance

| Actor | Dedicated | Pool |
| --- | --- | --- |
| Anonymous internet user | Cannot reach the sandbox: the Jobs proxy requires an HF token with namespace read access. | Same. |
| Namespace member with read access | Reaches the proxy, but not the API: they cannot derive your sandbox token. Can see the job exists, its labels, and `/health`. | Same, plus they can read the pool's labels and nonce. |
| Namespace member who can create Jobs | — | Can publish a Job carrying your pool's labels and nonce, but it is not adopted: `adopt_hosts` defaults to hosts this principal started. Relevant again if you opt into `adopt_hosts="namespace"`. |
| Code running in the sandbox | Runs as root in the VM alongside `sbx-server`; owns the VM. | Confined by uid + Landlock as described above, but shares the kernel, VM and control plane with its neighbours. |
| Your own client | Holds the HF token and the sandbox token. | Same; in a pool the sandbox token covers the whole host. |

### Isolation gaps in pool mode

- **The control plane is reachable and privileged.** `sbx-server` runs as root outside every sandbox's Landlock domain, and the sandbox can reach it on loopback. Two of its endpoints act on names the sandbox controls:
  - the **file API** follows symlinks, so a symlink placed in a sandbox's home can direct a root-privileged read, write, `chown` or delete outside that home;
  - the **port proxy** connects to `$SBX_PROXY_DIR/<port>.sock` without rejecting symlinks or checking the socket's owner, so a sandbox can point it at another sandbox's socket.

  Both require a legitimate caller to invoke the endpoint (the sandbox has no token of its own), so they are confused-deputy problems rather than direct escapes — but they do break confidentiality and integrity between pooled sandboxes.
- **Host discovery starts from Job labels**, which any Job creator in the namespace can set — so a label match is a claim, not proof. By default (`adopt_hosts="own"`) the client only adopts hosts *this principal started*, per the Jobs API's `initiator`, and additionally checks the image, flavor, command and exposed URL. Setting `adopt_hosts="namespace"` restores cross-user host sharing and, with it, the ability of any Job creator in that namespace to publish a host your client will send a token to — only use it in a namespace whose members you trust.
- **Landlock can degrade silently.** If Landlock is unavailable, or its ruleset cannot be built, the server currently falls back to uid-only isolation and creates the sandbox anyway — without telling the client. Under uid-only isolation, `/tmp`, `/dev/shm`, TCP bind and cross-home filesystem access are *not* denied. The server also accepts Landlock ABI 1, while the ✅ list above needs ABI 4 (TCP bind) and ABI 6 (abstract sockets); production kernels provide ABI 6, but a lower one would silently drop those two guarantees.
- **Residual shared channels**, none of which Landlock or uid isolation closes: unrestricted outbound TCP; loopback access to the control server; **UDP bind is allowed** (Landlock has no UDP coverage); a sibling's `/proc/<pid>/cmdline` and `status` are readable (`environ` is not — that is the part that would leak credentials, and it is denied); `/proc` and `/sys` are readable and `/dev` is broadly readable and writable; kernel IPC and all machine resources are shared.
- **No CPU, disk, FD or total-memory quotas.** Only per-process `RLIMIT_NPROC` and `RLIMIT_AS` are set; cgroup delegation is not available on Jobs. One sandbox can starve its neighbours. The `max_procs`/`max_mem_mb` values are caller-supplied and not clamped server-side.
- **GPU flavors are untested in pool mode.** The client does not prevent one. Use `Sandbox.create` for GPU.

### Lifecycle and operational gaps

- **A detached descendant can outlive `kill()`.** `SandboxProcess.kill()` signals the command's process group; a descendant that calls `setsid()` leaves it. Deleting the sandbox (pool) or the job (dedicated) does terminate everything. Use `timeout=` if you need a hard bound.
- **`SandboxProcess.kill()` does not currently stop the process** — the client sends the OS pid where the server expects its own opaque process id, and the server answers `200` either way. Until this is fixed, stop background work by deleting the sandbox.
- **A long foreground command can trip the idle watchdog.** `idle_timeout` counts API requests, and a running foreground command is not counted as activity, so a command that runs longer than `idle_timeout` without other API traffic can have its sandbox shut down under it. Raise `idle_timeout` (or pass `None`) for long single commands.
- **`max_hosts` is a per-process cap.** Two processes using the same pool each count only their own hosts, so the global number of host Jobs can exceed it. Per-host `sandboxes_per_host` *is* enforced server-side.
- **`close()` on a pool may cancel hosts it did not create.** A pool handle that discovered a warm host started by another process (or another user in the same namespace) will cancel it on exit, killing that host's sandboxes. Use `SandboxPool.connect()` for handles that must not tear hosts down, and prefer distinct `name=` values for independent pools.
- **A failed teardown is reported as success.** `close()` logs a warning and returns normally if it cannot cancel a host job, and removes its cache entry — so a host can keep running and billing. Check `hf jobs ps --label hf-sandbox=1` if teardown matters to you.
- **The server binary is not pinned or verified.** Each job downloads `sbx-server` from a mutable public bucket path and executes it as root without checking a digest or signature.
- **The server's HTTP front end is minimal.** No connection cap and no read deadlines, so a slow-request flood can exhaust its threads; a request with a malformed `Content-Length` is treated as having no body. Reaching it still requires passing the proxy gate.

### Also worth knowing

- `/health` does not require the sandbox token and reports the server version, uptime and sandbox count to anyone who can pass the proxy gate.
- Sandbox ids come from `/dev/urandom`, with a timestamp fallback if that read fails — predictable ids in that (very unlikely) case.
- Pooled uids are allocated monotonically from 20000 and never recycled, so a host that has created ~45,000 sandboxes over its lifetime can no longer create more, even when empty.

## Performance

All numbers are measured against real HF Jobs on `cpu-basic`, with the client on a laptop and all traffic flowing through the Jobs proxy.

**Dedicated sandbox:**

| metric                                            | value                                                           |
| ------------------------------------------------- | --------------------------------------------------------------- |
| cold start (`create()` returns, server answering) | ~5.8s median                                                    |
| `run()` round-trip                                | p50 ~110ms (the proxy RTT floor is ~105ms; client overhead ≈ 0) |
| file transfer (parallel ranged, >8 MiB)           | ~340 MiB/s down, ~441 MiB/s up                                  |

**Pool (shared/host mode):**

| N sandboxes | hosts (50/host) | provision + create all | exec in all | kill all | total     |
| ----------- | --------------- | ---------------------- | ----------- | -------- | --------- |
| 100         | 2               | 6.1s                   | 1.5s        | 0.6s     | **8.2s**  |
| 1000        | 20              | 7.4s                   | 4.2s        | 4.2s     | **15.8s** |

1000 sandboxes created, exec'd and killed in ~16s cost roughly one host cold start (~6s) amortized across all of them — about **$0.0009** total (20 × `cpu-basic`), versus ~$0.06 and a 1000-VM scheduling burst for one Job per sandbox. Server-side create/exec/delete are each ~1ms; the budget is entirely the network round-trip.

## Design decisions, recap

| decision                                                              | why                                                                             |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Build on Jobs, no new service                                         | inherits billing, hardware, permissions; works in any image                     |
| Static Rust binary, downloaded at startup                             | no Python/pip; ~6s cold start vs 30–90s for a pip-based bootstrap               |
| Hand-rolled HTTP/1.1                                                  | minimal frameworks buffer chunked responses and break live streaming (verified) |
| Stateless HMAC auth                                                   | reconnect from anywhere; a derived token instead of the HF token itself (scoped per job — see [Token scope](#token-scope)) |
| `run()` raises on non-zero exit (`check=False` opts out)              | best DX for "run code, see the error" loops (E2B-style)                         |
| `idle_timeout` watchdog instead of client-side cleanup                | persistent sandboxes are a feature; leaked ones still die                       |
| Pools = uid + Landlock, server-authoritative capacity, no local state | fast same-user fan-out; correct under concurrency; reattachable anywhere        |
