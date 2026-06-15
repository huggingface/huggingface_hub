# Methodology — snapshot_download HTTP call benchmark

**Dates**: 2026-06-12 (small repo) · 2026-06-13 (large repo) · **Base**: `huggingface_hub` 1.20.0.dev0 (main @ `ca9284561`) · **hf-xet**: 1.4.3 · **Python**: 3.10

## Two test repositories

Every variant is benchmarked on **two repos of identical shape but 10× different size**. The second one deliberately crosses `LARGE_REPO_THRESHOLD = 1000`, which changes the client's behavior (see reports).

### Small — [`Wauplin/snapshot-download-bench`](https://huggingface.co/Wauplin/snapshot-download-bench) — 201 files (~105 MB)

- 100 regular text files (`file_000.txt` …), each a unique UUID (~37 B), stored in git.
- 100 binary files (`bin_000.bin` …), each 1 MiB of unique random bytes, stored as LFS/Xet (all have a `xetHash`).
- +1 `.gitattributes`.
- Tag **`v1`** = commit `668054422…` (all 200 files in one commit).
- PR **`refs/pr/1`** = commit `89f58185d…` updating 50 text + 50 binary files in place (100 differ from v1, 100 identical).
- Validation per run: **201 files, 104,862,819 bytes**.

### Large — [`Wauplin/snapshot-download-bench-2k`](https://huggingface.co/Wauplin/snapshot-download-bench-2k) — 2001 files (~1.05 GB)

- 1000 regular text files + 1000 binary (1 MiB) xet files + `.gitattributes`.
- Tag **`v1`** = commit `389cf1edd…`; PR **`refs/pr/1`** = commit `b1254dd11…` updating 500 text + 500 binary in place.
- Validation per run: **2001 files, 1,048,614,519 bytes**.

## Measurement

All Python HTTP traffic is routed through a local mitmproxy that logs every request (method, host, path, status, redirect). Per the agreed scope, **only Python-side calls are measured**: hf_xet's data-plane traffic (CAS reconstructions, CDN xorb ranges) bypasses the proxy via `NO_PROXY=.xethub.hf.co,.cdn.hf.co` and is not counted.

One exception is shown in the tables: `GET /api/models/…/xet-read-token/{rev}` is issued by hf_xet but **lands on huggingface.co**. It is reported on its own line (and excluded from "excl. token" totals) so both views are available.

## Protocol (identical for every variant, both repos)

Fresh empty cache directory per variant, then 4 sequential runs against it:

1. **run1** — `snapshot_download(repo, revision="v1")` (cold cache)
2. **run2** — same call again (warm)
3. **run3** — `snapshot_download(repo, revision="refs/pr/1")` (half the files new, half reusable from v1)
4. **run4** — same call again (warm)

Content was additionally spot-checked by sha256 (xet files) and git-sha1 (pack-fetched files) in smoke tests.

Variants are run from git worktrees via `PYTHONPATH` override:

- **A** = `main` (baseline)
- **B** = PR [#4348](https://github.com/huggingface/huggingface_hub/pull/4348) (`reuse-tree-data-in-snapshot`); on the small repo also run with its `LARGE_REPO_THRESHOLD` gate lowered ("B-forced") since the gate otherwise keeps it inert below 1000 files
- **C** = branch [`bench-c-tree-cache`](https://github.com/huggingface/huggingface_hub/tree/bench-c-tree-cache)
- **D** = branch [`bench-d-min-calls`](https://github.com/huggingface/huggingface_hub/tree/bench-d-min-calls)
- **E** = branch [`bench-e-no-git`](https://github.com/huggingface/huggingface_hub/tree/bench-e-no-git) (added after review: same goal as D but without git endpoints)

Harness & raw captures: `~/projects/hub-download-bench/` — `scripts/` (parametrized via `BENCH_REPO`/`BENCH_N`/`BENCH_N_UPDATED`; `run_all_2k.sh` driver), `captures/<variant>[-2k]/run*.jsonl`.
