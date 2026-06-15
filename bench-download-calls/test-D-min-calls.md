# Test D — minimal hub calls (`bench-d-min-calls`)

Branch: [`bench-d-min-calls`](https://github.com/huggingface/huggingface_hub/tree/bench-d-min-calls) (no PR opened). Builds on top of Test C.

## Results

Totals (cold-v1 / warm / PR-cold / warm): **small 4 / 1 / 4 / 1** · **large 6 / 1 / 6 / 1**.

### Small repo (201 files)

| Run | Calls | Breakdown |
|-----|-------|-----------|
| run1 v1 cold | **4** | 1 revision + 1 tree + 1 git-upload-pack + 1 xet-token |
| run2 v1 warm | **1** | 1 revision |
| run3 PR cold | **4** | 1 revision + 1 tree + 1 git-upload-pack + 1 xet-token |
| run4 PR warm | **1** | 1 revision |

### Large repo (2001 files)

| Run | Calls | Breakdown |
|-----|-------|-----------|
| run1 v1 cold | **6** | 1 revision + 3 tree (paginated) + 1 git-upload-pack + 1 xet-token |
| run2 v1 warm | **1** | 1 revision |
| run3 PR cold | **6** | 1 revision + 3 tree + 1 git-upload-pack + 1 xet-token |
| run4 PR warm | **1** | 1 revision |

Cold-run hub calls are **O(1) in file count**, the only growth being tree pagination (`ceil(N/1000)` pages): small 504 → 4 (−99.2%), large 5007 → 6 (−99.9%). One `git-upload-pack` fetches all regular blobs and one xet group fetches all xet blobs regardless of count. Wall-clock cold also dropped (small 18s→6s, large 230s→46s), despite time not being a goal.

## What it does on top of C

### 1. All regular (non-LFS) files in ONE request — git smart HTTP

The Hub's git server supports protocol v2 with arbitrary **blob oids in `want` lines** (discovered empirically; capability line: `fetch=shallow wait-for-done filter`). `snapshot_download` collects the blob oids of all missing regular files (already known from the tree listing) and POSTs a single `fetch` to `/{repo}.git/git-upload-pack`. The returned packfile is parsed in pure Python (~200 lines: pkt-line/side-band demux, varint headers, zlib objects, OFS/REF delta support), each blob is **verified against its git sha1**, and written directly to `blobs/` — the per-file loop then only creates symlinks, zero HTTP.

- Small repo: 101 files in one ~80 KB response, replacing 202 GETs. Large repo: all 1000 regular blobs in a single request (one POST regardless of count).
- In run3, only the changed text blobs are requested (50 small / 500 large) — the tree cache + blob presence check make the request minimal.
- Threshold: used when ≥ 4 blobs are missing (below that, per-file GETs are CDN-cacheable and simpler). Falls back to the normal per-file path on *any* error (older deployments, parse failure, missing blob).
- Bearer-token auth on the git endpoint worked fine (sent; repo is public — private-repo auth not exercised).

### 2. All xet files in ONE download group — single token fetch

`hf_hub_download` creates one hf_xet download group per file, and **hf_xet fetches a fresh read token per group** → 100 × `GET /api/…/xet-read-token` on huggingface.co in the baseline (these are "ignorable" per scope but still hit the hub). D registers all missing xet blobs in a single `XetSession.new_file_download_group()` → exactly **1 token call**, parallelism handled inside hf_xet. Blobs are downloaded to `blobs/<sha256>` directly (sha256 spot-verified) and symlinked by the normal loop.

## Decisions & trade-offs

- **`repo_info` kept**: the only single-call way to resolve a ref to a commit hash (`/tree` doesn't return it, `ls-refs` would cost 2 git calls). Warm floor stays 1 call — unavoidable as long as refs are mutable and must be revalidated.
- **`paths-info` not used**: `POST /api/{type}s/{repo}/paths-info/{rev}` (max 2000 paths, returns oid/size/lfs/xetHash) would be ideal for *partial* downloads with `allow_patterns`, but for full snapshots `/tree` is equivalent at 1 call ≤1000 files.
- **Request count vs request weight**: one `git-upload-pack` is heavier server-side (gitaly pack generation) than one resolve GET — but vastly lighter than 202 of them. If git-protocol use from the library is unwanted, the same shape could be a dedicated batch-content endpoint (e.g. POST paths → multipart/zip), which doesn't exist today.
- Prefetches are skipped for `dry_run` and `local_dir` modes (the latter falls back to C behavior: metadata available, no HEAD, per-file GETs).
- `snapshot_download`'s signature is unchanged; per-file progress bars for prefetched files are bypassed (bytes shown once at symlink time) — cosmetic regression to fix in a real PR.

## Caveats

- Same 3 offline-simulation test failures as C (split-brain mock; see test C report). 10/13 pass.
- Pack fetch of a very large text corpus buffers the response in memory; a streaming pack parser (or size cap + fallback) is needed before productionizing.
- Failed group downloads leave `*.incomplete`/`*.pack.tmp` files behind (cleaned on next successful run); acceptable for an experiment.
