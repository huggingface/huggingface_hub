# Test E — minimal hub calls, no git endpoints (`bench-e-no-git`)

Branch: [`bench-e-no-git`](https://github.com/huggingface/huggingface_hub/tree/bench-e-no-git) (no PR opened). Builds on Test C; answers "how low can we go using only regular HTTP endpoints?" after Test D was (rightly) called out for leaning on the git smart-HTTP protocol.

## Results

Totals (cold-v1 / warm / PR-cold / warm): **small 104 / 1 / 53 / 1** · **large 1006 / 1 / 505 / 1**.

### Small repo (201 files)

| Run | Calls | Breakdown |
|-----|-------|-----------|
| run1 v1 cold | **104** | 1 revision + 1 tree + 101 GET resolve-cache (one per regular file) + 1 xet-token |
| run2 v1 warm | **1** | 1 revision |
| run3 PR cold | **53** | 1 revision + 1 tree + 50 GET resolve-cache + 1 xet-token |
| run4 PR warm | **1** | 1 revision |

### Large repo (2001 files)

| Run | Calls | Breakdown |
|-----|-------|-----------|
| run1 v1 cold | **1006** | 1 revision + 3 tree (paginated) + 1001 GET resolve-cache + 1 xet-token |
| run2 v1 warm | **1** | 1 revision |
| run3 PR cold | **505** | 1 revision + 3 tree + 500 GET resolve-cache + 1 xet-token |
| run4 PR warm | **1** | 1 revision |

Every call is a clean 200 — zero redirects, zero HEADs. vs main: cold −79% (small) / −80% (large), PR update −87% / −87%. Excl. tokens, v1 cold: small 404→103, large 4007→1005; PR cold: small 353→52, large 3506→504. Cold cost is `revision + ceil(N/1000) tree pages + files_to_download + 1 token`.

## What it does on top of C

### 1. Regular files: skip the systematic 307 redirect (2 GETs → 1 GET)

On public repos, `GET /resolve/...` *always* answers 307 → `/api/resolve-cache/{type}s/{repo}/{commit}/{file}?{resolve-path}=&etag="{etag}"`. That target is fully deterministic — commit hash, path and etag are all in the tree listing — so E builds it client-side and downloads directly. Verified: same content, same `ETag`/`X-Repo-Commit` response headers as the redirected flow; the exact query string the server's 307 would produce is included (it looks like a CDN cache key, so behaving identically to redirected clients matters).

Guards: private repos keep the `/resolve` URL (they serve content directly, no redirect), and LFS/xet files are untouched (signed CDN redirects can't be built client-side).

### 2. Xet files: one download group → one token (same as D)

All missing xet blobs are registered in a single `XetSession` download group instead of one group per file: 1 × `xet-read-token` instead of 100.

## Decisions & trade-offs

- **`/api/resolve-cache` is an internal, undocumented endpoint.** Today it's where 100% of public-repo download traffic already lands (every resolve call is redirected there), so the load profile is identical — we only remove the redirect hop. But the URL shape is not a public contract; shipping this requires either (a) moon-landing blessing the route as stable, or (b) the server including the direct URL in the `/tree` / `paths-info` response so clients don't hardcode it. Until then it's redirect-shaped coupling, with the plain `/resolve` URL as the natural fallback if the route ever changes (a 404 would surface loudly — a production version should catch and retry via `/resolve`).
- **Per-file content GETs are the irreducible floor for REST**: without a batch-content endpoint (or git packs, see Test D), N regular files cost N GETs. E's cold cost is `3 + regular_files_to_download` (+1 tree page per 1000 files); the gap to D's O(1) is exactly the batch-content capability.
- Same caveats as C/D: 3/13 offline-simulation tests fail (split-brain mock artifact — see Test C report; 10 pass), and prefetched xet files bypass the per-file progress bar.

## E vs D

| | E (no git) | D (git pack) |
|---|---|---|
| v1 cold — small / large | 104 / 1006 | 4 / 6 |
| PR cold — small / large | 53 / 505 | 4 / 6 |
| Scaling | O(files downloaded) | O(1) |
| Endpoints used | revision, tree, resolve-cache, xet-token | revision, tree, **git-upload-pack**, xet-token |
| Server cost profile | same as today's redirected traffic | shifts small-file load to gitaly |
| Coupling risk | resolve-cache URL shape (internal) | git protocol (stable, but unconventional for the lib) |
