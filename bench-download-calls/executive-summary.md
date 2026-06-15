# Executive summary — what does `snapshot_download` really cost the Hub?

**Setup.** Two controlled repos — a 201-file (100 git + 100 xet/LFS) and a 2001-file (1000 + 1000) — with every HTTP call from the Python client to huggingface.co measured via an intercepting proxy, across 4 scenarios each: cold download, warm re-run, download of a PR updating half the files, warm re-run. Five client variants compared. hf_xet's data-plane (CAS/CDN) is excluded — it doesn't touch hub rate limits.

**Today (v1.20 main): a cold snapshot costs ~2.5 calls to huggingface.co per file** — 504 calls for 201 files, 5007 for 2001. The cost scales with file count, not changed bytes: each file pays a HEAD metadata round-trip (doubled by the public-repo resolve-cache redirect) even when it's already cached, so re-downloading a PR where half the files are unchanged barely helps (403 / 4006).

**Almost all of this is avoidable without any server change**, because one `/tree` API call already returns every file's etag, size and xet hash, and a commit's tree is immutable so it can be cached on disk forever. The same five variants on both repos (cold / warm / PR-cold / PR-warm):

| Variant | 201-file repo | 2001-file repo |
|---|---|---|
| A. main (baseline) | 504 / 1 / 403 / 1 | 5007 / 4 / 4006 / 4 |
| B. community PR #4348 | 504 / 1 / 403 / 1 *(inert ≤1000 files)* | 3006 / 4 / 1504 / 4 |
| C. tree listing + on-disk cache, no per-file HEAD | 304 / 1 / 152 / 1 | 3006 / **1** / 1504 / **1** |
| E. C + redirect-free downloads + 1 xet token (plain HTTP) | 104 / 1 / 53 / 1 | 1006 / 1 / 505 / 1 |
| D. C + batched content fetches (git protocol) | **4 / 1 / 4 / 1** | **6 / 1 / 6 / 1** |

**C is the recommended short-term change** (moderate diff, caches stay compatible): always list the tree once per commit, cache it under the existing layout (`trees/`), skip every per-file HEAD. PR #4348 validates the same idea but **only activates above 1000 files** and re-fetches the listing on every call — including warm ones. The 2001-file column makes both gaps visible: B and C are identical on cold, but C's cache keeps warm pulls at 1 call where B (and main) cost 4.

**A scale trap worth flagging on its own**: above 1000 files, today's client lists the repo tree on *every* call without caching it, so even a **no-op warm re-pull costs 4 calls** (a paginated 3-page listing). C removes this; it is invisible below 1000 files.

**E is the floor using only regular HTTP endpoints (−79% to −80% cold)**: every public-repo download today pays a systematic 307 redirect to an internal resolve-cache URL the client can build itself from tree data — so each file costs one clean 200. It also fixes a hidden hub load: **hf_xet requests a read token per file** (100/1000 calls per cold run in A–C); batching all files into one xet download group reduces that to 1. Needs the server to bless the resolve-cache route (or return direct URLs in the tree listing).

**D shows the absolute floor with today's server: a full snapshot in 4–6 calls (−99%+) regardless of size** — ref resolution, tree listing, one git-protocol POST fetching all small files as a verified pack, one xet token. The git-protocol fetch trades many cheap requests for one heavier one — a deliberate call (sanction it, or add a batch-content endpoint to get D's numbers with E's cleanliness).

**Rate-limit implication.** Identical user work costs anywhere from 4 to 5007 requests depending on client version and path. Raw request counts are therefore a poor billing unit for downloads; weight by route class (metadata vs resolve vs git), and expect headline `/resolve` volumes to drop dramatically as tree-based clients roll out.

Branches `bench-c-tree-cache`, `bench-d-min-calls` and `bench-e-no-git` pushed (no PRs). Full details in `combined-report.md`, the per-test reports (each now covering both repos), and `00-methodology.md`.
