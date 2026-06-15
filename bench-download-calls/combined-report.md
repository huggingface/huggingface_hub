# snapshot_download HTTP call count — combined results

Scope: Python-side calls to huggingface.co only (hf_xet CAS/CDN data-plane excluded by agreement). Two repos of identical shape, 10× apart in size, each with tag `v1` and a PR updating half the files in place (see `00-methodology.md`):

- **Small** — 201 files (100 git-text + 100 xet-LFS + .gitattributes), PR updates 50+50.
- **Large** — 2001 files (1000 + 1000 + .gitattributes), PR updates 500+500. Crosses `LARGE_REPO_THRESHOLD = 1000`.

## Headline numbers (every run downloads/validates all files)

**Total calls to huggingface.co** (incl. the hf_xet-issued `xet-read-token`):

### Small repo (201 files)

| Run | A `main` | B #4348 as-is | B forced* | C tree-cache | E no-git** | D git-pack |
|---|---|---|---|---|---|---|
| 1. v1 cold | 504 | 504 | 304 | 304 | 104 | **4** |
| 2. v1 warm | 1 | 1 | 2 | **1** | **1** | **1** |
| 3. PR cold (50+50 changed) | 403 | 403 | 152 | 152 | 53 | **4** |
| 4. PR warm | 1 | 1 | 2 | **1** | **1** | **1** |

### Large repo (2001 files)

| Run | A `main` | B #4348 | C tree-cache | E no-git | D git-pack |
|---|---|---|---|---|---|
| 1. v1 cold | 5007 | 3006 | 3006 | 1006 | **6** |
| 2. v1 warm | 4 | 4 | **1** | **1** | **1** |
| 3. PR cold (500+500 changed) | 4006 | 1504 | 1504 | 505 | **6** |
| 4. PR warm | 4 | 4 | **1** | **1** | **1** |

**Excluding `xet-read-token`** (per agreed scope), cold runs:

| | A | B (active) | C | E | D |
|---|---|---|---|---|---|
| small v1 cold | 404 | 204 | 204 | 103 | **3** |
| small PR cold | 353 | 102 | 102 | 52 | **3** |
| large v1 cold | 4007 | 2006 | 2006 | 1005 | **5** |
| large PR cold | 3506 | 1004 | 1004 | 504 | **5** |

\* On the small repo B's tree path is inert (201 ≤ 1000 siblings) so it equals main; "forced" = gate lowered to show its potential. On the large repo it activates automatically (no forcing).
\** E added after review feedback: same goal as D but restricted to regular HTTP endpoints (no git protocol).

## Where the calls go (cold v1)

| Call | A | B / C | E | D |
|---|---|---|---|---|
| `GET /api/models/{repo}/revision/{rev}` (ref→sha) | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| `GET /api/models/{repo}/tree/{sha}` (paginated `ceil(N/1000)`) | 0 / 3 | 1 / 3 | 1 / 3 | 1 / 3 |
| `HEAD /resolve` → 307 + follow-up (regular) | 202 / 2002 | 0 | 0 | 0 |
| `HEAD /resolve` → 302 (xet) | 100 / 1000 | 0 | 0 | 0 |
| `GET` content, regular (307→resolve-cache, or direct) | 101 / 1001 † | 202 / 2002 | **101 / 1001** | 0 |
| `POST /{repo}.git/git-upload-pack` (all regular blobs, 1 pack) | – | – | – | 1 / 1 |
| `GET /api/…/xet-read-token` (hf_xet) | 100 / 1000 | 100 / 1000 | **1 / 1** | **1 / 1** |

Cells show `small / large`. On the small repo, main uses `siblings` (no tree call); above 1000 files it lists the tree like everyone else.
† On main the content GET goes straight to the resolve-cache URL learned from the preceding HEAD redirect (1 GET); without that HEAD (B/C) each GET pays its own 307 (2 GETs). Net effect identical.

## Key findings

1. **Main's cost is ~2.5 hub calls per file, linear in repo size, even when nothing changed.** Confirmed across 10× (504→5007 cold). In the PR-update runs the *unchanged* half still costs ~2 calls each: per-file HEAD metadata is fetched before the cache can declare a hit.
2. **The per-file HEAD is fully redundant with `/tree`.** etag (`lfs.sha256` | `blob_id`), size, and `xetHash` from the tree listing are byte-identical to what `/resolve` HEAD returns (verified against live responses and resulting `blobs/` layout). PR #4348 got this right.
3. **PR #4348 only helps above 1000 files, and re-lists the tree on every call.** Below the threshold it is inert (identical to main); above it, cold drops 5007→3006 (−40%) — but warm runs still pay the (now paginated, 3-page) listing: **4 calls** where they should be 1.
4. **A commit's tree is immutable → cache it on disk (Test C).** `trees/<commit>.json` makes warm runs **1 call at any size** (vs B's 2 small / 4 large), removes the threshold heuristic (same path everywhere), and gives `allow_patterns` filtering + future snapshot-completeness checks for free. C equals B on cold; its entire edge is the disk cache.
5. **Tree pagination is the only O(N) term left for C/D/E** — `ceil(N_files/1000)` calls (3 at 2001 files) — but they pay it **once per commit** (cached); A and B pay it on **every** call.
6. **`xet-read-token` is fetched once per FILE, not per session** (hf_xet creates one download group per file). 100/1000 hub calls per cold run in A/B/C. Declared ignorable for counting, but it's the single biggest remaining hub-call source after C — trivially fixed by batching all files into one download group (D/E do this: 1 call).
7. **Without git endpoints, the floor is `revision + ceil(N/1000) tree + files-to-download + 1 token`** (Test E: small 104, large 1006). The public-repo `/resolve` 307 target (`/api/resolve-cache/{type}s/{repo}/{commit}/{file}?…&etag=…`) is deterministic from tree data, so the redirect hop is skipped — every download a single clean 200. Caveat: resolve-cache is an internal route; shipping needs the server to bless it (or return the direct URL in `/tree`/`paths-info`), with `/resolve` as fallback.
8. **O(1) cold downloads are possible with today's server (Test D: small 4, large 6):** git smart-HTTP v2 accepts arbitrary blob `want`s → all regular files in one POST (sha1-verified pack, ~200-line parser, graceful fallback); one xet download group → one token; `/tree` without `expand` returns everything at 1000/page (`expand=true` silently drops the page to 50 — avoid). The only growth with size is tree pages.
9. **Server-side observations** for the rate-limit discussion:
   - the public-repo `307 → /api/resolve-cache` redirect doubles every regular-file HEAD/GET — a client calling `/resolve` n times generates 2n requests;
   - `POST /api/{type}s/{repo}/paths-info/{rev}` (2000 paths/call, returns oid/size/lfs/xetHash) is the natural batch-metadata endpoint for partial downloads — unused by the library today;
   - no batch endpoint exists for *content* of small files; D works around it via git-upload-pack (heavier per request for gitaly, but replaces hundreds/thousands of requests). A sanctioned batch-content endpoint would remove that trade-off.

## Recommendations

1. **Short term**: land C's approach — always-on tree listing at the pinned commit + `trees/` disk cache + per-file HEAD skip. Moderate, self-contained diff; biggest win per unit of risk. PR #4348 is a step toward it — un-gate it (the 2k run shows it only acts above 1000 files), list at `repo_info.sha`, add the disk cache, fix the warm-run re-listing.
2. **Short term (hub load)**: batch xet downloads into one group per `snapshot_download` (or fix token caching across groups in hf_xet) — eliminates ~1 hub call per LFS file (1000 → 1 on the large repo).
3. **Short/medium term (no git)**: bless direct resolve-cache downloads (Test E) — same server load as today's redirected traffic, halves regular-file requests. Cleanest form: have `/tree`/`paths-info` return the direct download URL so clients don't hardcode the route.
4. **Medium term**: decide whether git-upload-pack blob fetch is a sanctioned client path; if not, consider a batch content endpoint. Either turns cold snapshots into O(1) hub calls.
5. **Before merging C/D/E**: make `snapshot_download` degrade gracefully when a download fails after metadata prefetch (3 offline-simulation tests currently fail on a split-brain scenario: API reachable, downloads not), and fix per-file progress reporting for prefetched files.
6. **Rate-limit design**: identical user work costs 4–5007 hub calls depending on client version/path. Raw request counts are a poor billing unit for downloads; weight by route class (metadata vs resolve vs git), and expect `/resolve` volumes to fall sharply as tree-based clients roll out. Note the scale trap: above 1000 files even today's *no-op* warm re-pull costs 4 calls (uncached paginated listing).
