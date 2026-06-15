# Test B — PR #4348 (`reuse-tree-data-in-snapshot`)

The PR builds per-file metadata (`etag`, `size`, `xetHash`) from `list_repo_tree` and passes it to a new internal `_hf_hub_download`, skipping the per-file HEAD. **But the tree path is gated behind `unreliable_nb_files`** (siblings missing/empty or > `LARGE_REPO_THRESHOLD` = 1000). So the PR does nothing below 1000 files and activates above it — the two repos show exactly that.

Totals (cold-v1 / warm / PR-cold / warm): **small 504 / 1 / 403 / 1 (inert)** · **large 3006 / 4 / 1504 / 4 (active)**.

## Small repo (201 files) — gate never triggers

### B as-is: 504 / 1 / 403 / 1 — identical to main

201 siblings ≤ 1000, so the optimization stays dormant. It only helps kernel repos (no siblings) and >1000-file repos — by design, but worth stating: on the typical repo the PR changes **nothing**.

### B-forced (gate lowered to 100): 304 / 2 / 152 / 2

To show the intended effect at small scale, the threshold was lowered:

| Run | Calls | Breakdown |
|-----|-------|-----------|
| run1 v1 cold | 304 | 1 revision + 1 tree + 101 GET resolve (307) + 101 GET resolve-cache (200) + 100 xet-token* |
| run2 v1 warm | **2** | 1 revision + **1 tree (re-listed every warm run)** |
| run3 PR cold | 152 | 1 revision + 1 tree + 50 GET (307) + 50 GET (200) + 50 xet-token* |
| run4 PR warm | **2** | 1 revision + 1 tree |

## Large repo (2001 files) — gate triggers, optimization active

| Run | Calls | Breakdown |
|-----|-------|-----------|
| run1 v1 cold | **3006** | 1 revision + 3 tree + 1001 GET resolve (307) + 1001 GET resolve-cache (200) + 1000 xet-token* |
| run2 v1 warm | **4** | 1 revision + **3 tree (paginated, re-listed every warm run)** |
| run3 PR cold | **1504** | 1 revision + 3 tree + 500 GET (307) + 500 GET (200) + 500 xet-token* |
| run4 PR warm | **4** | 1 revision + 3 tree |

\* hf_xet-issued, ignorable per scope.

Cold drops **5007 → 3006 (−40%)** vs main: the PR removes exactly the ~3000 per-file HEAD round-trips (regular 307+200 follow-up, xet 302). It is **identical to Test C on every cold run** — both use the same tree-metadata path; the only difference shows up warm (below).

## Observations

- When active, the metadata construction is correct (etag = `lfs.sha256` or `blob_id`, xet data from `xetHash` + reconstructed refresh route — verified against actual `/resolve` etags and cache blob names) and removes all per-file HEADs.
- **Warm-run regression at both scales**: nothing is persisted, so every warm call re-lists the tree (small +1, large +3 paginated pages → 1 vs 2/4 calls). Test C fixes this with the on-disk `trees/` cache.
- Regular-file downloads still cost 2 GETs each (resolve 307 → resolve-cache); Test E removes that hop.
- Tree is listed at the resolved `repo_info.sha` (good — consistent listing) and without `expand` (good — 1000 items/page; `expand=true` drops the page size to 50).
- Caveat shared with C/D/E: with pre-fetched metadata, a download failure surfaces as a raw connection error instead of the HEAD-fallback path (`LocalEntryNotFoundError`) — only in split-brain situations (API reachable, downloads failing). The offline-simulation tests catch this now that the tree path activates.

**Verdict**: right direction, two gaps — (1) gated to >1000-file repos, (2) tree re-fetched on every call including warm ones. C generalizes it (any size) and caches the listing.
