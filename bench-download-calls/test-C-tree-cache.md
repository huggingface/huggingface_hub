# Test C — tree listing cached on disk (`bench-c-tree-cache`)

Branch: [`bench-c-tree-cache`](https://github.com/huggingface/huggingface_hub/tree/bench-c-tree-cache) (no PR opened).

## Results

Totals (cold-v1 / warm / PR-cold / warm): **small 304 / 1 / 152 / 1** · **large 3006 / 1 / 1504 / 1**.

### Small repo (201 files)

| Run | Calls | Breakdown |
|-----|-------|-----------|
| run1 v1 cold | 304 | 1 revision + 1 tree + 101 GET resolve (307) + 101 GET resolve-cache (200) + 100 xet-token* |
| run2 v1 warm | **1** | 1 revision (tree read from disk cache) |
| run3 PR cold | 152 | 1 revision + 1 tree + 50 GET (307) + 50 GET (200) + 50 xet-token* |
| run4 PR warm | **1** | 1 revision |

### Large repo (2001 files)

| Run | Calls | Breakdown |
|-----|-------|-----------|
| run1 v1 cold | 3006 | 1 revision + 3 tree (paginated) + 1001 GET resolve (307) + 1001 GET resolve-cache (200) + 1000 xet-token* |
| run2 v1 warm | **1** | 1 revision (tree read from disk cache) |
| run3 PR cold | 1504 | 1 revision + 3 tree + 500 GET (307) + 500 GET (200) + 500 xet-token* |
| run4 PR warm | **1** | 1 revision |

\* hf_xet-issued, ignorable per scope. vs main: cold −40% (both scales), PR-update −62%/−62%; excl. tokens: small 404→204, large 4007→2006.

**C vs B**: identical on every cold run (same tree-metadata path) — C's value is the **on-disk cache**, so warm/incremental pulls are **1 call** where B re-lists the tree (2 calls small, **4 large**). The gap grows with repo size because the listing is paginated (`ceil(N/1000)` calls), and C also drops the `LARGE_REPO_THRESHOLD` heuristic so the same code path runs at any size.

## Design decisions

1. **`list_repo_tree` on every snapshot_download** (as requested), replacing the `siblings`-based listing entirely — no more `LARGE_REPO_THRESHOLD` heuristic. Listed at the resolved `repo_info.sha`, `recursive=true`, **without `expand`** (1000 items/page vs 50 with expand; lfs oid/size and xetHash are included anyway — verified).
2. **New cache folder `trees/`**, sibling of `refs/ blobs/ snapshots/`: one `trees/<commit_hash>.json` per commit. A commit's tree is immutable → cache forever, no invalidation problem. Format is a human-readable JSON index, files sorted, keyed by path for direct lookup:
   ```json
   {
     "format_version": 1,
     "repo_id": "Wauplin/snapshot-download-bench",
     "repo_type": "model",
     "commit_hash": "66805442…",
     "files": {
       "bin_000.bin": {"size": 1048576, "blob_id": "0875fae8…", "lfs_sha256": "6e0065d2…", "lfs_size": 1048576, "xet_hash": "c1404b89…"},
       "file_000.txt": {"size": 37, "blob_id": "536fa2db…"}
     }
   }
   ```
3. **Per-file HEAD calls eliminated**: an `HfFileMetadata` is built from each tree entry and threaded through `hf_hub_download → _get_metadata_or_catch_error` (new optional `file_metadata` parameter; public behavior unchanged when not provided). Key equivalences, verified against live `/resolve` responses and resulting cache layout:
   - `etag` = `lfs.sha256` for LFS files, git `blob_id` otherwise (identical to `/resolve`'s `X-Linked-ETag`/`ETag` → identical `blobs/` filenames, so caches stay interoperable with main).
   - `size` = `lfs.size` or blob size.
   - `XetFileData` = (`xetHash` from tree, refresh route rebuilt as `/api/{type}s/{repo}/xet-read-token/{commit}`).
4. `repo_info` is kept as the one ref→commit resolution call. It cannot be replaced by `/tree`: the tree response does not expose the resolved commit hash (no `X-Repo-Commit` header — checked).

## Side benefits

- `allow_patterns`/`ignore_patterns` downloads no longer hit the network for the file list when the tree is cached (verified: filtered download on a cached commit = 0 listing calls).
- The cached index gives a future `snapshot_download` enough information to *verify* a snapshot folder is complete instead of blindly returning it (today's behavior after partial failures). Not implemented — out of scope.

## Caveats

- 3 of 13 `test_snapshot_download.py` tests fail (10 pass): all three simulate offline by patching only the file-download HTTP session, while `repo_info`/`tree` (HfApi client) stay reachable. With pre-fetched metadata the per-file HEAD fallback no longer converts that split-brain into `LocalEntryNotFoundError`. Real offline (everything down) keeps today's behavior — `repo_info` fails first. A mergeable version should catch download errors in `snapshot_download` and degrade gracefully (and the tests should mock at the client level).
- PR #4348 has the same blind spot once its tree path activates.
